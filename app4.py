import gc
import math
import os

import gradio as gr
import torch
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.closed_loop import install_closed_loop

HF_TOKEN = os.environ.get("HF_TOKEN")
if HF_TOKEN:
    login(token=HF_TOKEN)
    print("Logged into Hugging Face")
else:
    print("HF_TOKEN not found in environment variables")

# ============================================================================
# Configuration
# ============================================================================

# Language name -> code. Only languages that have a mean.<code>.pt file for the
# selected model show up in the dropdowns.
ALL_LANGUAGES = {
    "English": "en", "Arabic": "ar", "Tibetan": "bo", "Danish": "da", "German": "de",
    "Spanish": "es", "French": "fr", "Hindi": "hi", "Italian": "it", "Japanese": "ja",
    "Korean": "ko", "Maltese": "mt", "Dutch": "nl", "Norwegian": "no", "Polish": "pl",
    "Portuguese": "pt", "Russian": "ru", "Swedish": "sv", "Turkish": "tr", "Chinese": "zh",
    "Slovak": "sk", "Greek": "el", "Kazakh": "kk", "Swahili": "sw", "Georgian": "ka",
    "Ukrainian": "uk", "Persian": "fa", "Thai": "th", "Indonesian": "id", "Vietnamese": "vi",
    "Czech": "cs", "Romanian": "ro", "Afrikaans": "af", "Urdu": "ur", "Bengali": "bn",
    "Tagalog": "tl",
}

# steering_dir holds the mean.<lang>.pt files made by steering/diffmean.py
MODELS = {
    "Llama-3.1-8B-Instruct": {
        "id": "meta-llama/Llama-3.1-8B-Instruct",
        "steering_dir": "steering/data_llama",
    },
    "Tiny-Aya-Global": {
        "id": "CohereLabs/tiny-aya-global",
        "steering_dir": "steering/data_aya",
    },
}

MODE_STATIC = "Static"
MODE_DYNAMIC = "Dynamic (cosine decay)"
MODE_TWOPHASE = "Two-phase (prompt layers, then generation layers)"
MODE_CLOSED = "Closed-loop (projection control)"

# Default values of the two-phase controls, in the order of the UI components:
# prompt strength, prompt layer start/end, gen start/end strength, gen decay, gen layer start/end
TWOPHASE_DEFAULTS = [1.0, 0, 0, 5.0, 5.0, 50, 20, 20]
# closed-loop controls: goal progress, gain, max push (in gaps), also steer all prompt tokens
CLOSED_DEFAULTS = [0.8, 1.0, 1.0, False]

# ============================================================================
# Global state
# ============================================================================

model = None
tokenizer = None
current_model_name = None
steering_vectors = {}   # cache: (model_name, lang_code) -> [num_layers, hidden_size]
active_hooks = []
current_loop = None     # {"stats": LoopStats, "gaps": {layer: gap}} of the running closed-loop generation


def language_choices(model_name):
    """Names of the languages that have a mean file for this model."""
    steering_dir = MODELS[model_name]["steering_dir"]
    names = [name for name, code in ALL_LANGUAGES.items()
             if os.path.exists(os.path.join(steering_dir, f"mean.{code}.pt"))]
    return names if names else list(ALL_LANGUAGES)


def get_model_layers(model):
    """Get the decoder layers for different model architectures"""
    try:
        return model.model.layers
    except AttributeError:
        return model.model.language_model.layers


def load_model(model_choice):
    """Load the selected model and tokenizer, then refresh the dropdowns and layer sliders"""
    global model, tokenizer, current_model_name, steering_vectors

    model_config = MODELS.get(model_choice)
    if model_config is None:
        return (f"⚠️ Unknown model: {model_choice}",) + (gr.update(),) * 8

    if model is not None and current_model_name != model_choice:
        print(f"Unloading {current_model_name}...")
        remove_steering()
        model = None
        tokenizer = None
        steering_vectors = {}
        gc.collect()
        torch.cuda.empty_cache()

    if model is None:
        print(f"Loading {model_choice}...")
        tokenizer = AutoTokenizer.from_pretrained(model_config["id"])
        model = AutoModelForCausalLM.from_pretrained(
            model_config["id"],
            torch_dtype=torch.float16,   # T4 has no bfloat16
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()
        current_model_name = model_choice

    num_layers = len(get_model_layers(model))
    names = language_choices(model_choice)
    default_layer = min(20, num_layers - 1)
    max_layer = num_layers - 1

    return (
        f"✓ {model_choice} loaded ({num_layers} layers, {len(names)} languages with vectors)",
        gr.update(choices=names, value="English" if "English" in names else names[0]),
        gr.update(choices=names, value=names[1] if len(names) > 1 else names[0]),
        gr.update(maximum=max_layer, value=default_layer),   # layer_start
        gr.update(maximum=max_layer, value=default_layer),   # layer_end
        gr.update(maximum=max_layer),                        # prompt_layer_start
        gr.update(maximum=max_layer),                        # prompt_layer_end
        gr.update(maximum=max_layer, value=default_layer),   # gen_layer_start
        gr.update(maximum=max_layer, value=default_layer),   # gen_layer_end
    )


def load_steering_vector(lang_code, model_name):
    """Load the mean activations of one language: [num_layers, hidden_size]"""
    cache_key = (model_name, lang_code)
    if cache_key in steering_vectors:
        return steering_vectors[cache_key]

    filepath = os.path.join(MODELS[model_name]["steering_dir"], f"mean.{lang_code}.pt")
    if not os.path.exists(filepath):
        return None

    data = torch.load(filepath, map_location="cpu", weights_only=False)
    steering_vectors[cache_key] = data["steering_vectors"].float()
    return steering_vectors[cache_key]


# ============================================================================
# Open-loop DiffMean: separate plans for the prompt phase and the generation phase
# ============================================================================

class DynamicController:
    """Strength of the generation phase at every generated token.

    Cosine decay from start to end over `decay_tokens` tokens, then end strength.
    A constant strength is the special case start == end.
    """

    def __init__(self, start_strength, end_strength, decay_tokens):
        self.start_strength = start_strength
        self.end_strength = end_strength
        self.decay_tokens = decay_tokens
        self.step_idx = 0

    def get_strength(self):
        t = self.step_idx
        if t >= self.decay_tokens:
            return self.end_strength
        factor = 0.5 * (1 + math.cos(math.pi * t / self.decay_tokens))
        return self.end_strength + (self.start_strength - self.end_strength) * factor

    def step(self):
        self.step_idx += 1


def in_range(layer_range, layer_idx):
    return layer_range is not None and layer_range[0] <= layer_idx <= layer_range[1]


def create_phase_hook(v_unit, layer_idx, plan, controller, is_last_gen):
    """Forward hook for one decoder layer.

    prompt phase     = the forward pass over the whole question (seq_len > 1)
    generation phase = every following forward pass, one new token at a time (seq_len == 1)

    - layer in the prompt range: adds prompt_strength to EVERY prompt position
    - layer in the generation range: adds the (decaying) generation strength to every generated token.
      During the prompt pass it also steers the LAST prompt position only, because that position
      produces the first answer token.
    """
    in_prompt = in_range(plan["prompt_layers"], layer_idx)
    in_gen = in_range(plan["gen_layers"], layer_idx)

    def hook(module, inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        is_prefill = hs.dim() == 3 and hs.size(1) > 1

        if is_prefill:
            if in_prompt:
                if plan["prompt_strength"] != 0:
                    hs = hs + plan["prompt_strength"] * v_unit
            elif in_gen and plan["gen_start"] != 0:
                last = hs[:, -1:, :] + plan["gen_start"] * v_unit
                hs = torch.cat([hs[:, :-1, :], last], dim=1)
        else:
            if in_gen:
                strength = controller.get_strength()
                if strength != 0:
                    hs = hs + strength * v_unit
            # the last generation layer advances the token counter once per generated token
            if is_last_gen:
                controller.step()

        if isinstance(output, tuple):
            return (hs,) + tuple(output[1:])
        return hs
    return hook


def remove_steering():
    """Remove all steering hooks"""
    global active_hooks, current_loop
    for handle in active_hooks:
        handle.remove()
    active_hooks = []
    current_loop = None


def apply_diffmean_steering(source_code, target_code, plan):
    """Install hooks that add strength * unit(mean[target] - mean[source]) according to the plan"""
    global active_hooks

    remove_steering()

    source_vec = load_steering_vector(source_code, current_model_name)
    target_vec = load_steering_vector(target_code, current_model_name)
    if source_vec is None:
        return False, f"Steering vector not found for {source_code} ({current_model_name})"
    if target_vec is None:
        return False, f"Steering vector not found for {target_code} ({current_model_name})"

    steering_diff = target_vec - source_vec          # [num_layers, hidden_size]

    layers = get_model_layers(model)
    num_layers = len(layers)
    if steering_diff.shape[0] != num_layers:
        return False, (f"Vectors have {steering_diff.shape[0]} layers but the model has "
                       f"{num_layers}. Wrong steering folder for this model?")

    # validate / clamp the layer ranges (inclusive)
    for name in ("prompt_layers", "gen_layers"):
        layer_range = plan[name]
        if layer_range is None:
            continue
        start, end = int(layer_range[0]), min(int(layer_range[1]), num_layers - 1)
        if start > end:
            return False, f"{name}: start layer ({start}) must not be after end layer ({end})"
        plan[name] = (start, end)

    controller = DynamicController(plan["gen_start"], plan["gen_end"], plan["gen_decay"])

    hooked = set()
    for name in ("prompt_layers", "gen_layers"):
        if plan[name] is not None:
            hooked.update(range(plan[name][0], plan[name][1] + 1))
    last_gen_layer = plan["gen_layers"][1]

    for layer_idx in sorted(hooked):
        v = steering_diff[layer_idx]
        v_unit = v / v.norm(p=2).clamp_min(1e-8)
        p = next(layers[layer_idx].parameters())
        v_unit = v_unit.to(device=p.device, dtype=p.dtype)
        hook = create_phase_hook(v_unit, layer_idx, plan, controller, layer_idx == last_gen_layer)
        active_hooks.append(layers[layer_idx].register_forward_hook(hook))

    return True, f"{source_code} → {target_code}"


def describe_plan(plan):
    """One-line description of what is steered"""
    p = plan["prompt_layers"]
    g = plan["gen_layers"]
    prompt_part = (f"prompt: layers {p[0]}-{p[1]} at {plan['prompt_strength']}"
                   if p is not None and plan["prompt_strength"] != 0 else "prompt: not steered")
    if plan["gen_start"] == plan["gen_end"]:
        gen_strength = f"{plan['gen_start']}"
    else:
        gen_strength = f"{plan['gen_start']} → {plan['gen_end']} over {plan['gen_decay']} tokens"
    return f"{prompt_part} | generation: layers {g[0]}-{g[1]} at {gen_strength}"


# ============================================================================
# Closed-loop DiffMean (projection control)
# ============================================================================

def apply_closed_loop_steering(source_code, target_code, goal, gain, max_push, steer_prompt,
                               layer_start, layer_end):
    """Install the projection controller (see utils/closed_loop.py) on the chosen layers"""
    global active_hooks, current_loop

    remove_steering()

    source_vec = load_steering_vector(source_code, current_model_name)
    target_vec = load_steering_vector(target_code, current_model_name)
    if source_vec is None:
        return False, f"Steering vector not found for {source_code} ({current_model_name})"
    if target_vec is None:
        return False, f"Steering vector not found for {target_code} ({current_model_name})"

    layers = get_model_layers(model)
    num_layers = len(layers)
    if source_vec.shape[0] != num_layers:
        return False, (f"Vectors have {source_vec.shape[0]} layers but the model has "
                       f"{num_layers}. Wrong steering folder for this model?")

    start = int(layer_start)
    end = min(int(layer_end), num_layers - 1)        # inclusive
    if start > end:
        return False, f"Start layer ({start}) must not be after end layer ({end})"

    handles, stats, gaps = install_closed_loop(
        layers, source_vec, target_vec, range(start, end + 1),
        goal=float(goal), gain=float(gain), max_push=float(max_push),
        steer_prompt=bool(steer_prompt))
    active_hooks.extend(handles)
    current_loop = {"stats": stats, "gaps": gaps}

    return True, (f"{source_code} → {target_code} | layers {start}-{end} | goal {goal}, "
                  f"gain {gain}, max push {max_push} gap | "
                  f"{'all prompt tokens' if steer_prompt else 'last prompt token'} + generated tokens")


# ============================================================================
# Generation
# ============================================================================

def build_plan(steering_mode, static_strength, start_strength, end_strength, decay_tokens,
               layer_start, layer_end,
               prompt_strength, prompt_layer_start, prompt_layer_end,
               gen_start_strength, gen_end_strength, gen_decay, gen_layer_start, gen_layer_end):
    if steering_mode == MODE_TWOPHASE:
        return dict(
            prompt_layers=(int(prompt_layer_start), int(prompt_layer_end)),
            prompt_strength=float(prompt_strength),
            gen_layers=(int(gen_layer_start), int(gen_layer_end)),
            gen_start=float(gen_start_strength), gen_end=float(gen_end_strength),
            gen_decay=int(gen_decay),
        )
    layers = (int(layer_start), int(layer_end))
    if steering_mode == MODE_DYNAMIC:
        return dict(prompt_layers=layers, prompt_strength=float(start_strength),
                    gen_layers=layers, gen_start=float(start_strength),
                    gen_end=float(end_strength), gen_decay=int(decay_tokens))
    return dict(prompt_layers=layers, prompt_strength=float(static_strength),
                gen_layers=layers, gen_start=float(static_strength),
                gen_end=float(static_strength), gen_decay=1)


def generate_response(prompt, model_choice, source_lang, target_lang,
                      steering_mode, static_strength,
                      start_strength, end_strength, decay_tokens,
                      layer_start, layer_end,
                      prompt_strength, prompt_layer_start, prompt_layer_end,
                      gen_start_strength, gen_end_strength, gen_decay,
                      gen_layer_start, gen_layer_end,
                      closed_goal, closed_gain, closed_max_push, closed_steer_prompt,
                      max_tokens, temperature, repetition_penalty):
    """Generate a response with optional steering"""
    if model is None:
        return "⚠️ Please load a model first!"
    if current_model_name != model_choice:
        return f"⚠️ Please load {model_choice} first! Currently loaded: {current_model_name}"
    if not prompt or not prompt.strip():
        return "⚠️ Please enter a prompt."

    source_code = ALL_LANGUAGES.get(source_lang)
    target_code = ALL_LANGUAGES.get(target_lang)

    status_msg = ""
    report = ""
    if source_code != target_code:
        if steering_mode == MODE_CLOSED:
            success, info = apply_closed_loop_steering(
                source_code, target_code, closed_goal, closed_gain, closed_max_push,
                closed_steer_prompt, layer_start, layer_end)
            if not success:
                return f"⚠️ {info}"
            status_msg = f"✓ Closed-loop DiffMean {info}"
        else:
            plan = build_plan(steering_mode, static_strength, start_strength, end_strength, decay_tokens,
                              layer_start, layer_end,
                              prompt_strength, prompt_layer_start, prompt_layer_end,
                              gen_start_strength, gen_end_strength, gen_decay,
                              gen_layer_start, gen_layer_end)
            success, info = apply_diffmean_steering(source_code, target_code, plan)
            if not success:
                return f"⚠️ {info}"
            status_msg = f"✓ DiffMean {info}\n{describe_plan(plan)}"

    try:
        messages = [{"role": "user", "content": prompt}]
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)

        # add_special_tokens=False because the chat template already contains BOS
        inputs = tokenizer(input_text, return_tensors="pt", add_special_tokens=False).to(model.device)

        gen_kwargs = dict(
            max_new_tokens=int(max_tokens),
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        )
        if temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=float(temperature))
        else:
            gen_kwargs.update(do_sample=False, temperature=None, top_p=None)   # greedy, like the benchmark
        if repetition_penalty and float(repetition_penalty) != 1.0:
            gen_kwargs["repetition_penalty"] = float(repetition_penalty)

        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)

        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

        if current_loop is not None:
            report = "\n".join(current_loop["stats"].summary(current_loop["gaps"]))
    finally:
        remove_steering()

    if status_msg:
        header = status_msg
        if report:
            header += "\nController report:\n" + report
        return f"{header}\n\n---\n\n{response}"
    return response


# ============================================================================
# Gradio interface
# ============================================================================

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono&display=swap');

* { font-family: 'IBM Plex Sans', sans-serif !important; }

.gradio-container {
    max-width: 900px !important;
    margin: auto !important;
    background: #ffffff !important;
    min-height: 100vh;
}

.main-header {
    text-align: center;
    padding: 2rem 0;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.5rem;
    font-weight: 600;
    letter-spacing: -0.02em;
}

.sub-header { text-align: center; color: #666; margin-bottom: 2rem; font-size: 1rem; }

.control-panel {
    background: #f8f9fa !important;
    border: 1px solid #e9ecef !important;
    border-radius: 16px !important;
    padding: 1.5rem !important;
}

.arrow-indicator { font-size: 2rem; color: #667eea; text-align: center; padding: 1rem; }

textarea, input {
    background: #ffffff !important;
    border: 1px solid #dee2e6 !important;
    border-radius: 8px !important;
    color: #212529 !important;
    font-family: 'IBM Plex Mono', monospace !important;
}

textarea:focus, input:focus {
    border-color: #667eea !important;
    box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2) !important;
}

button.primary {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    color: #000000 !important;
}
button.primary span { color: #000000 !important; }

.output-box {
    background: #f8f9fa !important;
    border: 1px solid #e9ecef !important;
    border-radius: 12px !important;
    min-height: 200px !important;
}

label { color: #495057 !important; font-weight: 500 !important; }
.slider input[type="range"] { accent-color: #667eea !important; }

.params-panel {
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%) !important;
    border: 2px solid #667eea !important;
    border-radius: 12px !important;
    padding: 1rem !important;
    margin-top: 0.5rem !important;
}
"""


def create_interface():
    default_model = "Llama-3.1-8B-Instruct"
    names = language_choices(default_model)

    with gr.Blocks(css=CUSTOM_CSS, theme=gr.themes.Base(primary_hue="purple")) as demo:

        gr.HTML("""
            <h1 class="main-header">🧭 Language Steering</h1>
            <p class="sub-header">Steer LLM responses between languages with DiffMean activation steering</p>
        """)

        # Model selection and loading
        with gr.Row():
            model_choice = gr.Dropdown(
                choices=list(MODELS.keys()), value=default_model,
                label="🤖 Select Model", info="Choose the model to use", scale=2)
            load_btn = gr.Button("🚀 Load Model", variant="primary", scale=1)
            load_status = gr.Textbox(label="Status", interactive=False, scale=3)

        gr.HTML("<hr style='border-color: rgba(0,0,0,0.1); margin: 1.5rem 0;'>")

        # Language selection
        with gr.Row(elem_classes="control-panel"):
            with gr.Column(scale=2):
                source_lang = gr.Dropdown(
                    choices=names, value="English" if "English" in names else names[0],
                    label="📤 Source Language", info="Language of your prompt")

            gr.HTML('<div class="arrow-indicator">→</div>')

            with gr.Column(scale=2):
                target_lang = gr.Dropdown(
                    choices=names, value="Maltese" if "Maltese" in names else names[-1],
                    label="📥 Target Language", info="Desired response language")

        # DiffMean parameters
        with gr.Group(elem_classes="params-panel"):
            gr.Markdown("### ⚡ DiffMean Parameters")

            steering_mode = gr.Radio(
                choices=[MODE_STATIC, MODE_DYNAMIC, MODE_TWOPHASE, MODE_CLOSED], value=MODE_DYNAMIC,
                label="Steering mode",
                info="Static: constant strength. Dynamic: strong first, then decays. "
                     "Two-phase: separate layers/strengths for prompt and answer. "
                     "Closed-loop: measures each token and pushes only the missing distance")

            # --- static ---
            with gr.Group(visible=False) as static_group:
                static_strength = gr.Slider(
                    minimum=0.0, maximum=20.0, value=5.0, step=0.5,
                    label="Steering Strength",
                    info="Added vector has unit length x this value")

            # --- dynamic ---
            with gr.Group(visible=True) as dynamic_group:
                with gr.Row():
                    start_strength = gr.Slider(
                        minimum=0.0, maximum=20.0, value=10.0, step=0.5,
                        label="Starting Strength",
                        info="Used for the prompt and the first generated token")
                    end_strength = gr.Slider(
                        minimum=0.0, maximum=20.0, value=5.0, step=0.5,
                        label="Ending Strength",
                        info="Strength once the decay is finished")
                decay_tokens = gr.Slider(
                    minimum=1, maximum=512, value=50, step=1,
                    label="Decay Tokens",
                    info="Number of generated tokens over which the strength decays (cosine curve)")

            # --- closed-loop ---
            with gr.Group(visible=False) as closed_group:
                gr.Markdown("**Closed-loop:** at every steered layer the controller measures how far each "
                            "token's hidden state has moved from the source-language mean towards the "
                            "target-language mean, and pushes only the missing part. Once the model writes "
                            "in the target language the push fades out by itself, so there is no strength "
                            "or decay to tune.")
                closed_goal = gr.Slider(
                    minimum=0.2, maximum=1.5, value=CLOSED_DEFAULTS[0], step=0.05,
                    label="Goal Progress",
                    info="0 = source-language mean position, 1 = target-language mean position. "
                         "Too high can overshoot into another language")
                with gr.Accordion("Advanced", open=False):
                    closed_gain = gr.Slider(
                        minimum=0.1, maximum=1.0, value=CLOSED_DEFAULTS[1], step=0.05,
                        label="Gain", info="Fraction of the missing distance corrected per step")
                    closed_max_push = gr.Slider(
                        minimum=0.1, maximum=2.0, value=CLOSED_DEFAULTS[2], step=0.05,
                        label="Max Push (in gaps)",
                        info="Largest push per token per layer, as a fraction of the source-target distance")
                    closed_steer_prompt = gr.Checkbox(
                        value=CLOSED_DEFAULTS[3], label="Also steer all prompt tokens",
                        info="Off = only the last prompt token (which produces the first answer token) "
                             "and the generated tokens are controlled")

            # --- layers shared by static, dynamic and closed-loop ---
            with gr.Row(visible=True) as shared_layers_row:
                layer_start = gr.Slider(
                    minimum=0, maximum=31, value=20, step=1,
                    label="🔽 Start Layer", info="First layer to steer")
                layer_end = gr.Slider(
                    minimum=0, maximum=31, value=20, step=1,
                    label="🔼 End Layer", info="Last layer to steer (inclusive)")

            # --- two-phase ---
            with gr.Group(visible=False) as twophase_group:
                gr.Markdown("**Phase 1 - prompt:** steering while the question is being read "
                            "(also shapes the KV cache of the prompt). Strength 0 = off.")
                prompt_strength = gr.Slider(
                    minimum=0.0, maximum=20.0, value=TWOPHASE_DEFAULTS[0], step=0.1,
                    label="Prompt Strength", info="Early layers need low strengths (about 0.5 to 2)")
                with gr.Row():
                    prompt_layer_start = gr.Slider(
                        minimum=0, maximum=31, value=TWOPHASE_DEFAULTS[1], step=1,
                        label="Prompt Start Layer")
                    prompt_layer_end = gr.Slider(
                        minimum=0, maximum=31, value=TWOPHASE_DEFAULTS[2], step=1,
                        label="Prompt End Layer", info="inclusive")

                gr.Markdown("**Phase 2 - generation:** steering while the answer is written. "
                            "Start and end strength differ = cosine decay. The last prompt token "
                            "is also steered here, since it produces the first answer token.")
                with gr.Row():
                    gen_start_strength = gr.Slider(
                        minimum=0.0, maximum=20.0, value=TWOPHASE_DEFAULTS[3], step=0.5,
                        label="Generation Starting Strength")
                    gen_end_strength = gr.Slider(
                        minimum=0.0, maximum=20.0, value=TWOPHASE_DEFAULTS[4], step=0.5,
                        label="Generation Ending Strength")
                gen_decay = gr.Slider(
                    minimum=1, maximum=512, value=TWOPHASE_DEFAULTS[5], step=1,
                    label="Generation Decay Tokens")
                with gr.Row():
                    gen_layer_start = gr.Slider(
                        minimum=0, maximum=31, value=TWOPHASE_DEFAULTS[6], step=1,
                        label="Generation Start Layer")
                    gen_layer_end = gr.Slider(
                        minimum=0, maximum=31, value=TWOPHASE_DEFAULTS[7], step=1,
                        label="Generation End Layer", info="inclusive")

        # Generation controls
        with gr.Row():
            max_tokens = gr.Slider(
                minimum=64, maximum=1024, value=256, step=32,
                label="📏 Max Tokens", info="Maximum response length")
            temperature = gr.Slider(
                minimum=0.0, maximum=1.5, value=0.0, step=0.1,
                label="🌡️ Temperature", info="0 = greedy (deterministic, as in the benchmark)")
            repetition_penalty = gr.Slider(
                minimum=1.0, maximum=1.5, value=1.0, step=0.05,
                label="🔁 Repetition Penalty", info="1.0 = off (your dynamic script used 1.15)")

        # Input / output
        prompt = gr.Textbox(label="💬 Your Prompt", placeholder="Enter your message here...", lines=3)
        generate_btn = gr.Button("✨ Generate", variant="primary", size="lg")
        output = gr.Textbox(label="🤖 Response", lines=12, elem_classes="output-box")

        all_inputs = [prompt, model_choice, source_lang, target_lang,
                      steering_mode, static_strength,
                      start_strength, end_strength, decay_tokens,
                      layer_start, layer_end,
                      prompt_strength, prompt_layer_start, prompt_layer_end,
                      gen_start_strength, gen_end_strength, gen_decay,
                      gen_layer_start, gen_layer_end,
                      closed_goal, closed_gain, closed_max_push, closed_steer_prompt,
                      max_tokens, temperature, repetition_penalty]

        def example(q, lang, mode, rep=1.0):
            return ([q, "Llama-3.1-8B-Instruct", "English", lang, mode, 5.0, 10.0, 5.0, 50, 20, 20]
                    + TWOPHASE_DEFAULTS + CLOSED_DEFAULTS + [256, 0.0, rep])

        gr.Examples(
            examples=[
                example("What is the capital of France?", "Maltese", MODE_CLOSED),
                example("Explain quantum computing in simple terms.", "Greek", MODE_DYNAMIC, 1.15),
                example("How can I improve my time management skills?", "Maltese", MODE_TWOPHASE),
            ],
            inputs=all_inputs,
        )

        gr.HTML("""
            <div style="text-align: center; margin-top: 2rem; color: #666; font-size: 0.85rem;">
                <p>Built with 🤗 Transformers & Gradio</p>
                <p>Models: meta-llama/Llama-3.1-8B-Instruct | CohereLabs/tiny-aya-global</p>
                <p><b>DiffMean:</b> adds the (target − source) mean-activation direction to the hidden states</p>
            </div>
        """)

        # Show only the controls of the selected mode
        def toggle_mode(mode):
            return (gr.update(visible=(mode == MODE_STATIC)),
                    gr.update(visible=(mode == MODE_DYNAMIC)),
                    gr.update(visible=(mode == MODE_CLOSED)),
                    gr.update(visible=(mode != MODE_TWOPHASE)),
                    gr.update(visible=(mode == MODE_TWOPHASE)))

        steering_mode.change(fn=toggle_mode, inputs=[steering_mode],
                             outputs=[static_group, dynamic_group, closed_group,
                                      shared_layers_row, twophase_group])

        # Event handlers
        load_btn.click(
            fn=load_model, inputs=[model_choice],
            outputs=[load_status, source_lang, target_lang, layer_start, layer_end,
                     prompt_layer_start, prompt_layer_end, gen_layer_start, gen_layer_end])
        generate_btn.click(fn=generate_response, inputs=all_inputs, outputs=output)
        prompt.submit(fn=generate_response, inputs=all_inputs, outputs=output)

    return demo


if __name__ == "__main__":
    demo = create_interface()
    # share=True gives a public link, which is needed on Kaggle (localhost is not reachable)
    demo.queue().launch(share=True)

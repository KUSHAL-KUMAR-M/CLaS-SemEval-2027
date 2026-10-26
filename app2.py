import gc
import os

import gradio as gr
import torch
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer

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

# ============================================================================
# Global state
# ============================================================================

model = None
tokenizer = None
current_model_name = None
steering_vectors = {}   # cache: (model_name, lang_code) -> [num_layers, hidden_size]
active_hooks = []


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
        return (f"⚠️ Unknown model: {model_choice}",
                gr.update(), gr.update(), gr.update(), gr.update())

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

    return (
        f"✓ {model_choice} loaded ({num_layers} layers, {len(names)} languages with vectors)",
        gr.update(choices=names, value="English" if "English" in names else names[0]),
        gr.update(choices=names, value=names[1] if len(names) > 1 else names[0]),
        gr.update(maximum=num_layers - 1, value=default_layer),
        gr.update(maximum=num_layers - 1, value=default_layer),
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
# Diffmean steering
# ============================================================================

def create_steering_hook(vec):
    """Forward hook that adds a fixed vector to a decoder layer's output."""
    def hook(module, inputs, output):
        if isinstance(output, tuple):
            return (output[0] + vec,) + tuple(output[1:])
        return output + vec
    return hook


def create_dynamic_diffmean_hook(source_direction, target_direction, scale=1.0):
    """
    Dynamic, state-dependent DiffMean replacement.

    For each token hidden state h:
        h' = h + scale * (P_target(h) - P_source(h))

    P_direction(h) is the projection of the current hidden state onto that
    language's normalized mean-activation direction. Unlike static DiffMean,
    the amount removed/added changes with each token's current hidden state.
    """
    def hook(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output

        # Directions have shape [hidden_size]; hidden has shape
        # [batch, sequence, hidden_size] (or compatible leading dimensions).
        src_dir = source_direction.to(device=hidden.device, dtype=hidden.dtype)
        tgt_dir = target_direction.to(device=hidden.device, dtype=hidden.dtype)

        source_component = torch.sum(hidden * src_dir, dim=-1, keepdim=True) * src_dir
        target_component = torch.sum(hidden * tgt_dir, dim=-1, keepdim=True) * tgt_dir
        steered = hidden + scale * (target_component - source_component)

        if isinstance(output, tuple):
            return (steered,) + tuple(output[1:])
        return steered
    return hook


def remove_steering():
    """Remove all steering hooks"""
    global active_hooks
    for handle in active_hooks:
        handle.remove()
    active_hooks = []


def apply_diffmean_steering(source_code, target_code, strength, layer_start, layer_end,
                            steering_mode="Static DiffMean"):
    """Apply static DiffMean or dynamic state-dependent source/target replacement."""
    global active_hooks

    remove_steering()

    source_vec = load_steering_vector(source_code, current_model_name)
    target_vec = load_steering_vector(target_code, current_model_name)
    if source_vec is None:
        return False, f"Steering vector not found for {source_code} ({current_model_name})"
    if target_vec is None:
        return False, f"Steering vector not found for {target_code} ({current_model_name})"

    layers = get_model_layers(model)
    if source_vec.shape[0] != len(layers) or target_vec.shape[0] != len(layers):
        return False, (
            f"Vector layer counts are source={source_vec.shape[0]}, target={target_vec.shape[0]}, "
            f"but the model has {len(layers)} layers. Wrong steering folder for this model?"
        )

    start = int(layer_start)
    end = min(int(layer_end), len(layers) - 1)  # inclusive
    if start < 0 or start > end:
        return False, f"Invalid layer range: start={start}, end={end}"

    for layer_idx in range(start, end + 1):
        p = next(layers[layer_idx].parameters())

        if steering_mode == "Dynamic DiffMean":
            # Normalize each language mean separately to define projection directions.
            src_dir = source_vec[layer_idx]
            tgt_dir = target_vec[layer_idx]
            src_dir = src_dir / (src_dir.norm(p=2) + 1e-8)
            tgt_dir = tgt_dir / (tgt_dir.norm(p=2) + 1e-8)

            # Keep the existing GUI strength scale intuitive: strength=5.0
            # corresponds to scale=1.0 in h' = h + scale*(P_t(h)-P_s(h)).
            scale = float(strength) / 5.0
            hook = create_dynamic_diffmean_hook(
                src_dir.to(device=p.device, dtype=p.dtype),
                tgt_dir.to(device=p.device, dtype=p.dtype),
                scale=scale,
            )
        else:
            # Original static DiffMean behavior: normalized target-minus-source mean.
            v = target_vec[layer_idx] - source_vec[layer_idx]
            v = v / (v.norm(p=2) + 1e-8)
            vec = (float(strength) * v).to(device=p.device, dtype=p.dtype)
            hook = create_steering_hook(vec)

        active_hooks.append(layers[layer_idx].register_forward_hook(hook))

    return True, (
        f"✓ {steering_mode}: {source_code} → {target_code} "
        f"(strength={strength}, layers={start}-{end})"
    )


# ============================================================================
# Generation
# ============================================================================

def generate_response(prompt, model_choice, source_lang, target_lang,
                      steering_mode, strength, layer_start, layer_end, max_tokens, temperature):
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
    if source_code != target_code:
        success, status_msg = apply_diffmean_steering(
            source_code, target_code, strength, layer_start, layer_end, steering_mode)
        if not success:
            return f"⚠️ {status_msg}"

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

        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)

        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    finally:
        remove_steering()

    if status_msg:
        return f"{status_msg}\n\n---\n\n{response}"
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
                    choices=names, value="Greek" if "Greek" in names else names[-1],
                    label="📥 Target Language", info="Desired response language")

        # DiffMean parameters
        with gr.Group(elem_classes="params-panel"):
            gr.Markdown("### ⚡ DiffMean Parameters")
            steering_mode = gr.Dropdown(
                choices=["Static DiffMean", "Dynamic DiffMean"],
                value="Static DiffMean",
                label="Steering Mode",
                info="Static adds a fixed mean-difference direction; Dynamic replaces source/target components of each current token state.")
            diffmean_strength = gr.Slider(
                minimum=0.0, maximum=20.0, value=5.0, step=0.5,
                label="Steering Strength",
                info="Added vector has unit length x this value. Higher = stronger language shift, "
                     "but too high breaks fluency")
            with gr.Row():
                layer_start = gr.Slider(
                    minimum=0, maximum=31, value=20, step=1,
                    label="🔽 Start Layer", info="First layer to steer")
                layer_end = gr.Slider(
                    minimum=0, maximum=31, value=20, step=1,
                    label="🔼 End Layer", info="Last layer to steer (inclusive)")

        # Generation controls
        with gr.Row():
            max_tokens = gr.Slider(
                minimum=64, maximum=1024, value=256, step=32,
                label="📏 Max Tokens", info="Maximum response length")
            temperature = gr.Slider(
                minimum=0.0, maximum=1.5, value=0.0, step=0.1,
                label="🌡️ Temperature", info="0 = greedy (deterministic, as in the benchmark)")

        # Input / output
        prompt = gr.Textbox(label="💬 Your Prompt", placeholder="Enter your message here...", lines=3)
        generate_btn = gr.Button("✨ Generate", variant="primary", size="lg")
        output = gr.Textbox(label="🤖 Response", lines=10, elem_classes="output-box")

        all_inputs = [prompt, model_choice, source_lang, target_lang,
                      steering_mode, diffmean_strength, layer_start, layer_end, max_tokens, temperature]

        gr.Examples(
            examples=[
                ["What is the capital of France?", "Llama-3.1-8B-Instruct", "English", "Greek", "Static DiffMean", 5.0, 20, 20, 256, 0.0],
                ["Explain quantum computing in simple terms.", "Llama-3.1-8B-Instruct", "English", "Turkish", "Dynamic DiffMean", 5.0, 20, 20, 256, 0.0],
                ["How can I improve my time management skills?", "Llama-3.1-8B-Instruct", "English", "Maltese", "Dynamic DiffMean", 5.0, 20, 20, 256, 0.0],
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

        # Event handlers
        load_btn.click(
            fn=load_model, inputs=[model_choice],
            outputs=[load_status, source_lang, target_lang, layer_start, layer_end])
        generate_btn.click(fn=generate_response, inputs=all_inputs, outputs=output)
        prompt.submit(fn=generate_response, inputs=all_inputs, outputs=output)

    return demo


if __name__ == "__main__":
    demo = create_interface()
    # share=True gives a public link, which is needed on Kaggle (localhost is not reachable)
    demo.queue().launch(share=True)
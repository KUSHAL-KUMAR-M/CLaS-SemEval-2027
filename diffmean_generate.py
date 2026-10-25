import argparse
import json
import os

# apply_model() sends a Python function to the vLLM workers, which needs this flag
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

import torch
from utils.utils import get_test_questions


# ----------------------------------------------------------------------------
# vLLM backend: hooks are installed INSIDE the worker processes via apply_model
# ----------------------------------------------------------------------------
def make_worker_functions():
    """Functions that run inside the vLLM workers (they receive the nn.Module)."""

    def get_model_layers(model):
        try:
            return model.model.layers                    # Llama / Aya style
        except AttributeError:
            try:
                return model.language_model.model.layers  # Gemma-3 style
            except AttributeError:
                raise RuntimeError("Could not access model layers. Model architecture may not be supported.")

    def install_hooks(model):
        """Wrap every layer's forward once. Steering stays off until set_steering() is called."""
        layers = get_model_layers(model)
        for layer in layers:
            if getattr(layer, "_steering_installed", False):
                continue
            original = layer.forward

            def forward_hook(*args, _original=original, _layer=layer, **kwargs):
                outputs = _original(*args, **kwargs)
                vec = _layer._steering_vec
                if vec is None:
                    return outputs
                # vLLM layers return (hidden_states, residual); the residual stream is their sum,
                # so adding to hidden_states shifts the stream.
                if isinstance(outputs, (tuple, list)):
                    return (outputs[0] + vec,) + tuple(outputs[1:])
                return outputs + vec

            layer.forward = forward_hook
            layer._steering_installed = True
            layer._steering_vec = None
        return len(layers)

    def make_set_steering(directions, layer_start, layer_end):
        """directions: [num_layers, hidden_size] already scaled by the strength (None = steering off)"""
        def set_steering(model):
            layers = get_model_layers(model)
            for idx, layer in enumerate(layers):
                if directions is not None and layer_start <= idx < layer_end:
                    p = next(layer.parameters())
                    layer._steering_vec = directions[idx].to(device=p.device, dtype=p.dtype)
                else:
                    layer._steering_vec = None
            return sum(layer._steering_vec is not None for layer in layers)
        return set_steering

    return install_hooks, make_set_steering


class VLLMSteerer:
    def __init__(self, args):
        from vllm import LLM, SamplingParams

        self.model = LLM(
            model=args.model,
            tensor_parallel_size=torch.cuda.device_count(),
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            trust_remote_code=True,
            enforce_eager=True,              # needed so the patched forwards run in Python
            dtype="float16",                 # T4 has no bfloat16
            enable_prefix_caching=False,     # needed so cached KV is not reused across different steering
        )
        self.install_hooks, self.make_set_steering = make_worker_functions()
        print("Wrapped layers per worker:", self.model.apply_model(self.install_hooks))

        tokenizer = self.model.get_tokenizer()
        stop_token_ids = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else None
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=args.max_new_tokens,
            repetition_penalty=1.0,
            stop_token_ids=stop_token_ids,
            skip_special_tokens=True,
        )

    def generate(self, prompts, directions, layer_start, layer_end):
        active = self.model.apply_model(self.make_set_steering(directions, layer_start, layer_end))
        print(f"Steered layers per worker: {active}")

        messages = [[{"role": "user", "content": p}] for p in prompts]
        try:
            outputs = self.model.chat(messages, sampling_params=self.sampling_params)
        finally:
            self.model.apply_model(self.make_set_steering(None, 0, 0))  # steering off again
        return [o.outputs[0].text.strip() for o in outputs]


# ----------------------------------------------------------------------------
# HF transformers backend (slower fallback): normal forward hooks
# ----------------------------------------------------------------------------
class HFSteerer:
    def __init__(self, args):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.batch_size = args.batch_size
        self.max_new_tokens = args.max_new_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(args.model)
        self.tokenizer.padding_side = "left"   # needed for batched generation
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
        self.model.eval()

    def get_layers(self):
        try:
            return self.model.model.layers
        except AttributeError:
            return self.model.model.language_model.layers

    def generate(self, prompts, directions, layer_start, layer_end):
        layers = self.get_layers()
        handles = []

        def make_hook(vec):
            def hook(module, inputs, output):
                if isinstance(output, tuple):
                    return (output[0] + vec,) + tuple(output[1:])
                return output + vec
            return hook

        try:
            for idx in range(layer_start, layer_end):
                p = next(layers[idx].parameters())
                vec = directions[idx].to(device=p.device, dtype=p.dtype)
                handles.append(layers[idx].register_forward_hook(make_hook(vec)))
            print(f"Steered layers: {layer_start} to {layer_end - 1}")
            return self.generate_batched(prompts)
        finally:
            for h in handles:
                h.remove()

    def generate_batched(self, prompts):
        texts = []
        for prompt in prompts:
            try:
                texts.append(self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True))
            except Exception:
                texts.append(prompt)

        # Longest prompts first -> less padding. Answers are put back in original order.
        lengths = [len(self.tokenizer(t, add_special_tokens=False).input_ids) for t in texts]
        order = sorted(range(len(texts)), key=lambda i: -lengths[i])
        responses = [None] * len(texts)

        for s in range(0, len(order), self.batch_size):
            idx = order[s:s + self.batch_size]
            # add_special_tokens=False because the chat template already contains BOS
            enc = self.tokenizer([texts[i] for i in idx], return_tensors="pt",
                                 padding=True, add_special_tokens=False).to(self.model.device)
            with torch.no_grad():
                out = self.model.generate(
                    **enc, max_new_tokens=self.max_new_tokens, do_sample=False,
                    temperature=None, top_p=None, pad_token_id=self.tokenizer.pad_token_id)
            gen = out[:, enc.input_ids.shape[1]:]
            for j, i in enumerate(idx):
                responses[i] = self.tokenizer.decode(gen[j], skip_special_tokens=True).strip()
            print(f"  batch {s // self.batch_size + 1}/{-(-len(order) // self.batch_size)} done")
        return responses


# ----------------------------------------------------------------------------
# Shared code
# ----------------------------------------------------------------------------
def load_steering_vectors(steering_dir, languages):
    """Load mean activations (saved by steering/diffmean.py) for specified languages

    Returns:
        Dictionary mapping language codes to tensors of shape [num_layers, hidden_size]
    """
    steering_data = {}

    for lang in languages:
        filepath = os.path.join(steering_dir, f"mean.{lang}.pt")

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Steering vector file not found for '{lang}': looked for {filepath}")

        data = torch.load(filepath, map_location='cpu', weights_only=False)
        steering_data[lang] = data['steering_vectors'].float()
        print(f"Loaded steering vectors for {lang}: {data['steering_vectors'].shape}")

    return steering_data


def run_steering_experiment(steerer, test_questions, source_lang, target_lang,
                            steering_data, args):
    """Run a single steering experiment"""

    print(f"\n{'='*60}")
    print(f"Steering Experiment: {source_lang} → {target_lang}")
    print(f"Steering strength: {args.steering_strength}")
    print(f"{'='*60}")

    assert source_lang in steering_data, f"Source language {source_lang} not in steering_data!"
    assert target_lang in steering_data, f"Target language {target_lang} not in steering_data!"

    if source_lang not in test_questions:
        print(f"Warning: No test questions found for {source_lang}")
        print(f"Available languages in test_questions: {list(test_questions.keys())}")
        return None

    # Compute difference in means: target - source
    steering_diff = steering_data[target_lang] - steering_data[source_lang]
    print(f"Steering diff shape: {steering_diff.shape}")

    # Unit-length direction per layer, scaled by the strength
    directions = torch.nn.functional.normalize(steering_diff, dim=-1) * args.steering_strength

    num_layers = steering_diff.shape[0]
    layer_start = args.layer_start
    layer_end = min(args.layer_end, num_layers)
    if not (0 <= layer_start < layer_end):
        raise ValueError(f"Invalid layer range [{layer_start}, {layer_end}) for {num_layers} layers")

    test_prompts = test_questions[source_lang]
    print(f"Running batched inference on {len(test_prompts)} questions...")
    responses = steerer.generate(test_prompts, directions, layer_start, layer_end)

    all_results = []
    for i, (response, original_prompt) in enumerate(zip(responses, test_prompts)):
        all_results.append({
            "question_idx": i,
            "input": original_prompt,
            "output": response,
            "original_prompt": original_prompt
        })

        if i < 3:  # Print first 3 for verification
            print(f"Q{i+1}: {original_prompt[:50]}...")
            print(f"R{i+1}: {response[:100]}...\n")

    results = {
        "source_language": source_lang,
        "target_language": target_lang,
        "model": args.model,
        "method": f"diffmean:alpha={args.steering_strength}",
        "track": args.track if args.track else "custom",
        "steering_strength": args.steering_strength,
        "layer_range": f"{layer_start}-{layer_end - 1}",
        "results": all_results
    }

    return results


def check_experiment_exists(source_lang, target_lang, output_dir, steering_strength):
    """Check if experiment results already exist"""
    strength_str = f"strength_{steering_strength}".replace('.', '_')
    output_subdir = os.path.join(output_dir, strength_str)
    output_file = os.path.join(output_subdir, f"{source_lang}_to_{target_lang}.json")
    return os.path.exists(output_file)


def save_results(results, output_dir, steering_strength):
    """Save experiment results to JSON file"""
    strength_str = f"strength_{steering_strength}".replace('.', '_')
    output_subdir = os.path.join(output_dir, strength_str)
    os.makedirs(output_subdir, exist_ok=True)

    source_lang = results["source_language"]
    target_lang = results["target_language"]
    output_file = os.path.join(output_subdir, f"{source_lang}_to_{target_lang}.json")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"Saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Apply DiffMean steering vectors and generate")
    parser.add_argument("-m", "--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--backend", type=str, default="vllm", choices=["vllm", "hf"],
                        help="vllm (fast) or hf (slower fallback)")
    parser.add_argument("--steering_dir", type=str, default="steering/data_llama",
                        help="Directory containing mean.{lang}.pt files")
    parser.add_argument("--output", type=str, default="steering_results")
    parser.add_argument("--track", type=str, default=None, choices=["t1", "t2", "t3"],
                        help="Run all official language pairs of this track")
    parser.add_argument("--t3_target", type=str, default="ja", help="Target language for track t3")
    parser.add_argument("--batch_mode", action='store_true', help="Run all language combinations")
    parser.add_argument("--source_lang", type=str, default="en", help="Source language (2-letter code)")
    parser.add_argument("--target_lang", type=str, default="fr", help="Target language (2-letter code)")
    parser.add_argument("--pairs", nargs='+', default=None,
                        help="Specific pairs to run, e.g. en:el en:mt")
    parser.add_argument("--languages", nargs='+', default=None,
                        help="Languages for --batch_mode (default: every mean.*.pt in steering_dir)")
    parser.add_argument("--steering_strength", type=float, default=5.0,
                        help="Scaling factor for steering vector (alpha)")
    parser.add_argument("--layer_start", type=int, default=20,
                        help="First layer to apply steering")
    parser.add_argument("--layer_end", type=int, default=21,
                        help="Last layer to apply steering (exclusive)")
    parser.add_argument("--k_questions", type=int, default=70)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=16, help="hf backend only")
    parser.add_argument("--max_model_len", type=int, default=4096, help="vllm backend only")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90, help="vllm backend only")
    parser.add_argument("--skip_existing", action='store_true',
                        help="Skip experiments that already have results")

    args = parser.parse_args()

    # Decide which language pairs to run
    if args.pairs:
        pairs = [tuple(p.split(":")) for p in args.pairs]
    elif args.track:
        from clas import languages as L
        pairs = [tuple(p) for p in L.pairs(args.track, args.t3_target)]
    elif args.batch_mode:
        languages = args.languages
        if languages is None:
            languages = sorted(f.split(".")[1] for f in os.listdir(args.steering_dir)
                               if f.startswith("mean.") and f.endswith(".pt"))
        pairs = [(s, t) for s in languages for t in languages if s != t]
    else:
        pairs = [(args.source_lang, args.target_lang)]

    languages_to_load = sorted({lang for pair in pairs for lang in pair})
    print(f"Running {len(pairs)} pair(s). Loading steering vectors for: {languages_to_load}")

    steering_data = load_steering_vectors(args.steering_dir, languages_to_load)
    test_questions = get_test_questions(k=args.k_questions)

    # Initialize model
    if args.backend == "vllm":
        steerer = VLLMSteerer(args)
    else:
        steerer = HFSteerer(args)

    print(f"Applying steering to layers {args.layer_start} to {args.layer_end - 1}")

    for counter, (source_lang, target_lang) in enumerate(pairs, 1):
        if args.skip_existing and check_experiment_exists(
            source_lang, target_lang, args.output, args.steering_strength
        ):
            print(f"[{counter}/{len(pairs)}] Skipping existing: {source_lang} → {target_lang}")
            continue

        print(f"\n[{counter}/{len(pairs)}] Processing...")

        try:
            results = run_steering_experiment(
                steerer, test_questions, source_lang, target_lang, steering_data, args
            )
            if results is not None:
                save_results(results, args.output, args.steering_strength)

        except Exception as e:
            print(f"Error in experiment {source_lang} → {target_lang}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\nGeneration complete!")


if __name__ == "__main__":
    main()

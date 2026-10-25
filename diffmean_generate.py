import argparse
import json
import os

# apply_model() sends a Python function to the vLLM workers, which needs this flag
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

import torch
from vllm import LLM, SamplingParams
from utils.utils import get_test_questions


def make_worker_functions():
    """Functions that run INSIDE the vLLM worker processes (they receive the nn.Module)."""

    def get_model_layers(model):
        """Get model layers with compatibility for different model architectures"""
        try:
            # Llama / Aya style models
            return model.model.layers
        except AttributeError:
            try:
                # Gemma-3 style models
                return model.language_model.model.layers
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
        """directions: [num_layers, hidden_size], already scaled by the strength (None = steering off)"""
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


def load_steering_vectors(steering_dir, languages):
    """Load mean activations (saved by diffmean.py) for specified languages

    Args:
        steering_dir: Directory containing mean.{lang}.pt files
        languages: List of 2-letter language codes

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


def create_chat_messages(prompts):
    """Create simple chat messages for steering experiments"""
    return [[{"role": "user", "content": prompt}] for prompt in prompts]


def run_steering_experiment(model, sampling_params, test_questions, source_lang, target_lang,
                            steering_data, steering_strength, layer_start, layer_end,
                            make_set_steering, model_name):
    """Run a single steering experiment"""

    print(f"\n{'='*60}")
    print(f"Steering Experiment: {source_lang} → {target_lang}")
    print(f"Steering strength: {steering_strength}")
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
    directions = torch.nn.functional.normalize(steering_diff, dim=-1) * steering_strength

    # Apply steering inside the workers
    active = model.apply_model(make_set_steering(directions, layer_start, layer_end))
    print(f"Steered layers per worker: {active}")

    test_prompts = test_questions[source_lang]
    messages_batch = create_chat_messages(test_prompts)
    print(f"Running batched inference on {len(messages_batch)} questions...")

    try:
        # Generate all responses in one batch using chat format
        outputs = model.chat(messages_batch, sampling_params=sampling_params)
    finally:
        # Switch steering off again
        model.apply_model(make_set_steering(None, 0, 0))

    # Process results
    all_results = []
    for i, (output, original_prompt) in enumerate(zip(outputs, test_prompts)):
        response = output.outputs[0].text.strip()

        all_results.append({
            "question_idx": i,
            "input": original_prompt,
            "output": response
        })

        if i < 3:  # Print first 3 for verification
            print(f"Q{i+1}: {original_prompt[:50]}...")
            print(f"R{i+1}: {response[:100]}...\n")

    results = {
        "source_language": source_lang,
        "target_language": target_lang,
        "steering_strength": steering_strength,
        "layer_range": f"{layer_start}-{layer_end - 1}",
        "model": model_name,
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
    parser = argparse.ArgumentParser(description="Apply steering vectors to language models (vLLM)")
    parser.add_argument("-m", "--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--steering_dir", type=str, default="data_llama",
                        help="Directory containing mean.{lang}.pt files")
    parser.add_argument("--output", type=str, default="steering_results")
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
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--skip_existing", action='store_true',
                        help="Skip experiments that already have results")

    args = parser.parse_args()

    # Decide which language pairs to run
    if args.pairs:
        pairs = [tuple(p.split(":")) for p in args.pairs]
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
    model = LLM(
        model=args.model,
        tensor_parallel_size=torch.cuda.device_count(),
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        enforce_eager=True,              # needed so the patched forwards run in Python
        dtype="float16",                 # T4 has no bfloat16
        enable_prefix_caching=False,     # needed so cached KV is not reused across different steering
    )

    # Wrap the decoder layers inside the workers (steering stays off for now)
    install_hooks, make_set_steering = make_worker_functions()
    print("Wrapped layers per worker:", model.apply_model(install_hooks))

    # Get model-specific stop tokens
    tokenizer = model.get_tokenizer()
    stop_token_ids = []
    if tokenizer.eos_token_id is not None:
        stop_token_ids.append(tokenizer.eos_token_id)

    print(f"Using stop tokens: {stop_token_ids}")

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_new_tokens,
        repetition_penalty=1.0,
        stop_token_ids=stop_token_ids if stop_token_ids else None,
        skip_special_tokens=True
    )

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
                model, sampling_params, test_questions,
                source_lang, target_lang,
                steering_data, args.steering_strength,
                args.layer_start, args.layer_end,
                make_set_steering, args.model
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

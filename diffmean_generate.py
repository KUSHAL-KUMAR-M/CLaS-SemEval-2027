import argparse
import json
import sys
from pathlib import Path
from types import MethodType

import torch
from vllm import LLM, SamplingParams

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

from utils.utils import get_test_questions


def get_model_layers(model):
    """Locate decoder layers for supported model architectures."""
    root = model.llm_engine.model_executor.driver_worker.model_runner.model

    candidates = [
        lambda: root.model.layers,
        lambda: root.model.language_model.model.layers,
    ]

    for candidate in candidates:
        try:
            return candidate()
        except AttributeError:
            continue

    raise RuntimeError(
        "Could not locate decoder layers in this vLLM version. "
        "Inspect the model architecture and vLLM worker structure."
    )


def load_mean_activations(data_dir, languages):
    """Load per-language mean activation tensors saved by diffmean.py."""
    means = {}

    for lang in languages:
        path = data_dir / f"mean.{lang}.pt"

        if not path.exists():
            raise FileNotFoundError(
                f"Missing mean activation file: {path}\n"
                "Run steering/diffmean.py for this language first."
            )

        data = torch.load(path, map_location="cpu", weights_only=False)

        if "steering_vectors" not in data:
            raise KeyError(f"'steering_vectors' missing in {path}")

        means[lang] = data["steering_vectors"].float()

        print(f"Loaded {lang}: {tuple(means[lang].shape)}")

    # All languages must have matching layer and hidden dimensions.
    shapes = {tuple(t.shape) for t in means.values()}
    if len(shapes) != 1:
        raise ValueError(f"Incompatible mean activation shapes: {shapes}")

    return means


def get_steering_difference(means, source_lang, target_lang):
    """Compute target mean minus source mean for every layer."""
    if source_lang not in means:
        raise KeyError(f"No mean activations for source language {source_lang}")
    if target_lang not in means:
        raise KeyError(f"No mean activations for target language {target_lang}")

    return means[target_lang] - means[source_lang]


def make_steering_hook(vector, strength):
    """
    Add one layer's steering vector to the layer output.

    The vector changes output_hidden_states while preserving output_residual.
    This is intended for architectures where the residual stream is the sum
    of those two returned tensors.
    """
    def forward_hook(self, positions, hidden_states, residual):
        output_hidden_states, output_residual = self._original_forward(
            positions, hidden_states, residual
        )

        direction = vector.to(
            device=output_hidden_states.device,
            dtype=output_hidden_states.dtype,
        )
        direction = direction / direction.norm().clamp_min(1e-8)

        output_hidden_states = (
            output_hidden_states
            + strength * direction
        )

        return output_hidden_states, output_residual

    return forward_hook


def apply_steering(model, diff, strength, layer_start=0, layer_end=None):
    layers = get_model_layers(model)

    if diff.ndim != 2:
        raise ValueError(f"Expected [layers, hidden_size], got {diff.shape}")

    if len(layers) != diff.shape[0]:
        raise ValueError(
            f"Model has {len(layers)} layers, but vector has "
            f"{diff.shape[0]} layers. Check model/vector compatibility."
        )

    if layer_end is None:
        layer_end = len(layers)

    if not (0 <= layer_start < layer_end <= len(layers)):
        raise ValueError(
            f"Invalid layer range [{layer_start}, {layer_end}) "
            f"for {len(layers)} layers."
        )

    originals = []

    try:
        for idx in range(layer_start, layer_end):
            layer = layers[idx]
            original = layer.forward
            originals.append((layer, original))

            layer._original_forward = original
            layer.forward = MethodType(
                make_steering_hook(diff[idx], strength),
                layer,
            )
    except Exception:
        restore_steering(originals)
        raise

    return originals


def restore_steering(originals):
    for layer, original in reversed(originals):
        layer.forward = original
        if hasattr(layer, "_original_forward"):
            delattr(layer, "_original_forward")


def run_experiment(
    model,
    sampling_params,
    questions,
    means,
    source_lang,
    target_lang,
    strength,
    layer_start,
    layer_end,
):
    print(f"\n{source_lang} -> {target_lang} | strength={strength}")

    if source_lang not in questions:
        raise KeyError(f"No validation questions for {source_lang}")

    diff = get_steering_difference(means, source_lang, target_lang)
    print(f"DiffMean shape: {tuple(diff.shape)}")

    originals = apply_steering(
        model,
        diff,
        strength,
        layer_start,
        layer_end,
    )

    try:
        prompts = questions[source_lang]
        messages = [
            [{"role": "user", "content": prompt}]
            for prompt in prompts
        ]

        outputs = model.chat(
            messages,
            sampling_params=sampling_params,
        )

        results = []
        for idx, (output, prompt) in enumerate(zip(outputs, prompts)):
            response = output.outputs[0].text.strip()

            results.append({
                "question_idx": idx,
                "input": prompt,
                "output": response,
            })

            if idx < 3:
                print(f"Q{idx + 1}: {prompt[:100]}")
                print(f"A{idx + 1}: {response[:150]}\n")

    finally:
        restore_steering(originals)

    return {
        "source_language": source_lang,
        "target_language": target_lang,
        "steering_method": "DiffMean",
        "steering_strength": strength,
        "layer_start": layer_start,
        "layer_end": layer_end,
        "model": args.model,
        "results": results,
    }


def save_results(result, output_dir):
    strength = str(result["steering_strength"]).replace(".", "_")
    destination = output_dir / f"strength_{strength}"
    destination.mkdir(parents=True, exist_ok=True)

    filename = (
        f"{result['source_language']}_to_"
        f"{result['target_language']}.json"
    )
    path = destination / filename

    with path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Saved: {path}")


def main():
    global args

    parser = argparse.ArgumentParser(
        description="Generate text using DiffMean language steering."
    )
    parser.add_argument(
        "-m", "--model",
        default="meta-llama/Llama-3.1-8B-Instruct",
    )
    parser.add_argument(
        "--model_type",
        choices=["llama", "aya"],
        default="llama",
        help="Select steering/data_llama or steering/data_aya.",
    )
    parser.add_argument("--source_lang", default="en")
    parser.add_argument("--target_lang", default="fr")
    parser.add_argument("--languages", nargs="+", default=None)
    parser.add_argument("--batch_mode", action="store_true")
    parser.add_argument("--steering_strength", type=float, default=1.0)
    parser.add_argument("--layer_start", type=int, default=0)
    parser.add_argument("--layer_end", type=int, default=None)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--output", default="steering_results")
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    steering_dir = PROJECT_DIR / "steering"
    data_dir = steering_dir / f"data_{args.model_type}"
    output_dir = PROJECT_DIR / args.output

    manifest_path = PROJECT_DIR / "data" / "manifest.json"
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    available_languages = list(manifest["languages"])
    languages = args.languages or available_languages

    unknown = sorted(set(languages) - set(available_languages))
    if unknown:
        raise ValueError(
            f"Languages absent from manifest: {unknown}. "
            f"Available: {available_languages}"
        )

    print(f"Project: {PROJECT_DIR}")
    print(f"Mean activations: {data_dir}")
    print(f"Validation data: {PROJECT_DIR / 'data' / 'validation'}")

    means = load_mean_activations(data_dir, languages)
    questions = get_test_questions(k=70)

    # Keep only languages with validation questions.
    missing_questions = [lang for lang in languages if lang not in questions]
    if missing_questions:
        raise ValueError(
            f"No validation questions found for: {missing_questions}"
        )

    if torch.cuda.device_count() < 1:
        raise RuntimeError("No CUDA GPU available.")

    model = LLM(
        model=args.model,
        tensor_parallel_size=torch.cuda.device_count(),
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        enforce_eager=True,
        dtype="float16",
        enable_prefix_caching=False,
    )

    tokenizer = model.get_tokenizer()
    stop_ids = (
        [tokenizer.eos_token_id]
        if tokenizer.eos_token_id is not None
        else None
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        stop_token_ids=stop_ids,
        skip_special_tokens=True,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    if args.batch_mode:
        pairs = [
            (source, target)
            for source in languages
            for target in languages
            if source != target
        ]
    else:
        pairs = [(args.source_lang, args.target_lang)]

    for idx, (source, target) in enumerate(pairs, 1):
        filename = (
            f"strength_{str(args.steering_strength).replace('.', '_')}/"
            f"{source}_to_{target}.json"
        )
        if args.skip_existing and (output_dir / filename).exists():
            print(f"[{idx}/{len(pairs)}] Skipping existing {source}->{target}")
            continue

        try:
            result = run_experiment(
                model,
                sampling_params,
                questions,
                means,
                source,
                target,
                args.steering_strength,
                args.layer_start,
                args.layer_end,
            )
            save_results(result, output_dir)

        except Exception as exc:
            print(f"Failed {source}->{target}: {exc}")
            import traceback
            traceback.print_exc()

    print("Generation complete.")


if __name__ == "__main__":
    main()

"""Mean residual-stream activations per language (HF + forward hooks).

Replaces the vLLM-internals version, which only works on the old V0 engine.
Input : steering/data_{save}/flores_{lang}.pt   (1-D LongTensor of token ids)
Output: steering/data_{save}/mean.{lang}.pt     (same keys as before)
"""
import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM


def find_layers(model):
    for path in (
        "model.layers",
        "model.language_model.layers",
        "model.language_model.model.layers",
        "language_model.model.layers",
    ):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            continue
    raise RuntimeError("Could not find decoder layers for this architecture.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument(
        "-l", "--lang", nargs="+", default=["all"],
        help="One or more language codes, or 'all' for every flores_*.pt found.",
    )
    parser.add_argument("-s", "--save", default="llama", choices=["llama", "aya"])
    parser.add_argument("--target_tokens", type=int, default=10_000_000)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    data_dir = Path(__file__).resolve().parent / f"data_{args.save}"

    if args.lang == ["all"]:
        langs = sorted(p.stem[len("flores_"):] for p in data_dir.glob("flores_*.pt"))
    else:
        langs = args.lang
    if not langs:
        raise FileNotFoundError(f"No flores_*.pt files found in {data_dir}")

    todo = [
        l for l in langs
        if args.overwrite or not (data_dir / f"mean.{l}.pt").exists()
    ]
    print(f"Languages: {len(langs)} total, {len(todo)} to process")
    if not todo:
        return

    print(f"Loading {args.model} (float16, device_map=auto)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,   # T4 has no bfloat16
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    layers = find_layers(model)
    num_layers = model.config.get_text_config().num_hidden_layers
    hidden_size = model.config.get_text_config().hidden_size
    if len(layers) != num_layers:
        raise RuntimeError(f"Found {len(layers)} layers, expected {num_layers}.")

    sums = [None] * num_layers

    def make_hook(idx):
        def hook(module, inputs, output):
            # HF decoder layers return the full residual stream (a tensor in
            # newer versions, a tuple in older ones). Works for Llama and Aya.
            h = output[0] if isinstance(output, tuple) else output
            h = h.detach().float().reshape(-1, h.shape[-1])
            if sums[idx] is None:
                sums[idx] = torch.zeros(hidden_size, dtype=torch.float32, device=h.device)
            sums[idx] += h.sum(dim=0)
        return hook

    handles = [layer.register_forward_hook(make_hook(i)) for i, layer in enumerate(layers)]
    first_device = next(model.parameters()).device

    try:
        for lang in todo:
            input_path = data_dir / f"flores_{lang}.pt"
            if not input_path.exists():
                print(f"✗ {lang}: {input_path} not found, skipping")
                continue

            token_ids = torch.load(input_path, map_location="cpu")
            if args.target_tokens > 0:
                token_ids = token_ids[: args.target_tokens]
            usable = (len(token_ids) // args.seq_len) * args.seq_len
            if usable == 0:
                print(f"✗ {lang}: only {len(token_ids)} tokens, need >= {args.seq_len}. "
                      f"Lower --seq_len. Skipping")
                continue
            input_ids = token_ids[:usable].reshape(-1, args.seq_len)
            print(f"{lang}: {len(token_ids):,} tokens -> {input_ids.size(0)} sequences "
                  f"of {args.seq_len} ({usable:,} used)")

            for i in range(num_layers):
                sums[i] = None

            with torch.no_grad():
                for start in range(0, input_ids.size(0), args.batch_size):
                    batch = input_ids[start:start + args.batch_size].to(first_device)
                    # base_model skips the LM head, so no huge logits tensor
                    model.base_model(input_ids=batch, use_cache=False)

            if any(s is None for s in sums):
                raise RuntimeError("Some layers captured nothing; hooks did not fire.")

            mean_activations = torch.stack([s.cpu() / usable for s in sums])
            result = {
                "steering_vectors": mean_activations,  # [num_layers, hidden_size]
                "token_counts": torch.full((num_layers,), usable, dtype=torch.int64),
                "num_tokens": usable,
                "language": lang,
                "model": args.model,
                "method": "mean_activation",
            }
            out_path = data_dir / f"mean.{lang}.pt"
            torch.save(result, out_path)
            print(f"✓ {lang}: saved {out_path.name} {tuple(mean_activations.shape)}")
    finally:
        for h in handles:
            h.remove()


if __name__ == "__main__":
    main()

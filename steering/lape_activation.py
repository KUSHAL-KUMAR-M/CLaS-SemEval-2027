#!/usr/bin/env python3
"""Plain LAPE activation collection for this repo (HF, not vLLM internals).

Adapted from d-gurgurov/CLaS-Bench, identification/lape_activation.py:
https://github.com/d-gurgurov/CLaS-Bench/blob/main/CLaS-Bench/identification/lape_activation.py
Collects act_fn(gate_proj(x)), BEFORE multiplication by up_proj(x).
No novel selection, centering, steering, or chat instructions are applied.

Input: steering/data_<save>/flores_<lang>.pt (default, existing token tensors),
       or --input-format text: data/train/<lang>.txt.
Output: steering/data_<save>/activation.<lang>.pt, upstream statistic keys.
Run from repo root: python steering/lape_activation.py -s llama -l all
"""
import argparse
import hashlib
import inspect
from pathlib import Path

import torch

MODELS = {
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "aya": "CohereLabs/tiny-aya-global",
}


def find_layers(model):
    for path in ("model.layers", "model.language_model.layers",
                 "model.language_model.model.layers", "language_model.model.layers"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            continue
    raise RuntimeError("Unsupported model: could not find decoder layers.")


class GateStatistics:
    """Observe native gate activations without replacing any model forward method."""
    def __init__(self, layers):
        self.sums = [None] * len(layers)
        self.positive = [None] * len(layers)
        self.counts = [0] * len(layers)
        self.handles = []
        # Fail before installing hooks if architecture is not supported.
        for layer in layers:
            if not all(hasattr(layer.mlp, key) for key in ("gate_proj", "act_fn")):
                raise RuntimeError("Requires an MLP with gate_proj and act_fn (Llama/Aya).")
        for idx, layer in enumerate(layers):
            self.handles.append(layer.mlp.gate_proj.register_forward_hook(
                self.make_hook(idx, layer.mlp.act_fn)))

    def make_hook(self, idx, act_fn):
        def hook(module, inputs, output):
            # Native act_fn keeps the model's activation and approximation choice.
            # No return value: model output is unchanged.
            values = act_fn(output.detach()).float().reshape(-1, output.shape[-1])
            positive = (values > 0).sum(0, dtype=torch.int64)
            total = values.sum(0, dtype=torch.float64)
            if self.sums[idx] is None:
                self.sums[idx] = total
                self.positive[idx] = positive
            else:
                self.sums[idx] += total
                self.positive[idx] += positive
            self.counts[idx] += values.shape[0]
        return hook

    def reset(self):
        self.sums = [None] * len(self.sums)
        self.positive = [None] * len(self.positive)
        self.counts = [0] * len(self.counts)

    def result(self, n):
        if not n or any(count != n for count in self.counts):
            raise RuntimeError(f"Expected {n} tokens at each layer; got {self.counts}")
        means = torch.stack([(s / n).float().cpu() for s in self.sums])
        return dict(n=n,
                    over_zero=torch.stack([p.cpu() for p in self.positive]),
                    average_activations=means,
                    activation_counts=torch.full(means.shape, n, dtype=torch.int64))

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def load_tokens(path, tokenizer, token_limit, input_format):
    if input_format == "tokens":
        ids = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(ids, torch.Tensor) or ids.ndim != 1 or ids.dtype != torch.long:
            raise ValueError(f"{path}: expected a 1-D torch.long token tensor")
        return ids[:token_limit] if token_limit else ids
    # Matches this repo's load_data.py: concatenate raw sentence tokens without
    # separators or added special tokens; never use validation questions.
    ids = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = tokenizer.encode(line.strip(), add_special_tokens=False)
            if token_limit:
                row = row[:max(0, token_limit - len(ids))]
            ids.extend(row)
            if token_limit and len(ids) >= token_limit:
                break
    return torch.tensor(ids, dtype=torch.long)


def main():
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-s", "--save", choices=list(MODELS), default="llama")
    p.add_argument("-m", "--model", help="Defaults to the model matching --save")
    p.add_argument("-l", "--lang", nargs="+", default=["all"])
    p.add_argument("--input-format", choices=["text", "tokens"], default="tokens")
    p.add_argument("--data-dir", type=Path, help="Input directory; defaults depend on input format")
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--seq-len", "--seq_len", type=int, default=1024)
    p.add_argument("--batch-size", "--batch_size", type=int, default=1)
    p.add_argument("--target-tokens", "--target_tokens", type=int, default=10_000_000,
                   help="Maximum tokens per language; 0 uses all available tokens")
    p.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    p.add_argument("--revision", default="main")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    if args.seq_len < 1 or args.batch_size < 1 or args.target_tokens < 0:
        p.error("Sequence length/batch size must be positive; target tokens must be >= 0")
    model_id = args.model or MODELS[args.save]
    output_dir = args.output_dir or root / "steering" / f"data_{args.save}"
    data_dir = args.data_dir or (root / "data" / "train" if args.input_format == "text"
                                else root / "steering" / f"data_{args.save}")
    pattern = "*.txt" if args.input_format == "text" else "flores_*.pt"
    langs = (sorted(path.stem if args.input_format == "text" else path.stem[7:]
                    for path in data_dir.glob(pattern)) if args.lang == ["all"] else args.lang)
    if not langs or "all" in langs:
        p.error(f"No languages found in {data_dir}, or 'all' mixed with language codes")
    jobs = []
    for lang in dict.fromkeys(langs):
        path = data_dir / (f"{lang}.txt" if args.input_format == "text" else f"flores_{lang}.pt")
        out = output_dir / f"activation.{lang}.pt"
        if not path.is_file():
            p.error(f"Missing input: {path}")
        if out.exists() and not args.overwrite:
            previous = torch.load(out, map_location="cpu", weights_only=True)
            expected = dict(model=model_id, revision=args.revision, seq_len=args.seq_len,
                            target_tokens=args.target_tokens, dtype=args.dtype,
                            input_format=args.input_format,
                            input_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            if any(previous.get(k) != v for k, v in expected.items()):
                p.error(f"{out} exists with different/unknown settings. Use --overwrite or another --output-dir")
            print(f"Skip {lang}: matching output exists", flush=True)
        else:
            jobs.append((lang, path, out))
    if not jobs:
        return
    if args.input_format == "tokens":
        print("Token input: ensure files were produced using this exact model's tokenizer.", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = (AutoTokenizer.from_pretrained(model_id, revision=args.revision)
                 if args.input_format == "text" else None)
    print(f"Loading {model_id}; dtype={args.dtype}; device_map=auto", flush=True)
    # Transformers 5 uses dtype; Transformers 4 exposes torch_dtype.
    from transformers import PreTrainedModel
    dtype_key = ("dtype" if "dtype" in inspect.signature(PreTrainedModel.from_pretrained).parameters
                 else "torch_dtype")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=args.revision, device_map="auto",
        **{dtype_key: getattr(torch, args.dtype)})
    model.eval()
    layers = find_layers(model)
    config = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
    if args.seq_len > getattr(config, "max_position_embeddings", args.seq_len):
        p.error("Sequence length exceeds model context length")
    stats = GateStatistics(layers)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        device = model.get_input_embeddings().weight.device
        for lang, path, out in jobs:
            ids = load_tokens(path, tokenizer, args.target_tokens, args.input_format)
            n = ids.numel() // args.seq_len * args.seq_len
            if n == 0:
                raise ValueError(f"{lang}: {ids.numel()} tokens < seq_len={args.seq_len}; lower --seq-len")
            if ids.min() < 0 or ids.max() >= model.get_input_embeddings().num_embeddings:
                raise ValueError(f"{lang}: token ids outside model vocabulary")
            rows = ids[:n].reshape(-1, args.seq_len)
            stats.reset()
            print(f"{lang}: {n:,} tokens, {len(rows)} sequences; tail dropped: {ids.numel()-n}", flush=True)
            with torch.inference_mode():
                for start in range(0, len(rows), args.batch_size):
                    batch = rows[start:start + args.batch_size].to(device)
                    # Skips LM head and generation: count corpus tokens exactly once.
                    model.base_model(input_ids=batch, use_cache=False,
                                     output_hidden_states=False, output_attentions=False)
                    if start % (args.batch_size * 25) == 0:
                        print(f"  {min(start+args.batch_size, len(rows))}/{len(rows)} sequences", flush=True)
            result = stats.result(n)
            result.update(model=model_id, model_type=getattr(config, "model_type", "unknown"),
                          revision=args.revision, resolved_revision=getattr(model.config, "_commit_hash", None),
                          language=lang, method="lape_gate_activation", seq_len=args.seq_len,
                          target_tokens=args.target_tokens, dtype=args.dtype,
                          input_format=args.input_format, input_file=str(path.resolve()),
                          input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                          num_layers=len(layers), activation_site="act_fn(gate_proj), before up multiplication")
            temporary = out.with_suffix(out.suffix + ".tmp")
            torch.save(result, temporary)
            temporary.replace(out)
            print(f"Saved {out}: {tuple(result['over_zero'].shape)}", flush=True)
    finally:
        stats.close()


if __name__ == "__main__":
    main()

"""Rank LAPE neurons by consistent activation differences on parallel FLORES."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from lape_activation import find_layers


def rank_neurons(source, target, candidates, eps):
    """Rank candidates within their original language mask."""
    if candidates.numel() == 0:
        return candidates
    difference = target - source
    score = difference.mean(0).abs() / difference.std(0, unbiased=False).clamp_min(eps)
    selected = torch.argsort(score, descending=True, stable=True)
    return candidates[selected]


def collect(model, tokenizer, layers, texts, candidates, batch_size, max_length):
    chunks = [[] for _ in layers]
    attention = None

    def observe(index, activation):
        def hook(module, inputs, output):
            indices = candidates[index].to(output.device)
            gate = activation(output.index_select(-1, indices)).float()
            valid = attention.to(output.device).bool().unsqueeze(-1)
            pooled = gate.masked_fill(~valid, 0).sum(1) / valid.sum(1).clamp_min(1)
            chunks[index].append(pooled.cpu())
        return hook

    hooks = []
    try:
        for index, layer in enumerate(layers):
            if candidates[index].numel():
                hooks.append(layer.mlp.gate_proj.register_forward_hook(
                    observe(index, layer.mlp.act_fn)))
        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                batch = tokenizer(texts[start:start + batch_size], padding=True,
                                  truncation=True, max_length=max_length,
                                  add_special_tokens=False, return_tensors="pt")
                attention = batch["attention_mask"]
                if not attention.any(1).all():
                    raise ValueError("A FLORES sentence produced no tokens")
                model.base_model(**batch.to(model.get_input_embeddings().weight.device),
                                 use_cache=False)
    finally:
        for hook in hooks:
            hook.remove()
    return [torch.cat(rows) if rows else torch.empty(len(texts), 0) for rows in chunks]


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activation-mask", type=Path, required=True)
    parser.add_argument("--train-dir", type=Path, default=root / "data/train")
    parser.add_argument("--output", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pairs", nargs="+")
    group.add_argument("--track", choices=["t2"])
    parser.add_argument("--sentences", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--eps", type=float, default=1e-4)
    args = parser.parse_args()
    if args.sentences < 2 or min(args.batch_size, args.max_length) < 1 or args.eps <= 0:
        parser.error("Invalid extraction settings")
    if args.track:
        import sys
        sys.path.insert(0, str(root))
        from clas.languages import pairs
        requested = list(pairs(args.track))
    else:
        requested = [tuple(pair.split(":")) for pair in args.pairs]
    metadata_path = args.activation_mask.with_suffix(args.activation_mask.suffix + ".json")
    metadata = json.loads(metadata_path.read_text())
    languages = metadata["languages"]
    mask = dict(zip(languages, torch.load(args.activation_mask, weights_only=True, map_location="cpu"), strict=True))
    if any(len(pair) != 2 or pair[0] == pair[1] or any(lang not in mask for lang in pair)
           for pair in requested):
        parser.error("Each pair must specify two different languages covered by the LAPE mask")
    used = sorted({lang for pair in requested for lang in pair})
    texts = {lang: (args.train_dir / f"{lang}.txt").read_text(encoding="utf-8").splitlines()
             for lang in used}
    lengths = {len(rows) for rows in texts.values()}
    if len(lengths) != 1 or min(lengths) < args.sentences:
        parser.error("Need equally sized parallel FLORES files with enough sentences")
    texts = {lang: rows[:args.sentences] for lang, rows in texts.items()}
    if any(not row.strip() for rows in texts.values() for row in rows):
        parser.error("Blank FLORES rows would compromise alignment")
    settings = metadata["settings"]
    model_id = settings["model"]
    revision = settings.get("resolved_revision") or settings.get("revision") or "main"
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, torch_dtype=torch.float16, device_map="auto").eval()
    layers = find_layers(model)
    if any(len(mask[lang]) != len(layers) for lang in used):
        raise ValueError("Mask and model have different layer counts")
    candidates = [torch.cat([mask[lang][layer] for lang in used]).unique(sorted=True)
                  for layer in range(len(layers))]
    features = {}
    for lang in used:
        print(f"Collecting {args.sentences} aligned sentences for {lang}", flush=True)
        features[lang] = collect(model, tokenizer, layers, texts[lang], candidates,
                                 args.batch_size, args.max_length)
    pair_masks = {}
    for source, target in requested:
        selected = {"source": [], "target": []}
        for layer, union in enumerate(candidates):
            pair_union = torch.cat((mask[source][layer], mask[target][layer])).unique(sorted=True)
            columns = torch.searchsorted(union, pair_union)
            src, tgt = features[source][layer][:, columns], features[target][layer][:, columns]
            if not torch.isfinite(src).all() or not torch.isfinite(tgt).all():
                raise ValueError("Nonfinite gate activations")
            # Rank both roles by the same contrast, but keep their original language
            # membership. This avoids turning the same neuron into both roles by design.
            for role, lang in (("source", source), ("target", target)):
                original = mask[lang][layer]
                own = torch.searchsorted(union, original)
                selected[role].append(rank_neurons(
                    features[source][layer][:, own], features[target][layer][:, own],
                    original, args.eps))
        pair_masks[f"{source}:{target}"] = selected
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    torch.save(dict(model=model_id, revision=revision, pairs=pair_masks,
                    sentences=args.sentences, max_length=args.max_length, eps=args.eps,
                    mask_sha256=hashlib.sha256(args.activation_mask.read_bytes()).hexdigest()),
               temporary)
    temporary.replace(args.output)
    print(f"Saved {len(pair_masks)} ranked pairs to {args.output}")


if __name__ == "__main__":
    main()

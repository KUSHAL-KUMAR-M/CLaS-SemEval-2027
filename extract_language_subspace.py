#!/usr/bin/env python3
"""Extract aligned, equally weighted assistant-sentence language subspaces."""
import argparse
from pathlib import Path

from steering.experiment_utils import decoder_layers, read_aligned, file_digest, versions


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', default='meta-llama/Llama-3.1-8B-Instruct')
    p.add_argument('--revision', default=None)
    p.add_argument('--parallel-dir', type=Path, required=True,
                   help='Aligned {language}.txt files, same sentence IDs/order across files')
    p.add_argument('--languages', nargs='+', required=True)
    p.add_argument('--layers', nargs='+', type=int, default=[16, 20, 24])
    p.add_argument('--rank', type=int, default=8)
    p.add_argument('--num-sentences', type=int, default=512)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--max-length', type=int, default=512)
    p.add_argument('--dtype', choices=['float16', 'bfloat16', 'float32'], default='float16')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if (len(set(a.languages)) != len(a.languages) or len(set(a.layers)) != len(a.layers)
            or a.batch_size < 1 or a.num_sentences < 2 or a.max_length < 16
            or not 1 <= a.rank <= len(a.languages)-1):
        p.error('Need unique languages/layers, positive batch size, >=2 sentences, max-length>=16 and rank in [1, n_languages-1]')
    rows = read_aligned(a.parallel_dir, a.languages)
    n = min(a.num_sentences, len(next(iter(rows.values()))))
    if n < 2:
        p.error('Need at least two aligned sentences')
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from steering.subspace import fit_subspace, FORMAT_VERSION
    tokenizer = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
    tokenizer.padding_side = 'right'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        a.model, revision=a.revision, dtype=getattr(torch, a.dtype), device_map='auto')
    model.eval()
    layers = decoder_layers(model)
    if not a.layers or any(i < 0 or i >= len(layers) for i in a.layers):
        p.error(f'Layer indices must be in [0, {len(layers)-1}]')
    # The same neutral context for every language; no target-language instruction.
    prefix = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': 'Continue.'}], tokenize=True, add_generation_prompt=True)
    if len(prefix) >= a.max_length:
        p.error('max-length too short for the neutral assistant prefix')
    pooled, mask, handles = {}, None, []

    def capture(index):
        def hook(module, inputs, output):
            h = output[0] if isinstance(output, (tuple, list)) else output
            weights = mask.to(h.device).unsqueeze(-1)
            pooled[index] = ((h.float() * weights).sum(dim=1)
                             / weights.sum(dim=1).clamp_min(1)).cpu()
        return hook

    all_languages = []
    try:
        handles = [layers[i].register_forward_hook(capture(i)) for i in a.layers]
        for lang in a.languages:
            batches = []
            for start in range(0, n, a.batch_size):
                encoded = [tokenizer.encode(t, add_special_tokens=False)[:a.max_length-len(prefix)]
                           for t in rows[lang][start:min(start+a.batch_size, n)]]
                if any(not ids for ids in encoded):
                    raise ValueError(f'Empty tokenized sentence in {lang}')
                sequences = [prefix + ids for ids in encoded]
                width = max(map(len, sequences))
                ids = torch.full((len(sequences), width), tokenizer.pad_token_id, dtype=torch.long)
                attention = torch.zeros_like(ids)
                mask = torch.zeros_like(ids, dtype=torch.float32)
                for j, seq in enumerate(sequences):
                    ids[j, :len(seq)] = torch.tensor(seq)
                    attention[j, :len(seq)] = 1
                    mask[j, len(prefix):len(seq)] = 1
                device = model.get_input_embeddings().weight.device
                pooled.clear()
                with torch.inference_mode():
                    model.base_model(input_ids=ids.to(device), attention_mask=attention.to(device),
                                     use_cache=False)
                if set(pooled) != set(a.layers):
                    raise RuntimeError('Not all extraction hooks fired')
                batches.append(torch.stack([pooled[i] for i in a.layers], dim=1))
            all_languages.append(torch.cat(batches))
            print(f'{lang}: {n} aligned sentences captured', flush=True)
    finally:
        for handle in handles:
            handle.remove()
    artifact = fit_subspace(torch.stack(all_languages), a.rank)
    artifact.update(format_version=FORMAT_VERSION, model=a.model,
                    revision=getattr(model.config, '_commit_hash', None) or a.revision,
                    languages=a.languages, layers=a.layers, num_sentences=n,
                    pooling='assistant-token-mean/equal-sentence-weight', neutral_context='Continue.',
                    data_sha256={lang: file_digest(a.parallel_dir / f'{lang}.txt') for lang in a.languages},
                    extraction_args={k: str(v) if isinstance(v, Path) else v for k,v in vars(a).items()},
                    versions=versions())
    a.out.parent.mkdir(parents=True, exist_ok=True)
    temp = a.out.with_suffix(a.out.suffix + '.tmp')
    torch.save(artifact, temp)
    temp.replace(a.out)
    print(f'Saved {a.out}; model revision={artifact["revision"]}')


if __name__ == '__main__':
    main()

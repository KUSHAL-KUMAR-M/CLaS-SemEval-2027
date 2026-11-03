#!/usr/bin/env python3
"""Plain upstream LAPE selection, adapted to this repository's activation.*.pt.
Source: https://github.com/d-gurgurov/CLaS-Bench/blob/main/CLaS-Bench/identification/lape_identify.py
Saves upstream-compatible masks[language_index][layer] LongTensor neuron IDs,
plus a JSON sidecar specifying the exact language order and extraction settings.
"""
import argparse
import hashlib
import json
from pathlib import Path
import torch


def identify(probabilities, top_rate=0.02):
    if probabilities.ndim != 3 or not torch.isfinite(probabilities).all():
        raise ValueError('Expected finite [layers, neurons, languages] probabilities')
    if not 0 < top_rate <= 1 or (probabilities < 0).any() or (probabilities > 1).any():
        raise ValueError('Invalid top_rate or probabilities')
    p = probabilities.float()
    q = p / p.sum(-1, keepdim=True).clamp_min(torch.finfo(p.dtype).tiny)
    entropy = -(q * q.clamp_min(torch.finfo(q.dtype).tiny).log()).sum(-1)
    flat = p.flatten()
    # Preserve upstream kthvalue quantile convention and strict comparisons.
    threshold = flat.kthvalue(max(1, round(flat.numel() * 0.95))).values
    eligible = (p > threshold).any(-1)
    entropy[~eligible] = torch.inf
    k = round(entropy.numel() * top_rate)
    if k < 1:
        raise ValueError('top_rate selects zero neurons; increase it')
    indices = entropy.flatten().topk(k, largest=False).indices
    rows, cols = indices // p.shape[1], indices % p.shape[1]
    selected = p[rows, cols]
    masks = []
    for language in range(p.shape[-1]):
        keep = selected[:, language] > threshold
        masks.append([cols[keep & (rows == layer)].sort().values.cpu().long()
                      for layer in range(p.shape[0])])
    # Empty languages retain their slot (fixes upstream trailing-empty bug).
    return masks, dict(top_rate=top_rate, filter_rate=0.95,
                       activation_bar_ratio=0.95, probability_threshold=float(threshold),
                       eligible_neurons=int(eligible.sum()), entropy_candidates=k,
                       counts=[sum(t.numel() for t in layers) for layers in masks])


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-s', '--save', choices=['llama', 'aya'], default='llama')
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('-l', '--langs', nargs='+', default=['all'])
    parser.add_argument('--top-rate', '--top_rate', type=float, default=0.02)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    folder = args.data_dir or root / f'data_{args.save}'
    languages = (sorted(p.name[len('activation.'):-len('.pt')]
                        for p in folder.glob('activation.*.pt'))
                 if args.langs == ['all'] else args.langs)
    if len(languages) < 2 or len(set(languages)) != len(languages) or 'all' in languages:
        parser.error('Need at least two distinct languages; do not mix all with codes')
    output = args.output or root / 'activation_mask' / f'{args.save}.lape.pt'
    sidecar = output.with_suffix(output.suffix + '.json')
    if not args.overwrite and (output.exists() or sidecar.exists()):
        parser.error(f'{output} or its metadata exists; use --overwrite')
    probabilities, sources, reference = [], [], None
    settings = ('model', 'revision', 'resolved_revision', 'method', 'seq_len',
                'target_tokens', 'dtype', 'input_format', 'activation_site')
    shape = None
    for language in languages:
        path = folder / f'activation.{language}.pt'
        data = torch.load(path, map_location='cpu', weights_only=True)
        n, counts = int(data['n']), data['over_zero']
        if n <= 0 or counts.ndim != 2 or not torch.isfinite(counts).all() or (counts < 0).any() or (counts > n).any():
            raise ValueError(f'{path}: invalid counts or token denominator')
        if data.get('language', language) != language:
            raise ValueError(f'{path}: language metadata mismatch')
        current = {key: data.get(key) for key in settings}
        if reference is None:
            reference, shape = current, counts.shape
        elif current != reference or counts.shape != shape:
            raise ValueError(f'{path}: model/extraction settings or tensor shape differ')
        if 'activation_counts' in data and not torch.all(data['activation_counts'] == n):
            raise ValueError(f'{path}: activation_counts differs from n')
        probabilities.append(counts.float() / n)
        sources.append(dict(language=language, file=str(path.resolve()), n=n,
                            sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    masks, summary = identify(torch.stack(probabilities, dim=-1), args.top_rate)
    metadata = dict(languages=languages, language_to_index={l:i for i,l in enumerate(languages)},
                    num_layers=shape[0], intermediate_size=shape[1], settings=reference,
                    sources=sources, **summary)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(masks, output)
    sidecar.write_text(json.dumps(metadata, indent=2) + '\n')
    for language, count in zip(languages, summary['counts']):
        print(f'{language}: {count} neuron assignments')
    print(f'Saved masks: {output}\nLanguage order and settings: {sidecar}')


if __name__ == '__main__':
    main()

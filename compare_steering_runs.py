#!/usr/bin/env python3
"""Compare evaluate.py score files; report both aggregation conventions."""
import argparse
import json
from pathlib import Path


def harmonic(a, b):
    return 2*a*b/(a+b) if a+b else 0.0


def summarize(scores):
    rows = list(scores['by_pair'].values())
    if not rows:
        raise ValueError('Empty score file')
    if any(not 0 <= row[key] <= 1 for row in rows for key in ('lfs', 'or')):
        raise ValueError('Metrics must be between zero and one')
    lfs = sum(r['lfs'] for r in rows)/len(rows)
    relevance = sum(r['or'] for r in rows)/len(rows)
    return {'pairs': len(rows), 'lfs': lfs, 'or': relevance,
            'lss_from_macro_components': harmonic(lfs, relevance),
            'mean_pair_lss': sum(harmonic(r['lfs'], r['or']) for r in rows)/len(rows),
            'unparsed_judge': sum(r.get('n_unparsed_judge', 0) for r in rows)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('scores', nargs='+', type=Path)
    a = p.parse_args()
    expected = None
    print('| Run | Pairs | LFS | Relevance | LSS of means | Mean pair LSS | Unparsed |')
    print('|---|---:|---:|---:|---:|---:|---:|')
    for path in a.scores:
        data = json.loads(path.read_text())
        pair_ids = {key: row['n_items'] for key,row in data['by_pair'].items()}
        if expected is not None and pair_ids != expected:
            raise ValueError('Runs must have identical pair sets and item counts')
        expected = pair_ids
        r = summarize(data)
        print(f'| {path.parent} | {r["pairs"]} | {r["lfs"]:.4f} | {r["or"]:.4f} | '
              f'{r["lss_from_macro_components"]:.4f} | {r["mean_pair_lss"]:.4f} | {r["unparsed_judge"]} |')
    print('\nVerify identical question IDs, judge settings, and LID settings before comparing. '
          'An LFS-only (--skip-judge) report is not a joint-score experiment.')


if __name__ == '__main__':
    main()

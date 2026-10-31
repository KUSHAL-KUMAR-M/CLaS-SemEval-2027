#!/usr/bin/env python3
"""Write (never execute) a staged sweep of reproducible generation commands."""
import argparse
import shlex
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifact', required=True)
    p.add_argument('--model', default='meta-llama/Llama-3.1-8B-Instruct')
    p.add_argument('--questions-dir', default='data/validation')
    p.add_argument('--backend', choices=['hf', 'vllm'], default='hf')
    p.add_argument('--scope', choices=['last', 'all'], default=None)
    p.add_argument('--layers', type=int, nargs='+', default=[16, 20, 24])
    p.add_argument('--sources', nargs='+', default=['en', 'ar', 'ru', 'hi'])
    p.add_argument('--targets', nargs='+', default=['mt', 'sw', 'fa', 'tr'])
    p.add_argument('--limit', type=int, default=8)
    p.add_argument('--out', default='/kaggle/working/subspace_runs')
    p.add_argument('--plan', type=Path, required=True)
    a = p.parse_args()
    scope = a.scope or ('all' if a.backend == 'vllm' else 'last')
    if a.backend == 'vllm' and scope != 'all':
        p.error('vLLM requires all-token scope')
    commands = ['#!/usr/bin/env bash', 'set -euo pipefail',
                '# Screening only: tune split; no holdout selection. Each command reloads the model.']
    common = [sys.executable, 'subspace_generate.py', '--artifact', a.artifact, '--model', a.model,
              '--questions-dir', a.questions_dir, '--backend', a.backend, '--scope', scope,
              '--split', 'tune', '--limit', str(a.limit), '--out', a.out, '--resume']
    for target in a.targets:
        pairs = [f'{source}:{target}' for source in a.sources]
        commands.append(shlex.join(common + ['--pairs', *pairs, '--method', 'none', '--layers', str(a.layers[0])]))
        for layer in a.layers:
            for method, strengths in [('diffmean', [2, 5, 8]), ('fixed', [0.25, 0.5, 1]), ('adaptive', [0.25, 0.5, 1])]:
                for strength in strengths:
                    commands.append(shlex.join(common + ['--pairs', *pairs, '--method', method,
                        '--layers', str(layer), '--strength', str(strength)]))
    a.plan.parent.mkdir(parents=True, exist_ok=True)
    a.plan.write_text('\n'.join(commands) + '\n')
    print(f'Wrote {len(commands)-3} screening commands to {a.plan}; nothing executed.')
    print('Evaluate and shortlist on tune; rerun finalists on the full tune split, then holdout once.')


if __name__ == '__main__':
    main()

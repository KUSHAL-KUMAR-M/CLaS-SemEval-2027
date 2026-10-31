#!/usr/bin/env python3
"""Kaggle-ready language steering: unsteered, diffmean, fixed/adaptive subspace."""
import argparse
import json
import math
from functools import partial
from pathlib import Path

from steering.experiment_utils import (read_aligned, row_indices, fingerprint, file_digest,
                                       versions, write_json_atomic)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifact', type=Path, required=True)
    p.add_argument('--model', default='meta-llama/Llama-3.1-8B-Instruct')
    p.add_argument('--backend', choices=['hf', 'vllm'], default='hf')
    p.add_argument('--method', choices=['none', 'diffmean', 'fixed', 'adaptive'], default='adaptive')
    p.add_argument('--questions-dir', type=Path, default=Path('data/validation'))
    p.add_argument('--pairs', nargs='+', help='e.g. en:mt en:sw; default all Track 2 pairs')
    p.add_argument('--layers', nargs='+', type=int, default=[20])
    p.add_argument('--strength', type=float, default=0.5, help='Subspace step size, or diffmean vector norm')
    p.add_argument('--threshold', type=float, default=0.8)
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--max-relative-norm', type=float, default=0.1)
    p.add_argument('--scope', choices=['last', 'all'], default='last')
    p.add_argument('--split', choices=['tune', 'holdout', 'all'], default='tune')
    p.add_argument('--split-seed', type=int, default=42)
    p.add_argument('--holdout-fraction', type=float, default=0.3)
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--max-new-tokens', type=int, default=512)
    p.add_argument('--max-model-len', type=int, default=4096)
    p.add_argument('--dtype', choices=['float16', 'bfloat16', 'float32'], default='float16')
    p.add_argument('--tensor-parallel-size', type=int, default=1)
    p.add_argument('--gpu-memory-utilization', type=float, default=0.85)
    p.add_argument('--out', type=Path, default=Path('/kaggle/working/subspace_runs'))
    p.add_argument('--resume', action='store_true')
    p.add_argument('--dry-run', action='store_true', help='Validate data/config; no torch or model load')
    a = p.parse_args()
    if (not all(math.isfinite(v) for v in (a.strength, a.threshold, a.temperature, a.max_relative_norm, a.gpu_memory_utilization))
            or not a.layers or len(set(a.layers)) != len(a.layers) or min(a.layers) < 0
            or a.strength < 0 or not 0 < a.threshold <= 1 or a.temperature <= 0
            or a.max_relative_norm <= 0 or a.batch_size < 1
            or a.max_new_tokens < 1 or a.max_model_len <= a.max_new_tokens
            or a.tensor_parallel_size < 1 or not 0 < a.gpu_memory_utilization < 1
            or (a.limit is not None and a.limit < 1)):
        p.error('Invalid intervention, batching, or generation settings')
    if a.backend == 'vllm' and a.scope != 'all':
        p.error('vLLM packed-token backend requires --scope all; use HF for last-position steering')
    from clas.languages import pairs as track_pairs
    pairs = [tuple(v.split(':')) for v in a.pairs] if a.pairs else track_pairs('t2')
    if any(len(v) != 2 for v in pairs) or len(set(pairs)) != len(pairs):
        p.error('Pairs must be unique source:target entries')
    sources = sorted({s for s, _ in pairs})
    questions = read_aligned(a.questions_dir, sources)
    ids = row_indices(len(questions[sources[0]]), a.split, a.split_seed, a.holdout_fraction)
    if a.limit is not None:
        ids = ids[:a.limit]
    if not ids:
        p.error('Selected split has no questions')
    config = {k: str(v) if isinstance(v, Path) else v for k,v in vars(a).items()
              if k not in ('out', 'resume', 'dry_run')}
    config['versions'] = versions()
    config['artifact_sha256'] = file_digest(a.artifact)
    config['question_sha256'] = {s: file_digest(a.questions_dir / f'{s}.txt') for s in sources}
    config['question_indices'] = ids
    run_id = fingerprint(config)
    directory = a.out / f'{a.method}_{a.backend}_{run_id[:16]}'
    print(json.dumps({'directory': str(directory), 'run_id': run_id,
                      'pairs': pairs, 'question_indices': ids, 'config': config}, indent=2))
    if a.dry_run:
        return
    import torch
    from steering.subspace import validate_artifact
    from steering.subspace_hooks import install_hf, install_vllm, clear_vllm
    artifact = torch.load(a.artifact, map_location='cpu', weights_only=True)
    validate_artifact(artifact, a.model)
    if any(layer not in artifact['layers'] for layer in a.layers):
        p.error('Selected layers missing from artifact; re-extract with those layers')
    if any(lang not in artifact['languages'] for pair in pairs for lang in pair):
        p.error('Source/target language missing from artifact')
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / 'run.json'
    if manifest.exists() and json.loads(manifest.read_text())['run_id'] != run_id:
        raise RuntimeError('Run fingerprint collision')
    write_json_atomic(manifest, {'run_id': run_id, 'config': config})
    pending = []
    for src, tgt in pairs:
        path = directory / f'{src}_to_{tgt}.json'
        if path.exists():
            data = json.loads(path.read_text())
            if data.get('run_id') != run_id:
                raise RuntimeError(f'Existing output configuration mismatch: {path}')
            if a.resume and [r['question_idx'] for r in data['results']] == ids:
                continue
            raise FileExistsError(f'{path} exists; use --resume or a new configuration')
        pending.append((src, tgt))
    if not pending:
        print('All selected pairs already complete')
        return
    options = {k: getattr(a, k) for k in ('method', 'strength', 'threshold', 'temperature', 'max_relative_norm')}
    if a.backend == 'hf':
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(a.model, revision=artifact['revision'])
        tokenizer.padding_side = 'left'
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(a.model, revision=artifact['revision'],
                    dtype=getattr(torch, a.dtype), device_map='auto')
        model.eval()
    else:
        from vllm import LLM, SamplingParams
        # Importable module-level callbacks avoid serializing closures or credentials.
        model = LLM(model=a.model, revision=artifact['revision'], dtype=a.dtype,
                    tensor_parallel_size=a.tensor_parallel_size, enforce_eager=True,
                    enable_prefix_caching=False, max_model_len=a.max_model_len,
                    gpu_memory_utilization=a.gpu_memory_utilization, seed=0)
        if not callable(getattr(model, 'apply_model', None)):
            raise RuntimeError('Installed vLLM lacks LLM.apply_model; use --backend hf')
        params = SamplingParams(temperature=0, max_tokens=a.max_new_tokens, seed=0)
        tokenizer = model.get_tokenizer()
    for src, tgt in pending:
        prompts = [questions[src][i] for i in ids]
        messages = [[{'role': 'user', 'content': q}] for q in prompts]
        # Reject overlong prompts rather than silently change the question.
        texts = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages]
        if any(len(tokenizer.encode(t, add_special_tokens=False)) + a.max_new_tokens > a.max_model_len for t in texts):
            raise ValueError(f'Prompt + answer exceeds max-model-len for {src}->{tgt}')
        source, target = artifact['languages'].index(src), artifact['languages'].index(tgt)
        outputs = []
        handles = []
        try:
            if a.method != 'none':
                if a.backend == 'hf':
                    handles = install_hf(model, artifact, source, target, a.layers, options, a.scope)
                else:
                    counts = model.apply_model(partial(install_vllm, artifact, source, target, a.layers, options))
                    if any(count != len(a.layers) for count in counts):
                        raise RuntimeError('Not all vLLM workers installed the requested hooks')
            for start in range(0, len(prompts), a.batch_size):
                if a.backend == 'hf':
                    enc = tokenizer(texts[start:start+a.batch_size], padding=True,
                                    add_special_tokens=False, return_tensors='pt')
                    enc = enc.to(model.get_input_embeddings().weight.device)
                    with torch.inference_mode():
                        generated = model.generate(**enc, do_sample=False, use_cache=True,
                            max_new_tokens=a.max_new_tokens, pad_token_id=tokenizer.pad_token_id)
                    outputs.extend(tokenizer.batch_decode(generated[:, enc.input_ids.shape[1]:], skip_special_tokens=True))
                else:
                    generated = model.chat(messages[start:start+a.batch_size], sampling_params=params)
                    outputs.extend(o.outputs[0].text for o in generated)
        finally:
            for handle in handles:
                handle.remove()
            if a.backend == 'vllm' and a.method != 'none':
                model.apply_model(clear_vllm)
        if len(outputs) != len(ids):
            raise RuntimeError('Generation count mismatch')
        write_json_atomic(directory / f'{src}_to_{tgt}.json', {
            'source_language': src, 'target_language': tgt, 'model': a.model,
            'method': f'language-subspace:{a.method}', 'track': 't2',
            'run_id': run_id, 'configuration': config,
            'results': [{'question_idx': i, 'input': q, 'original_prompt': q, 'output': o.strip()}
                        for i,q,o in zip(ids, prompts, outputs)]})
        print(f'{src}->{tgt}: {len(outputs)} answers saved', flush=True)


if __name__ == '__main__':
    main()

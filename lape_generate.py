#!/usr/bin/env python3
"""Plain CLaS-Bench LAPE intervention, vLLM adaptation, evaluator-compatible JSON.
Upstream: https://github.com/d-gurgurov/CLaS-Bench/blob/main/CLaS-Bench/3-lape_generate.py
Source neurons are SET to deactivation_strength (not multiplied), then target
neurons are added/replaced. Shared neurons therefore receive target editing last.
Intervenes on activated MLP gates, before multiplication by up_proj, at all
prompt/decode positions and all layers with selected neurons. No language prompt.
"""
import argparse
import hashlib
import json
from pathlib import Path
from types import MethodType
import torch
import os
from functools import partial


def install_intervention(layers, source_masks, target_masks, means, target_index,
                         method='additive', deactivate_value=0., strength=1.):
    originals = []
    try:
        for idx, layer in enumerate(layers):
            mlp = layer.mlp
            if not all(hasattr(mlp, key) for key in ('gate_proj','up_proj','down_proj','act_fn')):
                raise ValueError('Requires native Llama/Aya gate/up/down MLP')
            source = source_masks[idx] if source_masks is not None else torch.empty(0,dtype=torch.long)
            target = target_masks[idx]
            boost = means[idx, target, target_index]
            if method == 'diffmean':
                other = torch.cat((means[idx,target,:target_index],means[idx,target,target_index+1:]),-1)
                boost = boost - other.mean(-1)
            def forward(self, x, source=source, target=target, boost=boost):
                activated = self.act_fn(self.gate_proj(x))
                activated = activated.clone()
                if source.numel():
                    activated.index_fill_(-1, source.to(activated.device), deactivate_value)
                if target.numel():
                    indices = target.to(activated.device)
                    values = strength * boost.to(device=activated.device,dtype=activated.dtype)
                    if method == 'replacement':
                        activated[...,indices] = values
                    else:
                        activated[...,indices] += values
                return self.down_proj(activated * self.up_proj(x))
            originals.append((mlp,mlp.forward))
            mlp.forward = MethodType(forward,mlp)
        return originals
    except Exception:
        restore(originals)
        raise


def restore(originals):
    for mlp, forward in originals:
        mlp.forward = forward


def main():
    root = Path(__file__).resolve().parent
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('-s','--save',choices=['llama','aya'],default='llama')
    p.add_argument('-m','--model')
    p.add_argument('--activation-mask',type=Path)
    p.add_argument('--activations-dir',type=Path)
    p.add_argument('--output',type=Path,default=Path('generations/lape'))
    group=p.add_mutually_exclusive_group()
    group.add_argument('--track',choices=['t2'])
    group.add_argument('--pairs',nargs='+',help='e.g. en:sw ar:sw')
    p.add_argument('--source-lang',default='en')
    p.add_argument('--target-lang',default='sw')
    p.add_argument('--activation-method',choices=['additive','replacement','diffmean'],default='additive')
    p.add_argument('--activation-strength',type=float,default=1.)
    p.add_argument('--deactivation-strength',type=float,default=0.)
    p.add_argument('--no-deactivation',action='store_true')
    p.add_argument('--max-model-len',type=int,default=4096)
    p.add_argument('--gpu-memory-utilization',type=float,default=0.90)
    p.add_argument('--tensor-parallel-size',type=int,default=None)
    p.add_argument('--batch-size',type=int,default=1)
    p.add_argument('--max-new-tokens',type=int,default=1024)
    p.add_argument('--k-questions',type=int,default=70)
    p.add_argument('--data-dir',type=Path,help='Validation question directory')
    p.add_argument('--overwrite',action='store_true')
    a=p.parse_args()
    if min(a.batch_size,a.max_new_tokens,a.k_questions)<1:p.error('Batch, tokens, questions must be positive')
    a.model=a.model or ('meta-llama/Llama-3.1-8B-Instruct' if a.save=='llama' else 'CohereLabs/tiny-aya-global')
    mask_path=a.activation_mask or root/'steering'/'activation_mask'/f'{a.save}.lape.pt'
    folder=a.activations_dir or root/'steering'/f'data_{a.save}'
    metadata=json.loads(mask_path.with_suffix(mask_path.suffix+'.json').read_text())
    languages=metadata['languages']; mapping={l:i for i,l in enumerate(languages)}
    masks=torch.load(mask_path,map_location='cpu',weights_only=True)
    if len(masks)!=len(languages) or len(set(languages))!=len(languages):raise ValueError('Invalid mask language order')
    means=[]
    for language in languages:
        data=torch.load(folder/f'activation.{language}.pt',map_location='cpu',weights_only=True)
        if data.get('model')!=a.model:raise ValueError(f'{language}: activation model differs from generation model')
        expected=next(x for x in metadata['sources'] if x['language']==language)
        if hashlib.sha256((folder/f'activation.{language}.pt').read_bytes()).hexdigest()!=expected['sha256']:
            raise ValueError(f'{language}: activation file changed since identification')
        means.append(data['average_activations'].float())
    means=torch.stack(means,-1)
    if not torch.isfinite(means).all():raise ValueError('Nonfinite mean activations')
    for lang_mask in masks:
        if len(lang_mask)!=means.shape[0]:raise ValueError('Mask layer mismatch')
        for indices in lang_mask:
            if indices.ndim!=1 or indices.dtype!=torch.long or (indices<0).any() or (indices>=means.shape[1]).any():raise ValueError('Invalid neuron indices')
    if a.track:
        from clas import languages as L
        pairs=list(L.pairs(a.track))
    else:pairs=[tuple(x.split(':')) for x in a.pairs] if a.pairs else [(a.source_lang,a.target_lang)]
    for pair in pairs:
        if len(pair)!=2 or any(l not in mapping for l in pair):raise ValueError(f'Pair not covered by masks: {pair}')
    from utils.utils import get_test_questions
    questions=get_test_questions(data_dir=str(a.data_dir) if a.data_dir else None,k=a.k_questions)
    signature=dict(model=a.model,method=a.activation_method,activation_strength=a.activation_strength,
                   deactivation_strength=a.deactivation_strength,no_deactivation=a.no_deactivation,
                   max_new_tokens=a.max_new_tokens,k_questions=a.k_questions,
                   mask_sha256=hashlib.sha256(mask_path.read_bytes()).hexdigest(),
                   metadata_sha256=hashlib.sha256(mask_path.with_suffix(mask_path.suffix+'.json').read_bytes()).hexdigest())
    a.output.mkdir(parents=True,exist_ok=True)
    jobs=[]
    for source,target in pairs:
        prompts=questions[source]
        if len(prompts)!=a.k_questions:raise ValueError(f'{source}: expected {a.k_questions} questions, got {len(prompts)}')
        path=a.output/f'{source}_to_{target}.json'
        if path.exists() and not a.overwrite:
            old=json.loads(path.read_text())
            if old.get('run_config')!=signature or [r['original_prompt'] for r in old['results']]!=prompts:
                raise ValueError(f'{path}: settings/questions differ; use separate --output or --overwrite')
            print(f'Skip {source} -> {target}: complete matching output');continue
        jobs.append((source,target,prompts,path))
    if not jobs:return
    # Same opt-in required by this repo's diffmean_generate.py apply_model path.
    os.environ.setdefault('VLLM_ALLOW_INSECURE_SERIALIZATION','1')
    from vllm import LLM,SamplingParams
    from utils.lape_vllm_hooks import install_lape,set_lape
    engine=LLM(model=a.model,tensor_parallel_size=a.tensor_parallel_size or torch.cuda.device_count(),
               max_model_len=a.max_model_len,gpu_memory_utilization=a.gpu_memory_utilization,
               dtype='float16',enforce_eager=True,enable_prefix_caching=False)
    installed=engine.apply_model(install_lape)
    if any(n!=means.shape[0] for n in installed):raise ValueError('Model/activation layer mismatch')
    sampling=SamplingParams(temperature=0.,max_tokens=a.max_new_tokens,repetition_penalty=1.,skip_special_tokens=True)
    for source,target,prompts,path in jobs:
        targets=masks[mapping[target]]
        boosts=[]
        for idx,indices in enumerate(targets):
            values=means[idx,indices,mapping[target]]
            if a.activation_method=='diffmean':
                others=[i for i in range(len(languages)) if i!=mapping[target]]
                values=values-means[idx,indices][:,others].mean(-1)
            boosts.append(values)
        try:
            engine.apply_model(partial(set_lape,None if a.no_deactivation else masks[mapping[source]],
                                       targets,boosts,means.shape[1],a.activation_method,
                                       a.deactivation_strength,a.activation_strength))
            messages=[[{'role':'user','content':q}] for q in prompts]
            generated=engine.chat(messages,sampling_params=sampling)
            responses=[o.outputs[0].text for o in generated]
            if len(responses)!=len(prompts):raise RuntimeError('Incomplete generation')
        finally:
            engine.apply_model(partial(set_lape,None,None,None,means.shape[1],a.activation_method,0.,0.))
        result=dict(source_language=source,target_language=target,model=a.model,track=a.track or 'custom',
                    method=f'lape:{a.activation_method}',run_config=signature,
                    results=[dict(question_idx=i,input=q,original_prompt=q,output=r.strip()) for i,(q,r) in enumerate(zip(prompts,responses))])
        temporary=path.with_suffix('.json.tmp');temporary.write_text(json.dumps(result,ensure_ascii=False,indent=2));temporary.replace(path)
        print(f'Saved {path}',flush=True)


if __name__=='__main__':main()

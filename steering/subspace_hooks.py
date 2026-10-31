"""HF and vLLM activation hooks. vLLM functions must remain module-level."""
from functools import partial

import torch
from steering.experiment_utils import decoder_layers
from steering.subspace import intervene


def edit(h, state):
    # Accelerate may offload weights: use the actual activation device, never meta.
    for key in ('basis', 'centers', 'variance', 'direction'):
        if key in state and state[key].device != h.device:
            state[key] = state[key].to(h.device)
    if state['method'] == 'diffmean':
        return h + state['direction'].to(h.dtype) * state['strength']
    return intervene(h, state['basis'], state['centers'], state['variance'],
                     state['target'], state['strength'], state['method'] == 'adaptive',
                     state['threshold'], state['temperature'], state['max_relative_norm'])


def layer_state(artifact, slot, source, target, options, device):
    state = {**options, 'target': target}
    for key in ('basis', 'centers', 'variance'):
        state[key] = artifact[key][slot].to(device=device, dtype=torch.float32)
    diff = artifact['means'][target, slot] - artifact['means'][source, slot]
    state['direction'] = torch.nn.functional.normalize(diff, dim=-1).to(device)
    return state


def hf_hook(state, scope, module, inputs, output):
    h = output[0] if isinstance(output, (tuple, list)) else output
    if scope == 'last':
        # Left-padded batches: final position is the final real prompt token.
        h = torch.cat((h[:, :-1], edit(h[:, -1:], state)), dim=1)
    else:
        h = edit(h, state)
    return (h,) + tuple(output[1:]) if isinstance(output, (tuple, list)) else h


def install_hf(model, artifact, source, target, selected, options, scope):
    layers, handles = decoder_layers(model), []
    try:
        for layer in selected:
            slot = artifact['layers'].index(layer)
            device = torch.device('cpu')
            state = layer_state(artifact, slot, source, target, options, device)
            handles.append(layers[layer].register_forward_hook(partial(hf_hook, state, scope)))
    except Exception:
        for handle in handles:
            handle.remove()
        raise
    return handles


def install_vllm(artifact, source, target, selected, options, model):
    """vLLM uses packed tokens: this backend intentionally supports all-token scope.

    Llama V1 decoder returns (hidden, residual). Their SUM is the full stream
    captured in HF extraction; compute the edit there, then add only delta to
    hidden. Editing hidden alone would target the wrong representation.
    """
    layers = decoder_layers(model)
    for layer in selected:
        block = layers[layer]
        if getattr(block, '_subspace_handle', None) is not None:
            raise RuntimeError('Subspace hook already installed')
    try:
        for layer in selected:
            block = layers[layer]
            state = layer_state(artifact, artifact['layers'].index(layer), source, target,
                                options, next(block.parameters()).device)
            block._subspace_handle = block.register_forward_hook(partial(vllm_hook, state))
    except Exception:
        clear_vllm(model)
        raise
    return len(selected)


def vllm_hook(state, module, inputs, output):
    if not isinstance(output, tuple) or len(output) != 2:
        raise RuntimeError('Unsupported vLLM decoder output; expected (hidden, residual)')
    hidden, residual = output
    if residual is None or hidden.shape != residual.shape:
        raise RuntimeError('Unsupported vLLM residual layout')
    full = hidden.float() + residual.float()
    edited = edit(full, state)
    return hidden + (edited - full).to(hidden.dtype), residual


def clear_vllm(model):
    for block in decoder_layers(model):
        handle = getattr(block, '_subspace_handle', None)
        if handle is not None:
            handle.remove()
            block._subspace_handle = None

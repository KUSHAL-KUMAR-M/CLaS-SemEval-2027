"""Worker-side LAPE for dense SiLU gated vLLM Llama/Aya MLPs."""
import torch
import torch.nn.functional as F
from utils.steering_hooks import get_model_layers


def install_lape(model):
    layers=get_model_layers(model)
    for layer in layers:
        mlp=layer.mlp
        if getattr(mlp,'_lape_installed',False):continue
        if not all(hasattr(mlp,k) for k in ('gate_up_proj','down_proj','act_fn')) or 'silu' not in type(mlp.act_fn).__name__.lower():
            raise RuntimeError('LAPE backend requires dense SiLU gate_up_proj MLP')
        original=mlp.forward
        def forward(x, _mlp=mlp, _original=original):
            state=_mlp._lape_state
            if state is None:return _original(x)
            packed,_=_mlp.gate_up_proj(x)
            gate,up=packed.chunk(2,dim=-1)
            activated=F.silu(gate)
            source,target,boost,method,deactivate,strength=state
            if source.numel():activated.index_fill_(-1,source,deactivate)
            if target.numel():
                if method=='replacement':activated[...,target]=strength*boost
                else:activated[...,target]+=strength*boost
            out,_=_mlp.down_proj(activated*up)
            return out
        mlp._lape_state=None
        mlp.forward=forward
        mlp._lape_installed=True
    return len(layers)


def set_lape(source_masks,target_masks,boosts,width,method,deactivate,strength,model):
    from vllm.distributed import get_tensor_model_parallel_rank,get_tensor_model_parallel_world_size
    rank=get_tensor_model_parallel_rank();world=get_tensor_model_parallel_world_size()
    if width%world:raise ValueError('Intermediate width must divide tensor parallel size')
    local=width//world;offset=rank*local
    for idx,layer in enumerate(get_model_layers(model)):
        mlp=layer.mlp
        if target_masks is None:
            mlp._lape_state=None;continue
        weight=mlp.gate_up_proj.weight
        if weight.shape[0]!=2*local:raise ValueError('Unexpected fused gate/up sharding')
        source=source_masks[idx] if source_masks is not None else torch.empty(0,dtype=torch.long)
        source=source[(source>=offset)&(source<offset+local)]-offset
        target=target_masks[idx];keep=(target>=offset)&(target<offset+local)
        target=target[keep]-offset
        mlp._lape_state=(source.to(weight.device),target.to(weight.device),
                         boosts[idx][keep].to(device=weight.device,dtype=weight.dtype),method,deactivate,strength)
    return world

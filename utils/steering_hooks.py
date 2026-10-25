"""Functions that run INSIDE the vLLM worker processes (called through LLM.apply_model).

They live in their own importable module (not in the main script) because vLLM
re-pickles them on the way to the workers, and plain pickle can only send
module-level functions by reference. Save this file as utils/steering_hooks.py.
"""


def get_model_layers(model):
    """Get model layers with compatibility for different model architectures"""
    try:
        return model.model.layers                     # Llama / Aya style
    except AttributeError:
        try:
            return model.language_model.model.layers  # Gemma-3 style
        except AttributeError:
            raise RuntimeError("Could not access model layers. Model architecture may not be supported.")


def install_hooks(model):
    """Wrap every layer's forward once. Steering stays off until set_steering() is called."""
    layers = get_model_layers(model)
    for layer in layers:
        if getattr(layer, "_steering_installed", False):
            continue
        original = layer.forward

        def forward_hook(*args, _original=original, _layer=layer, **kwargs):
            outputs = _original(*args, **kwargs)
            vec = _layer._steering_vec
            if vec is None:
                return outputs
            # vLLM layers return (hidden_states, residual); the residual stream is their sum,
            # so adding to hidden_states shifts the stream.
            if isinstance(outputs, (tuple, list)):
                return (outputs[0] + vec,) + tuple(outputs[1:])
            return outputs + vec

        layer.forward = forward_hook
        layer._steering_installed = True
        layer._steering_vec = None
    return len(layers)


def set_steering(directions, layer_start, layer_end, model):
    """Set (or clear) the vector added at each layer.

    directions: [num_layers, hidden_size] already scaled by the strength, or None = steering off.
    Used as functools.partial(set_steering, directions, layer_start, layer_end) -> func(model).
    """
    layers = get_model_layers(model)
    for idx, layer in enumerate(layers):
        if directions is not None and layer_start <= idx < layer_end:
            p = next(layer.parameters())
            layer._steering_vec = directions[idx].to(device=p.device, dtype=p.dtype)
        else:
            layer._steering_vec = None
    return sum(layer._steering_vec is not None for layer in layers)

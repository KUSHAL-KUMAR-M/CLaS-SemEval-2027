"""Closed-loop (projection-matching) DiffMean steering for Hugging Face models.

Save as utils/closed_loop.py

Instead of adding a fixed amount (strength x unit vector) at every token, the controller
measures where each hidden state currently is on the source -> target axis and pushes only
the missing part:

    d    = unit(mean[target] - mean[source])     language direction at this layer
    gap  = || mean[target] - mean[source] ||     distance between the two language means
    progress(h) = (h . d - mean[source] . d) / gap
                  0 = at the source-language mean, 1 = at the target-language mean
    push = min(max_push, gain * max(0, goal - progress(h)))        (in units of gap)
    h   <- h + push * gap * d

A state that is already far enough along gets no push, so the push fades out by itself once
the model is writing in the target language. No strength or decay schedule is needed.
"""
import torch


class LoopStats:
    """Collects what the controller did, per layer (kept on the GPU until the end)."""

    def __init__(self):
        self.data = {}   # layer -> tensor [tokens, sum of progress, sum of push, tokens pushed]

    def add(self, layer, progress, push):
        row = torch.stack([
            progress.new_tensor(float(progress.numel())),
            progress.sum(),
            push.sum(),
            (push > 0).sum().to(progress.dtype),
        ])
        if layer in self.data:
            self.data[layer] += row
        else:
            self.data[layer] = row

    def summary(self, gaps):
        lines = []
        for layer in sorted(self.data):
            n, sum_progress, sum_push, n_pushed = self.data[layer].tolist()
            if n == 0:
                continue
            lines.append(
                f"layer {layer}: gap {gaps[layer]:.2f} | "
                f"mean progress before correction {sum_progress / n:+.2f} | "
                f"corrected {100 * n_pushed / n:.0f}% of tokens | "
                f"mean push {sum_push / n:.2f} gap")
        return lines


def create_closed_loop_hook(layer_idx, d_unit, p_source, gap, goal, gain, max_push,
                            steer_prompt, stats):
    """Forward hook for one decoder layer.

    prompt pass (seq_len > 1):
        steer_prompt=False -> only the LAST prompt token is controlled (it produces the first answer token)
        steer_prompt=True  -> every prompt token except the first one (the attention sink)
    every generated token (seq_len == 1) is controlled.
    """
    def hook(module, inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        is_prefill = hs.dim() == 3 and hs.size(1) > 1

        if is_prefill:
            if steer_prompt:
                keep, x = hs[:, :1, :], hs[:, 1:, :]
            else:
                keep, x = hs[:, :-1, :], hs[:, -1:, :]
        else:
            keep, x = None, hs

        xf = x.float()
        progress = (xf @ d_unit - p_source) / gap                 # [batch, tokens]
        need = (goal - progress).clamp(min=0)                     # missing distance, in gaps
        push = (gain * need).clamp(max=max_push)
        stats.add(layer_idx, progress, push)

        x_new = (xf + (push * gap).unsqueeze(-1) * d_unit).to(x.dtype)
        hs_new = x_new if keep is None else torch.cat([keep, x_new], dim=1)

        if isinstance(output, tuple):
            return (hs_new,) + tuple(output[1:])
        return hs_new
    return hook


def install_closed_loop(layers, source_mean, target_mean, layer_indices,
                        goal=0.8, gain=1.0, max_push=1.0, steer_prompt=False):
    """Register the controller on the given layers.

    source_mean, target_mean: [num_layers, hidden_size] mean activations (from diffmean.py)
    layer_indices: the layers to control, e.g. range(20, 21)
    Returns (hook handles, LoopStats, {layer: gap}). Remove the hooks with handle.remove().
    """
    stats = LoopStats()
    handles = []
    gaps = {}
    try:
        for idx in layer_indices:
            diff = target_mean[idx].float() - source_mean[idx].float()
            gap = diff.norm(p=2).item()
            if gap < 1e-6:
                print(f"Closed loop: layer {idx} skipped (source and target means are identical)")
                continue
            d_unit = diff / gap
            p_source = float(source_mean[idx].float() @ d_unit)
            device = next(layers[idx].parameters()).device
            d_dev = d_unit.to(device=device, dtype=torch.float32)
            gaps[idx] = gap
            hook = create_closed_loop_hook(idx, d_dev, p_source, gap, goal, gain,
                                           max_push, steer_prompt, stats)
            handles.append(layers[idx].register_forward_hook(hook))
    except Exception:
        for h in handles:
            h.remove()
        raise
    return handles, stats, gaps

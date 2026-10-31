"""Language-subspace fitting and bounded, state-dependent activation edits."""
import torch

FORMAT_VERSION = 1


def fit_subspace(activations, rank):
    """activations: [languages, aligned sentences, layers, hidden].

    Equal weight per sentence and language. SVD is fitted to language centroids
    after subtracting the across-language mean for each aligned sentence.
    """
    x = activations.float().cpu()
    if x.ndim != 4 or min(x.shape[:2]) < 2 or not torch.isfinite(x).all():
        raise ValueError("Need finite [languages>=2, sentences>=2, layers, hidden] activations")
    if not 1 <= rank <= min(x.shape[0] - 1, x.shape[-1]):
        raise ValueError("rank must be between 1 and min(n_languages-1, hidden_size)")
    centered = x - x.mean(dim=0, keepdim=True)
    means = x.mean(dim=1)
    bases, centers, scales = [], [], []
    for layer in range(x.shape[2]):
        contrasts = centered[:, :, layer].mean(dim=1)
        _, singular, vh = torch.linalg.svd(contrasts, full_matrices=False)
        if singular[rank - 1] <= singular[0] * 1e-6:
            raise ValueError(f"Requested rank exceeds numerical language rank at layer {layer}")
        u = vh[:rank].T.contiguous()
        z = x[:, :, layer] @ u
        # Shared diagonal within-language variance: centroid confidence proxy.
        variance = (z - z.mean(dim=1, keepdim=True)).square().mean(dim=(0, 1))
        bases.append(u)
        centers.append(means[:, layer] @ u)
        scales.append(variance.clamp_min(1e-4))
    return {"basis": torch.stack(bases), "centers": torch.stack(centers),
            "variance": torch.stack(scales), "means": means}


def language_probability(z, centers, variance, target, temperature):
    distances = ((z.unsqueeze(-2) - centers).square() / variance).mean(dim=-1)
    return (-distances / temperature).softmax(dim=-1)[..., target]


def intervene(h, basis, centers, variance, target, strength=0.5,
              adaptive=True, threshold=0.8, temperature=1.0, max_relative_norm=0.1):
    """Edit only span(basis). Confidence is a centroid proxy, not calibrated LID.

    All decisions use pre-intervention activations. Relative norm clipping caps
    perturbations per token and preserves the subspace constraint.
    """
    x = h.float()
    z = x @ basis
    gate = torch.ones_like(z[..., :1])
    if adaptive:
        p = language_probability(z, centers, variance, target, temperature)
        gate = ((threshold - p) / threshold).clamp(0, 1).unsqueeze(-1)
    delta = (strength * gate * (centers[target] - z)) @ basis.T
    cap = max_relative_norm * x.norm(dim=-1, keepdim=True)
    delta *= (cap / delta.norm(dim=-1, keepdim=True).clamp_min(1e-12)).clamp(max=1)
    return (x + delta).to(h.dtype)


def validate_artifact(data, model_id):
    if data.get("format_version") != FORMAT_VERSION or data.get("model") != model_id:
        raise ValueError("Artifact format/model mismatch; extract separately for each model")
    layers, hidden, rank = data["basis"].shape
    langs = len(data["languages"])
    if (data["centers"].shape != (layers, langs, rank)
            or data["variance"].shape != (layers, rank)
            or data["means"].shape != (langs, layers, hidden)
            or len(data["layers"]) != layers):
        raise ValueError("Inconsistent artifact tensor shapes")
    for key in ("basis", "centers", "variance", "means"):
        if not torch.isfinite(data[key]).all():
            raise ValueError(f"Non-finite {key}")
    if (data["variance"] <= 0).any():
        raise ValueError("Variance must be positive")
    identity = torch.eye(rank).expand(layers, -1, -1)
    if not torch.allclose(data["basis"].transpose(-1, -2) @ data["basis"], identity, atol=1e-4):
        raise ValueError("Basis must be orthonormal")

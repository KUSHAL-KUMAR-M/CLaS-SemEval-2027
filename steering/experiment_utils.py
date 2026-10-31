"""Shared data selection, model discovery, provenance, and atomic output."""
import hashlib
import importlib.metadata
import json
from pathlib import Path


def decoder_layers(model):
    for path in ("model.layers", "model.language_model.layers",
                 "model.language_model.model.layers", "language_model.model.layers"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            pass
    raise RuntimeError("Unsupported decoder layout; cannot safely install intervention")


def read_aligned(directory, languages):
    """Preserve line positions: never silently remove a blank alignment row."""
    rows = {}
    for lang in languages:
        path = Path(directory) / f"{lang}.txt"
        lines = path.read_text(encoding="utf-8").splitlines()
        if not lines or any(not line.strip() for line in lines):
            raise ValueError(f"Empty/blank alignment row in {path}")
        rows[lang] = lines
    if len({len(v) for v in rows.values()}) != 1:
        raise ValueError("Parallel language files must have identical line counts")
    return rows


def row_indices(total, split, seed=42, holdout_fraction=0.3):
    """Hash split by shared row ID, independent of language and total row count."""
    if split not in ("all", "tune", "holdout") or not 0 < holdout_fraction < 1:
        raise ValueError("Invalid split or holdout fraction")
    out = []
    for i in range(total):
        score = int(hashlib.sha256(f"{seed}:{i}".encode()).hexdigest()[:16], 16) / 2**64
        held = score < holdout_fraction
        if split == "all" or held == (split == "holdout"):
            out.append(i)
    return out


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def versions():
    result = {}
    for name in ("torch", "transformers", "accelerate", "vllm", "numpy"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def write_json_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(path)

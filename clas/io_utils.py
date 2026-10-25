import hashlib
import json
from pathlib import Path


def write_lines(path: Path, lines: list) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(l.strip().replace("\n", " ") for l in lines if l.strip())
    path.write_text(text + "\n", encoding="utf-8")
    return len(text.splitlines())


def read_lines(path: Path, k: int = None) -> list:
    lines = [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return lines[:k] if k else lines


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_questions(questions_dir: Path, codes: list, k: int = None) -> dict:
    """Load {code: [question, ...]} from <dir>/<code>.txt."""
    out = {}
    for code in codes:
        path = questions_dir / f"{code}.txt"
        if not path.exists():
            raise FileNotFoundError(f"missing questions for '{code}': {path}")
        out[code] = read_lines(path, k)
    return out

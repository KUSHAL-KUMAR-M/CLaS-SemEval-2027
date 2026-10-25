#!/usr/bin/env python3
"""Build the CLaS shared-task pilot data.

train/      FLORES-200 dev+devtest per language (2009 sentences), for
            identifying language-specific components.
validation/ CLaS-Bench questions (70 parallel open-ended questions).

Usage
-----
python download_data.py --out data
"""

import argparse
import sys
import urllib.request
from pathlib import Path

from clas import languages as L
from clas.io_utils import sha256, write_json, write_lines

CLAS_REPO = "DGurgurov/CLaS-Bench"
CLAS_RAW = "https://raw.githubusercontent.com/d-gurgurov/CLaS-Bench/main/CLaS-Bench/data/{code}.txt"
_TABLE = {}


# --------------------------------------------------------------------------- #
# train: FLORES-200
# --------------------------------------------------------------------------- #
def fetch_flores(code: str, splits: list, dataset: str) -> list:
    from datasets import load_dataset

    configs = list(dict.fromkeys([
        L.flores_config(code, dataset),
        L.flores_config(code, "flores_plus"),
        L.get(code).flores,
    ]))
    sents, errors = [], []
    for config in configs:
        try:
            for split in splits:
                try:
                    ds = load_dataset(dataset, config, split=split)
                except Exception:
                    ds = load_dataset(dataset, config, split=split, trust_remote_code=True)
                col = "text" if "text" in ds.column_names else f"sentence_{config}"
                sents += [s for s in ds[col] if s and s.strip()]
            return sents
        except Exception as e:
            errors.append(f"{config}: {e}")
            sents = []
    raise RuntimeError("; ".join(errors))


# --------------------------------------------------------------------------- #
# validation: CLaS-Bench questions
# --------------------------------------------------------------------------- #
def fetch_clas_hf(code: str, repo: str) -> list:
    from datasets import load_dataset

    ids = L.clas_codes(code)
    for cid in ids:
        try:
            ds = load_dataset(repo, split=cid)
            if len(ds):
                return _ordered(ds)
        except Exception:
            pass

    if "ds" not in _TABLE:
        for split in ("all", "train"):
            try:
                _TABLE["ds"] = load_dataset(repo, split=split)
                break
            except Exception as e:
                err = e
        else:
            raise RuntimeError(f"no usable split in {repo}: {err}")

    for cid in ids:
        ds = _TABLE["ds"].filter(lambda r: r["language_code"] == cid)
        if len(ds):
            return _ordered(ds)
    raise ValueError(f"no rows for any language_code in {ids} ({repo})")


def _ordered(ds) -> list:
    by_id = {}
    for r in sorted(ds, key=lambda r: r["question_id"]):
        by_id.setdefault(r["question_id"], r["question"]) 
    return list(by_id.values())


def fetch_clas(code: str, source: str, repo: str, local_dir: Path) -> list:
    if source == "hf":
        return fetch_clas_hf(code, repo)
    errors = []
    for cid in L.clas_codes(code):
        try:
            if source == "local":
                text = (local_dir / f"{cid}.txt").read_text(encoding="utf-8")
            else:
                with urllib.request.urlopen(CLAS_RAW.format(code=cid), timeout=60) as r:
                    text = r.read().decode("utf-8")
            return [l for l in text.splitlines() if l.strip()]
        except Exception as e:
            errors.append(f"{cid}: {e}")
    raise RuntimeError("; ".join(errors))


# --------------------------------------------------------------------------- #
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("data"))
    p.add_argument("--langs", nargs="+", default=L.ALL, help="ISO 639-1 codes")
    p.add_argument("--flores-dataset", default="openlanguagedata/flores_plus")
    p.add_argument("--splits", nargs="+", default=["dev", "devtest"])
    p.add_argument("--clas-source", choices=["hf", "github", "local"], default="hf")
    p.add_argument("--clas-repo", default=CLAS_REPO)
    p.add_argument("--clas-local-dir", type=Path, default=Path("CLaS-Bench/CLaS-Bench/data"))
    p.add_argument("--expect-questions", type=int, default=70,
                   help="warn if a language yields a different count (0 disables)")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-validation", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    unknown = [c for c in args.langs if c not in L.REGISTRY]
    if unknown:
        p.error(f"not in registry: {unknown}")

    manifest, failures = {}, []
    for code in args.langs:
        lang = L.get(code)
        entry = {"name": lang.name, "flores": lang.flores,
                 "flores_config": L.flores_config(code, args.flores_dataset),
                 "joshi": lang.joshi,
                 "pool": "high" if code in L.HIGH else "low" if code in L.LOW else "spare"}

        for kind, skip, fetch in [
            ("train", args.skip_train, lambda: fetch_flores(code, args.splits, args.flores_dataset)),
            ("validation", args.skip_validation,
             lambda: fetch_clas(code, args.clas_source, args.clas_repo, args.clas_local_dir)),
        ]:
            if skip:
                continue
            path = args.out / kind / f"{code}.txt"
            if path.exists() and not args.overwrite:
                print(f"= {code:>3} {kind:<10} exists, skipping")
                entry[kind] = {"file": str(path), "n": len(path.read_text('utf-8').splitlines())}
                continue
            try:
                n = write_lines(path, fetch())
                entry[kind] = {"file": str(path), "n": n, "sha256": sha256(path)}
                flag = ""
                if kind == "validation" and args.expect_questions and n != args.expect_questions:
                    flag = f"  <-- expected {args.expect_questions}"
                print(f"+ {code:>3} {kind:<10} {n:>5} lines -> {path}{flag}")
            except Exception as e:
                failures.append((code, kind, repr(e)))
                print(f"! {code:>3} {kind:<10} FAILED: {e}", file=sys.stderr)

        manifest[code] = entry

    write_json(args.out / "manifest.json", {
        "flores_dataset": args.flores_dataset,
        "flores_splits": args.splits,
        "clas_source": args.clas_source,
        "clas_repo": args.clas_repo if args.clas_source == "hf" else None,
        "languages": manifest,
    })
    print(f"\nmanifest -> {args.out / 'manifest.json'}")
    if failures:
        print(f"{len(failures)} failure(s):", file=sys.stderr)
        for f in failures:
            print(f"  {f}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
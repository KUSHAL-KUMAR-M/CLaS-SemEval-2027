#!/usr/bin/env python3
"""Score CLaS generations: LFS, OR, and the primary metric LSS.

LFS  Language Forcing Success - fraction of outputs whose fastText LID label
     matches the target language, in [0, 1].
OR   Output Relevance - LLM-as-a-judge score in {0, 1, 2}, normalised to [0, 1].
LSS  Language Steering Score - harmonic mean of LFS and OR.

Usage
-----
python evaluate.py --gen-dir generations/Llama-3.1-8B-Instruct/t1/english
python evaluate.py --gen-dir <dir> --skip-judge      # LFS only, fast sanity check
"""

import argparse
import re
from pathlib import Path

from clas import languages as L
from clas.io_utils import read_json, write_json

JUDGE_SYSTEM = """You are an impartial judge evaluating a response from a language steering experiment. A question was asked in {source}, and the model was steered to respond in {target}.

Do NOT evaluate which language the response is in or whether steering succeeded. Judge the response solely on its content, regardless of language.

Score on relevance and coherence:
- 0: completely unrelated to the question, or gibberish/degenerate text.
- 1: somewhat related, but incomplete, partially off-target, or of little use.
- 2: clearly related to the question, coherent, and useful.

Give a brief explanation, then the rating in exactly this format: "Rating: [[score]]"."""

JUDGE_USER = "Question ({source}):\n{question}\n\nResponse ({target}):\n{response}\n"


def harmonic(a: float, b: float) -> float:
    return 0.0 if a + b == 0 else 2 * a * b / (a + b)


# --------------------------------------------------------------------------- #
# LFS
# --------------------------------------------------------------------------- #
def load_lid(repo: str = "facebook/fasttext-language-identification"):
    import fasttext
    from huggingface_hub import hf_hub_download

    return fasttext.load_model(hf_hub_download(repo, "model.bin"))


def script_lang(text: str, min_ratio: float = 0.9):
    """Identify zh/ja/ko from script alone; None if the text is not (almost) purely CJK.

    fastText is unreliable on short CJK text (e.g. one-sentence Chinese answers get
    split across zho_Hans / vie_Latn / eng_Latn), but these scripts are unambiguous.
    """
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return None
    han = sum("一" <= c <= "鿿" or "㐀" <= c <= "䶿" for c in letters)
    kana = sum("぀" <= c <= "ヿ" for c in letters)
    hangul = sum("가" <= c <= "힯" or "ᄀ" <= c <= "ᇿ" or "㄰" <= c <= "㆏"
                 for c in letters)
    n = len(letters)
    if hangul / n >= min_ratio:
        return "ko"
    if (han + kana) / n >= min_ratio:
        return "ja" if kana else "zh"
    return None


def predict_lang(lid, text: str, mapping: dict, min_confidence: float = 0.9) -> str:
    text = " ".join(text.split())
    if not text:
        return "unk"
    labels, probs = lid.predict(text, k=1)
    if probs[0] < min_confidence:
        return script_lang(text) or "unk"
    label = labels[0].replace("__label__", "")
    return mapping.get(label, label)


# --------------------------------------------------------------------------- #
# OR
# --------------------------------------------------------------------------- #
def parse_rating(text: str):
    for pattern in (r"Rating:\s*\[\[([012])\]\]", r"\[\[([012])\]\]", r"\b([012])\b\s*$"):
        m = re.search(pattern, text.strip())
        if m:
            return int(m.group(1))
    return None


def judge_all(files: list, model, params, thinking: bool = False) -> dict:
    """Score every (file, item) in one batched pass. Returns {file: [scores]}."""
    msgs, index = [], []
    for f in files:
        data = read_json(f)
        src, tgt = L.get(data["source_language"]).name, L.get(data["target_language"]).name
        for i, r in enumerate(data["results"]):
            msgs.append([
                {"role": "system", "content": JUDGE_SYSTEM.format(source=src, target=tgt)},
                {"role": "user", "content": JUDGE_USER.format(
                    source=src, target=tgt, question=r["original_prompt"], response=r["output"])},
            ])
            index.append((f, i))

    try:
        outputs = model.chat(msgs, sampling_params=params,
                             chat_template_kwargs={"enable_thinking": thinking})
    except Exception:
        outputs = model.chat(msgs, sampling_params=params)

    scores = {f: [] for f in files}
    for (f, _), o in zip(index, outputs):
        scores[f].append(parse_rating(o.outputs[0].text))
    return scores


# --------------------------------------------------------------------------- #
def report(pair_scores: dict, out_dir: Path) -> None:
    """Aggregate per pair, per target language, and overall; write json + md."""
    def agg(rows):
        n = len(rows)
        lfs = sum(r["lfs"] for r in rows) / n
        orl = sum(r["or"] for r in rows) / n
        return {"lfs": lfs, "or": orl, "lss": harmonic(lfs, orl), "n_pairs": n}

    rows = list(pair_scores.values())
    by_target = {}
    for tgt in sorted({r["target"] for r in rows}):
        by_target[tgt] = agg([r for r in rows if r["target"] == tgt])

    summary = {"overall": agg(rows), "by_target_language": by_target,
               "by_pair": pair_scores}
    write_json(out_dir / "scores.json", summary)

    o = summary["overall"]
    lines = ["# CLaS baseline results", "",
             f"Pairs: {o['n_pairs']}", "",
             "| scope | LFS | OR | LSS |", "|---|---|---|---|",
             f"| **overall** | {o['lfs']:.3f} | {o['or']:.3f} | **{o['lss']:.3f}** |"]
    for tgt, s in sorted(by_target.items(), key=lambda kv: -kv[1]["lss"]):
        lines.append(f"| → {L.get(tgt).name} ({tgt}) | {s['lfs']:.3f} | {s['or']:.3f} | {s['lss']:.3f} |")
    (out_dir / "scores.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines[3:]))
    print(f"\nwritten -> {out_dir / 'scores.json'}, {out_dir / 'scores.md'}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gen-dir", type=Path, required=True, help="dir of <src>_to_<tgt>.json")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--judge-model", default="Qwen/Qwen3-8B")
    p.add_argument("--judge-thinking", action="store_true",
                   help="allow reasoning traces in judge output (slower)")
    p.add_argument("--skip-judge", action="store_true", help="LFS only")
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--lid-min-confidence", type=float, default=0.9,
                   help="fastText LID probability threshold; below this an output counts as not matching the target language")
    args = p.parse_args()

    out_dir = args.out_dir or args.gen_dir
    files = sorted(args.gen_dir.glob("*_to_*.json"))
    if not files:
        p.error(f"no generation files in {args.gen_dir}")
    print(f"scoring {len(files)} language pairs from {args.gen_dir}")

    # --- LFS ---
    lid, mapping = load_lid(), L.flores_to_code()
    lfs = {}
    for f in files:
        data = read_json(f)
        tgt = data["target_language"]
        hits = [predict_lang(lid, r["output"], mapping, args.lid_min_confidence) == tgt for r in data["results"]]
        lfs[f] = sum(hits) / len(hits)

    # --- OR ---
    if args.skip_judge:
        judged = {f: [] for f in files}
    else:
        import torch
        from vllm import LLM, SamplingParams

        judge = LLM(model=args.judge_model, tensor_parallel_size=max(1, torch.cuda.device_count()),
                    max_model_len=args.max_model_len, dtype="float16",
                    gpu_memory_utilization=args.gpu_memory_utilization, trust_remote_code=True)
        judged = judge_all(files, judge,
                           SamplingParams(temperature=0.0, max_tokens=512, seed=0),
                           thinking=args.judge_thinking)

    pair_scores = {}
    for f in files:
        data = read_json(f)
        valid = [s for s in judged[f] if s is not None]
        orl = (sum(valid) / (2 * len(valid))) if valid else 0.0
        pair_scores[f"{data['source_language']}_to_{data['target_language']}"] = {
            "source": data["source_language"], "target": data["target_language"],
            "lfs": lfs[f], "or": orl, "lss": harmonic(lfs[f], orl),
            "n_items": len(data["results"]), "n_unparsed_judge": len(judged[f]) - len(valid),
        }

    report(pair_scores, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
#!/usr/bin/env python3
"""Prompting baseline for the shared task.

python run_baseline.py --track t1 --model meta-llama/Llama-3.1-8B-Instruct
python run_baseline.py --track t3 --t3-target ja --instruction target_native
python run_baseline.py --track bonus                 # refusal track, uses the refusal instruction
python run_baseline.py --track t1 --dry-run          # list pairs, no model
"""

import argparse
from pathlib import Path

from clas import languages as L
from clas.io_utils import load_questions, write_json

INSTRUCTIONS = {
    "none": lambda q, s, t: q,
    "english": lambda q, s, t: f"{q} Respond in {L.get(t).name}.",
    "target_native": lambda q, s, t: f"{q} {L.get(t).instruction}",
    "source_native": lambda q, s, t: f"{q} {L.get(s).instruction}",
    "refusal": lambda q, s, t: f"{q} Refuse answering this question and respond in {L.get(t).name}.",
}


def build_prompts(questions: list, src: str, tgt: str, mode: str) -> list:
    fmt = INSTRUCTIONS[mode]
    return [fmt(q, src, tgt) for q in questions]


def generate(model, prompts: list, params) -> list:
    """vLLM batched chat generation."""
    msgs = [[{"role": "user", "content": p}] for p in prompts]
    return [o.outputs[0].text.strip() for o in model.chat(msgs, sampling_params=params)]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-m", "--model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--track", default="t1", choices=["t1", "t2", "t3", "bonus"])
    p.add_argument("--t3-target", default="ja", choices=L.T3_TARGETS)
    p.add_argument("--instruction", default=None, choices=list(INSTRUCTIONS),
                   help="default: refusal for the bonus track, english otherwise")
    p.add_argument("--questions-dir", type=Path, default=Path("data/validation"))
    p.add_argument("--out", type=Path, default=Path("generations"))
    p.add_argument("-k", "--num-questions", type=int, default=None,
                   help="use only the first k questions per language (default: all)")
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    args.instruction = args.instruction or ("refusal" if args.track == "bonus" else "english")

    pairs = L.pairs(args.track, args.t3_target)
    sources = sorted({s for s, _ in pairs})
    track_dir = args.track if args.track != "t3" else f"t3/{args.t3_target}"
    out_dir = args.out / args.model.split("/")[-1] / track_dir / args.instruction

    print(f"{args.track.upper()}: {len(pairs)} pairs, {len(sources)} source languages")
    if args.dry_run:
        for s, t in pairs:
            print(f"  {s} -> {t}")
        return 0

    import torch
    from vllm import LLM, SamplingParams

    questions = load_questions(args.questions_dir, sources, args.num_questions)
    model = LLM(model=args.model, tensor_parallel_size=max(1, torch.cuda.device_count()),
                max_model_len=args.max_model_len, dtype="float16",
                gpu_memory_utilization=args.gpu_memory_utilization, trust_remote_code=True)
    params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, seed=0)

    for i, (src, tgt) in enumerate(pairs, 1):
        path = out_dir / f"{src}_to_{tgt}.json"
        if path.exists():
            print(f"[{i}/{len(pairs)}] {src}->{tgt}: exists, skipping")
            continue
        prompts = build_prompts(questions[src], src, tgt, args.instruction)
        outputs = generate(model, prompts, params)
        write_json(path, {
            "source_language": src, "target_language": tgt, "model": args.model,
            "method": f"prompting:{args.instruction}", "track": args.track,
            "results": [{"question_idx": j, "input": p, "output": o, "original_prompt": q}
                        for j, (q, p, o) in enumerate(zip(questions[src], prompts, outputs))],
        })
        print(f"[{i}/{len(pairs)}] {src}->{tgt}: {len(outputs)} generations -> {path}")

    print(f"\ndone -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
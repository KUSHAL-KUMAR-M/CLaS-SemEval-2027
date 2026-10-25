import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer


PROJECT_DIR = Path(__file__).resolve().parent.parent
STEERING_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
MANIFEST_PATH = DATA_DIR / "manifest.json"

MODELS = {
    "aya": "CohereLabs/aya-23-8B",
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
}


def process_language(language, info, tokenizer, output_dir, max_tokens=None):
    train_file = PROJECT_DIR / info["train"]["file"]
    output_file = output_dir / f"flores_{language}.pt"

    if not train_file.exists():
        print(f"[SKIP] Missing file: {train_file}")
        return

    if output_file.exists():
        print(f"[SKIP] Already exists: {output_file}")
        return

    with open(train_file, "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f if line.strip()]

    token_ids = []

    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False)

        if max_tokens is not None:
            remaining = max_tokens - len(token_ids)
            if remaining <= 0:
                break
            ids = ids[:remaining]

        token_ids.extend(ids)

        if max_tokens is not None and len(token_ids) >= max_tokens:
            break

    if not token_ids:
        print(f"[SKIP] No tokens found: {language}")
        return

    tensor = torch.tensor(token_ids, dtype=torch.long)
    torch.save(tensor, output_file)

    print(
        f"[SAVED] {output_file} | "
        f"{len(texts)} examples | {tensor.numel():,} tokens"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", choices=["aya", "llama"], default="aya")
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--max_tokens", type=int, default=None)
    args = parser.parse_args()

    model_name = args.model_name or MODELS[args.model_type]
    output_dir = STEERING_DIR / f"data_{args.model_type}"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
    )

    print(f"Tokenizer: {model_name}")
    print(f"Input:     {DATA_DIR / 'train'}")
    print(f"Output:    {output_dir}")

    for language, info in manifest["languages"].items():
        process_language(
            language,
            info,
            tokenizer,
            output_dir,
            args.max_tokens,
        )

    print("FLORES tokenization complete.")


if __name__ == "__main__":
    main()

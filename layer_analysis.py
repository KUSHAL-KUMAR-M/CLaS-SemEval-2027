"""Which layer makes each language most distinct from the other languages?

Uses the mean.<lang>.pt files made by steering/diffmean.py (no model, no GPU needed).

For every layer: deviation(lang) = || mean[lang] - mean over all languages ||

Raw deviations grow with depth because activations get larger, so two normalised
versions are reported:
  rel : deviation / || global mean ||          (relative to the size of the activations)
  z   : deviation / average deviation of all languages at that layer
        (how unusual this language is compared with a typical language at the same layer)

Usage (from the project root):
  python layer_analysis.py --steering_dir steering/data_llama
  python layer_analysis.py --steering_dir steering/data_llama --lang mt --csv layer_analysis_llama.csv
"""
import argparse
import csv
import os

import torch

parser = argparse.ArgumentParser()
parser.add_argument("--steering_dir", default="steering/data_llama")
parser.add_argument("--lang", default=None, help="also print the per-layer curve of this language")
parser.add_argument("--csv", default=None, help="save every language x layer value to this file")
parser.add_argument("--top", type=int, default=3, help="how many best layers to list per language")
args = parser.parse_args()

langs = sorted(f.split(".")[1] for f in os.listdir(args.steering_dir)
               if f.startswith("mean.") and f.endswith(".pt"))
if len(langs) < 3:
    raise SystemExit(f"Need at least 3 mean.<lang>.pt files in {args.steering_dir}, found {langs}")

means = torch.stack([
    torch.load(os.path.join(args.steering_dir, f"mean.{l}.pt"),
               map_location="cpu", weights_only=False)["steering_vectors"].float()
    for l in langs
])                                                       # [languages, layers, hidden]

global_mean = means.mean(dim=0)                          # [layers, hidden]
dev = (means - global_mean).norm(dim=-1)                 # [languages, layers]
rel = dev / global_mean.norm(dim=-1)                     # [languages, layers]
z = dev / dev.mean(dim=0, keepdim=True)                  # [languages, layers]

print(f"{len(langs)} languages, {means.shape[1]} layers\n")
print(f"{'lang':5s} | {'top layers by rel':28s} | {'top layers by z':28s}")
print("-" * 68)
for i, lang in enumerate(langs):
    top_rel = rel[i].topk(args.top).indices.tolist()
    top_z = z[i].topk(args.top).indices.tolist()
    print(f"{lang:5s} | {str(top_rel):28s} | {str(top_z):28s}")

if args.lang:
    i = langs.index(args.lang)
    print(f"\nPer-layer values for '{args.lang}':")
    print("layer     dev      rel      z")
    for layer in range(means.shape[1]):
        print(f"{layer:5d} {dev[i, layer]:8.2f} {rel[i, layer]:8.3f} {z[i, layer]:6.2f}")

if args.csv:
    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["lang", "layer", "deviation", "rel", "z"])
        for i, lang in enumerate(langs):
            for layer in range(means.shape[1]):
                w.writerow([lang, layer, f"{dev[i, layer]:.4f}", f"{rel[i, layer]:.5f}", f"{z[i, layer]:.4f}"])
    print(f"\nSaved {args.csv}")

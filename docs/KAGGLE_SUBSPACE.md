# Adaptive language-subspace experiments on Kaggle

Experimental code; no claim of novelty or score improvement until evaluated.
The original baseline scripts and requirements.txt are unchanged. Do **not** run
`pip install -r requirements.txt` for this experiment. Keep your working Kaggle
PyTorch/vLLM stack. The runner records installed package versions in every run.
Required imports: torch, transformers, accelerate; vllm only for its backend.
The existing evaluator additionally needs its usual fastText/Hugging Face packages.

## Method and controls

For parallel sentences, extract the mean residual activation over the sentence's
assistant tokens under the same neutral `Continue.` context. Each sentence has
one vote. Center across translations of each sentence and fit a rank-r SVD to
language centroids. In balanced data, averaging these centered contrasts is
algebraically equivalent to centering the language means; alignment ensures
matched semantic coverage, not an extra denoising effect by itself. Store an orthonormal basis U, language centroids c, and pooled
within-language diagonal variance at each layer.

The edit is `h + eta * gate * U(c_target - U.T h)`, clipped to at most a fraction
of that token's residual norm. `fixed` sets gate=1; `adaptive` reduces gate as
nearest-centroid target confidence rises. This confidence is an **uncalibrated
proxy**, trained on FLORES sentences, not a validated generative-language probe.
Tune temperature/threshold on tuning questions; test generalization on holdout.
The edit preserves the complement of U at the intervention point, not necessarily
semantics downstream. Stronger or multilayer edits may still harm relevance.

`none` is an unprompted control. `diffmean` uses normalized differences of the
new sentence-weighted centroids: it is a matched-data ablation, **not** an exact
reproduction of the repository's old concatenated-token diffmean experiment.
Use the original script/results as a separate historical reference. Prompted
baselines remain reference-only; no target-language instruction is added here.

## 1. Data and model setup

Run from the repository root in a writable Kaggle directory. Accept model access
on Hugging Face and sign in using Kaggle Secrets/your existing login; do not paste
tokens into committed notebook cells. Use the exact official model ID, separately:
`meta-llama/Llama-3.1-8B-Instruct` or `CohereLabs/tiny-aya-global`.
Artifacts are model-specific and record the resolved revision. Generation loads
that same revision. Default float16 works on T4; choose bfloat16 only on supported
GPUs. HF uses device_map=auto; for vLLM set tensor-parallel-size to your GPU count.
No quantization is silently enabled. CPU offloading can make HF considerably slower.

Prepare `data/train/{code}.txt` containing **aligned translations in identical row
order**, using the existing FLORES download workflow. Raw text is required: the
concatenated `flores_*.pt` files cannot recover sentence alignment. Verify alignment
at the source: equal line counts alone cannot detect reordered translations.
Use extraction sentences disjoint from evaluation questions. Include all 20 Track 2
languages so confidence can compare plausible competing languages.

```bash
python extract_language_subspace.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --parallel-dir data/train \
  --languages ar de en es fr ja zh ru hi ko cs el fa vi uk mt no sw th tr \
  --layers 16 20 24 --rank 8 --num-sentences 512 --batch-size 4 \
  --out /kaggle/working/subspaces/llama.pt
```

Choose valid layers per model; layer indices are zero-based. Start with fewer
sentences or batch-size 1 if needed. Extraction rejects invalid or numerically
rank-deficient artifacts. It never overwrites the existing mean.*.pt files.

## 2. Tests and smoke run (when ready)

```bash
python -m unittest discover -s tests -v
python subspace_generate.py --artifact /kaggle/working/subspaces/llama.pt \
  --pairs en:mt en:sw --layers 20 --limit 2 --dry-run
python subspace_generate.py --artifact /kaggle/working/subspaces/llama.pt \
  --pairs en:mt en:sw --layers 20 --limit 2 --method adaptive --backend hf
```

The dry run validates configuration/data and hashes the artifact without loading
PyTorch or a model; tensor/artifact compatibility is checked on the real run.
The HF `last` scope edits only the final prompt position, then each decode token.
Use a separate `--scope all` experiment to edit all prompt positions too.

Latest vLLM exposes LLM.apply_model, but worker/model internals remain version
sensitive. The optional vLLM backend requires `--scope all`, eager mode, and prefix
caching disabled. It edits **hidden+residual** and returns the delta through hidden.
It fails on unsupported output layouts rather than silently editing the wrong
stream. It intentionally does not guess per-request boundaries in packed tokens.
Compare HF/all against vLLM/all first; HF/last and vLLM/all are different ablations.
If your installed vLLM requires an explicit serialization opt-in for apply_model,
review its warning and enable it only in your trusted notebook; this code does not
automatically alter that security setting. Actual Kaggle GPU compatibility still
needs a smoke test on your installed versions.

```bash
python subspace_generate.py --artifact /kaggle/working/subspaces/llama.pt \
  --backend vllm --scope all --tensor-parallel-size 2 \
  --pairs en:mt en:sw --layers 20 --limit 2 --method adaptive
```

## 3. Screen configurations; do not spend the full GPU budget at once

```bash
python plan_subspace_sweep.py --artifact /kaggle/working/subspaces/llama.pt \
  --layers 16 20 24 --sources en ar ru hi --targets mt sw fa tr \
  --limit 8 --plan /kaggle/working/screen.sh
```

This **writes commands without executing them** (112 model-loading runs by default).
Inspect the plan and start with one target and one layer to reduce cost. Each
command is a separate process so GPU memory is released between runs. Defaults
screen diffmean strengths 2/5/8 and subspace strengths .25/.5/1. Keep norm cap .1
initially, then tune .05/.1/.2 and ranks 2/4/8 only for promising configurations.
The runner supports multiple selected layers, but start with one for attribution.

Select on `--split tune`. Shared row IDs are deterministically hash-split 70/30
across source files. **This assumes translations of the same question share row
positions.** If not, align those files before using the split. The code preserves
line positions and rejects blanks/count mismatches; it cannot infer semantic IDs.
Rerun finalists without --limit on tune, freeze the configuration, then evaluate
once on `--split holdout`. Omit --pairs to generate all 100 Track 2 pairs.
An entirely new run directory is used if settings, data, artifact, or package
versions change. `--resume` skips only completed pairs with matching fingerprints
and question IDs; incomplete pairs rerun. Do not run concurrent writers for the
same configuration. Fingerprints include absolute paths, so moving inputs also
creates a new run. run.json stores the full configuration and selected IDs.

## 4. Evaluate in separate processes

Generation outputs retain the schema expected by the existing evaluate.py.
After generation exits, use the printed run directory:

```bash
python evaluate.py --gen-dir /kaggle/working/subspace_runs/EXACT_RUN_DIRECTORY
python compare_steering_runs.py /path/to/control/scores.json /path/to/adaptive/scores.json
```

Use the same judge, LID threshold, question IDs, and generation length for every
comparison. A `--skip-judge` pass is a compliance smoke test, not an LSS result.
The comparison reports both harmonic-of-macro-components and mean pairwise LSS;
confirm the organizer's final aggregation. It checks pair sets/counts but you must
verify question IDs and evaluator settings. Track parse failures, inspect mixed
language/repetition failures, and evaluate both official models independently.
Do not select from holdout or promise gains from proxy confidence alone.

## Research framing

This combines sentence-aligned language contrasts, bounded subspace transport,
and state-dependent gating. It is an experimental hypothesis, not a demonstrated
new method. Closely related work must be compared, especially:
- https://arxiv.org/abs/2609.16967 (target-language generation and optimal control)
- https://arxiv.org/abs/2605.23036 (multilingual SAEs and layer selection)
- https://aclanthology.org/2026.emnlp-main.1290/ (language/quality steering)

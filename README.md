# CLaS Shared Task at SemEval 2027

This repository contains the code and baselines for the **CLaS Shared Task** on multilingual language steering.

See the official shared task website for more details:

https://clas-semeval2027.github.io/clas.semeval2027/

## Tracks

There are **three main tracks** and one bonus track.

### Track 1: High-Resource Steering

Given inputs in 10 high-resource languages, steer the model to produce outputs in the same set of languages.

* Languages: `ar`, `de`, `en`, `es`, `fr`, `ja`, `zh`, `ru`, `hi`, `ko`
* 90 language pairs (10 × 9, excluding self-pairs)
* Results are averaged across languages
* 2 rankings: one for each model

### Track 2: Low-Resource Steering

Given inputs in the 10 high-resource languages, steer the model to produce outputs in 10 low-resource languages.

* Source languages: the 10 languages from Track 1
* Target languages: `cs`, `el`, `fa`, `vi`, `uk`, `mt`, `no`, `sw`, `th`, `tr`
* 100 language pairs (10 × 10)
* Results are averaged across languages
* 2 rankings: one for each model

### Track 3: Full Multi-Language Steering

Given inputs in all 20 languages from Tracks 1 and 2, steer the model into one selected high-resource target language.

* Target languages: `ja`, `ru`, `de`
* 19 source languages per target
* 6 rankings: one for each model and target language

### Bonus Track: Refusal Steering

Participants steer inputs from the 10 high-resource languages into a refusal mode for each language.

* 10 source languages and 10 target languages
* 100 language pairs (10 × 10, including same-language pairs)
* 2 rankings: one for each model

## Models

The two required models are:

* `meta-llama/Llama-3.1-8B-Instruct`
* `CohereLabs/tiny-aya-global`

Participants can also experiment with additional models.

## Running the Baseline

First install the requirements and log in to Hugging Face:

```bash
pip install -r requirements.txt

HF_TOKEN=*
hf auth login --token $HF_TOKEN
```

Download the training and validation data:

```bash
python download_data.py --out data
```

Run the baseline prompting method and evaluation:

```bash
MODELS_DEFAULT=(
  meta-llama/Llama-3.1-8B-Instruct
  CohereLabs/tiny-aya-global
)

for MODEL in "${MODELS_DEFAULT[@]}"; do
    MODEL_NAME="${MODEL##*/}"

    echo "===== Running $MODEL ====="

    python run_baseline.py --track t1 --model "$MODEL"
    python run_baseline.py --track t2 --model "$MODEL"

    python evaluate.py --gen-dir "generations/$MODEL_NAME/t1/english"
    python evaluate.py --gen-dir "generations/$MODEL_NAME/t2/english"

    for TGT in ja de ru; do
        python run_baseline.py --track t3 --t3-target "$TGT" \
            --instruction english --model "$MODEL"
        python evaluate.py \
            --gen-dir "generations/$MODEL_NAME/t3/$TGT/english"
    done
done
```

The example above runs the **baseline prompting method** and then evaluates the generated outputs.

For the bonus track, the baseline appends a refusal instruction ("Refuse answering this question and respond in X.") and is scored with a separate script, which reports the refusal rate (LLM judge), the language forcing success (LID), and their harmonic mean:

```bash
python run_baseline.py --track bonus --model "$MODEL"
python evaluate_refusal.py --gen-dir "generations/$MODEL_NAME/bonus/refusal"
```

## Data

We provide:

* FLORES-200 devtest subsets for each language as training data for identifying language-related components. Participants are free to use any other data for this stage. 
* CLaS-Bench as the validation dataset.

## Output Format

Each generation run should be saved as a JSON file. For example:

`ar_to_de.json`

```json
{
  "source_language": "ar",
  "target_language": "de",
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "method": "prompting:english",
  "track": "t1",
  "results": [
    {
      "question_idx": 0,
      "input": "blabla",
      "output": "bla",
      "original_prompt": "blibla"
    }
  ]
}
```

The expected number of language pairs is:

* **T1:** 90 pairs (10 × 9, excluding self-pairs)
* **T2:** 100 pairs (10 × 10)
* **T3:** 19 pairs for each selected target language (19 × 1)
* **Bonus:** 100 pairs (10 × 10)

For fair comparison, use deterministic generation:

```text
do_sample=False
max_tokens=1024
```

Use the model's chat template and generation prompt. An example is available in the `generate` function in the codebase.

## Test Set and Submissions

The final evaluation will use **100 previously unseen questions per language** in the evaluation suite. Test questions will not be provided to participants in advance.

The organizers will run submitted systems on their own hardware, using **A100 and H100 GPUs**. Participants will submit their **code and dependency requirements** through a form on the shared task website. Further details about the submission format and requirements will be announced closer to the evaluation period.

During the evaluation period (10–31 January 2027), each team will be allowed **two test submissions per track** and **one final submission per track**. The final submission will be used for the official rankings.

Participants are encouraged to investigate small improvements to existing steering methods and to try new methods that have not previously been applied to language steering.

## Timetable

* Registration deadline: **31 December 2026** ([register here](https://clas-semeval2027.github.io/clas.semeval2027/register/))
* Evaluation starts: **10 January 2027**
* Evaluation ends: **31 January 2027**
* Paper submission: **February 2027**
* Notification: **March 2027**
* Camera-ready: **April 2027**
* SemEval workshop: **Summer 2027**, co-located with ACL

## Reference

This shared task builds on the following paper:

```bibtex
@inproceedings{gurgurov-etal-2026-clas,
    title = "{CL}a{S}-Bench: A Cross-Lingual Alignment and Steering Benchmark",
    author = "Gurgurov, Daniil  and
      Al Ghussin, Yusser  and
      Baeumel, Tanja  and
      Chou, Cheng-Ting  and
      Schramowski, Patrick  and
      Mosbach, Marius  and
      van Genabith, Josef  and
      Ostermann, Simon",
    editor = "Liakata, Maria  and
      Moreira, Viviane P.  and
      Zhang, Jiajun  and
      Jurgens, David",
    booktitle = "Findings of the {A}ssociation for {C}omputational {L}inguistics: {ACL} 2026",
    month = jul,
    year = "2026",
    address = "San Diego, California, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.findings-acl.1086/",
    doi = "10.18653/v1/2026.findings-acl.1086",
    pages = "21591--21628",
    ISBN = "979-8-89176-395-1",
    abstract = "Understanding and controlling the behavior of large language models (LLMs) is an increasingly important topic in multilingual NLP. Beyond prompting or fine-tuning, $\textit{language steering}$, i.e.,{~}manipulating internal representations during inference, has emerged as a more efficient and interpretable technique for adapting models to a target language. Yet, no dedicated benchmarks or evaluation protocols exist to quantify the effectiveness of steering techniques. We introduce $CLaS-Bench$, a lightweight parallel-question benchmark for evaluating language-forcing behavior in LLMs across 32 languages, enabling systematic evaluation of multilingual steering methods. We evaluate a broad array of steering techniques, including residual-stream DiffMean interventions, probe-derived directions, language-specific neurons, PCA/LDA vectors, Sparse Autoencoders, and prompting baselines. Steering performance is measured along two axes: language control and semantic relevance, combined into a single harmonic-mean steering score. We find that across languages simple residual-based DiffMean method consistently outperforms all other methods. Moreover, a layer-wise analysis reveals that language-specific structure emerges predominantly in later layers and steering directions cluster based on language family. $CLaS-Bench$ is the first standardized benchmark for multilingual steering, enabling both rigorous scientific analysis of language representations and practical evaluation of steering as a low-cost adaptation alternative."
}
```

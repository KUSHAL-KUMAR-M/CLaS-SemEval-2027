#!/bin/bash
pip install -r requirements.txt

HF_TOKEN=*
hf auth login --token $HF_TOKEN

MODELS_DEFAULT=(
  meta-llama/Llama-3.1-8B-Instruct
  CohereLabs/tiny-aya-global
)

# first, download the training and validation splits for each language
python download_data.py --out data

# second, run the baseline prompt-based generations for a desired track
for MODEL in "${MODELS_DEFAULT[@]}"; do
    MODEL_NAME="${MODEL##*/}"

    echo "===== Running $MODEL ====="

    python run_baseline.py --track t1 --model "$MODEL"
    python run_baseline.py --track t2 --model "$MODEL"
 
    python evaluate.py --gen-dir "generations/$MODEL_NAME/t1/english"
    python evaluate.py --gen-dir "generations/$MODEL_NAME/t2/english"

    for TGT in ja de ru; do
        python run_baseline.py --track t3 --t3-target "$TGT" --instruction english --model "$MODEL"
        python evaluate.py --gen-dir "generations/$MODEL_NAME/t3/$TGT/english"
    done
done
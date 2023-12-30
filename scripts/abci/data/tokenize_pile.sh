#!/bin/bash
#$ -l rt_F=1
#$ -l h_rt=24:00:00
#$ -j y
#$ -o outputs/dataset/
#$ -cwd

source .env/bin/activate

cd Megatron-LM

DATASET_PATH=/groups/gcd50698/fujii/datasets/pile/merged/merged.jsonl
OUTPUT_DIR=/groups/gcd50698/fujii/datasets/pile/bin-test
mkdir -p $OUTPUT_DIR

python tools/preprocess_data.py \
  --input $DATASET_PATH \
  --output-prefix $OUTPUT_DIR/pile-mamba-train \
  --tokenizer-type MambaTokenizer \
  --workers 32 \
  --append-eod

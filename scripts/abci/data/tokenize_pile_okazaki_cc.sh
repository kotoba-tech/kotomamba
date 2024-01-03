#!/bin/bash
#$ -l rt_F=1
#$ -l h_rt=24:00:00
#$ -j y
#$ -o outputs/dataset/
#$ -cwd

source .env/bin/activate

cd megatron_lm

DATASET_PATH=/groups/gaf51275/llama/datasets/mamba_ja_en/merged.jsonl
OUTPUT_DIR=/groups/gaf51275/llama/datasets/mamba_ja_en

mkdir -p $OUTPUT_DIR

python tools/preprocess_data.py \
  --input $DATASET_PATH \
  --output-prefix $OUTPUT_DIR/mamba-en-ja \
  --tokenizer-type SentencePieceTokenizer \
  --tokenizer-model /bb/llm/gaf51275/llm-jp/llm-ja-tokenizer/models/ver2/code10K_en20K_ja30K.ver2.2.model \
  --workers 64 \
  --append-eod

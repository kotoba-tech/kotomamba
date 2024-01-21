#!/bin/bash
#$ -l rt_F=1
#$ -l h_rt=24:00:00
#$ -j y
#$ -o outputs/dataset/
#$ -cwd

source .env/bin/activate

DATASET_PATH=/bb/llm/gaf51275/llama/datasets/merge_datasets/ja_wiki/ja_wiki_merged.jsonl
OUTPUT_DIR=/bb/grandchallenge/gaf51389/datasets/abci-grand-challenge-continual
mkdir -p $OUTPUT_DIR

python megatron_lm/tools/preprocess_data.py \
  --input $DATASET_PATH \
  --output-prefix $OUTPUT_DIR/ja_wikipedia \
  --tokenizer-type HuggingFaceTokenizer \
  --tokenizer-model EleutherAI/gpt-neox-20b \
  --workers 64 \
  --append-eod

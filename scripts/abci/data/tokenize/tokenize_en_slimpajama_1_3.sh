#!/bin/bash
#$ -l rt_F=1
#$ -l h_rt=24:00:00
#$ -j y
#$ -o outputs/dataset/
#$ -cwd

source .env/bin/activate

DATASET_DIR=/bb/grandchallenge/gaf51389/datasets/SlimPajama-627B/train
OUTPUT_DIR=/bb/grandchallenge/gaf51389/datasets/abci-grand-challenge-continual
mkdir -p $OUTPUT_DIR

DATASET_PATH=$DATASET_DIR/slimpajama-627b-1.jsonl

python megatron_lm/tools/preprocess_data.py \
  --input $DATASET_PATH \
  --output-prefix $OUTPUT_DIR/slimpajama_1 \
  --tokenizer-type HuggingFaceTokenizer \
  --tokenizer-model EleutherAI/gpt-neox-20b \
  --workers 64 \
  --append-eod

DATASET_PATH=$DATASET_DIR/slimpajama-627b-2.jsonl

python megatron_lm/tools/preprocess_data.py \
  --input $DATASET_PATH \
  --output-prefix $OUTPUT_DIR/slimpajama_2 \
  --tokenizer-type HuggingFaceTokenizer \
  --tokenizer-model EleutherAI/gpt-neox-20b \
  --workers 64 \
  --append-eod

DATASET_PATH=$DATASET_DIR/slimpajama-627b-3.jsonl

python megatron_lm/tools/preprocess_data.py \
  --input $DATASET_PATH \
  --output-prefix $OUTPUT_DIR/slimpajama_3 \
  --tokenizer-type HuggingFaceTokenizer \
  --tokenizer-model EleutherAI/gpt-neox-20b \
  --workers 64 \
  --append-eod

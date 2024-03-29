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

DATASET_PATH=$DATASET_DIR/slimpajama-627b-7.jsonl

python megatron_lm/tools/preprocess_data.py \
  --input $DATASET_PATH \
  --output-prefix $OUTPUT_DIR/slimpajama_7 \
  --tokenizer-type HuggingFaceTokenizer \
  --tokenizer-model EleutherAI/gpt-neox-20b \
  --workers 64 \
  --append-eod

DATASET_PATH=$DATASET_DIR/slimpajama-627b-8.jsonl

python megatron_lm/tools/preprocess_data.py \
  --input $DATASET_PATH \
  --output-prefix $OUTPUT_DIR/slimpajama_8 \
  --tokenizer-type HuggingFaceTokenizer \
  --tokenizer-model EleutherAI/gpt-neox-20b \
  --workers 64 \
  --append-eod

DATASET_PATH=$DATASET_DIR/slimpajama-627b-9.jsonl

python megatron_lm/tools/preprocess_data.py \
  --input $DATASET_PATH \
  --output-prefix $OUTPUT_DIR/slimpajama_9 \
  --tokenizer-type HuggingFaceTokenizer \
  --tokenizer-model EleutherAI/gpt-neox-20b \
  --workers 64 \
  --append-eod

DATASET_PATH=$DATASET_DIR/slimpajama-627b-10.jsonl

python megatron_lm/tools/preprocess_data.py \
  --input $DATASET_PATH \
  --output-prefix $OUTPUT_DIR/slimpajama_10 \
  --tokenizer-type HuggingFaceTokenizer \
  --tokenizer-model EleutherAI/gpt-neox-20b \
  --workers 64 \
  --append-eod

#!/bin/bash
#$ -l rt_F=1
#$ -l h_rt=24:00:00
#$ -j y
#$ -o outputs/index
#$ -cwd

source .env/bin/activate

python tools/indexing/index_dataset.py \
  --data-file-path /groups/gcd50698/fujii/datasets/pile/merged/merged.jsonl

python tools/indexing/index_dataset.py \
  --data-file-path /groups/gcd50698/fujii/datasets/pile/merged/merged_valid.jsonl

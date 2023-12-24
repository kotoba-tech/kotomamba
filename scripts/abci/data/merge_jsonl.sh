#!/bin/bash
#$ -l rt_F=1
#$ -l h_rt=24:00:00
#$ -j y
#$ -o outputs/dataset/
#$ -cwd

NUMBER=0

output_file="/groups/gcd50698/fujii/datasets/pile/merged/merged_$NUMBER.jsonl"

if [ -f "$output_file" ]; then
  rm "$output_file"
fi

INPUT_DIR=/groups/gcd50698/fujii/datasets/pile/en_pile
TARGET_FILES=$(find "$INPUT_DIR" -name "train_*$NUMBER.jsonl" | sort)

for file in $TARGET_FILES; do
  echo "Merging $file into $output_file"
  cat "$file" >>"$output_file"
done

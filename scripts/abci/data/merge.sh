#!/bin/bash
#$ -l rt_F=1
#$ -l h_rt=24:00:00
#$ -j y
#$ -o outputs/dataset/
#$ -cwd

output_file="/groups/gcd50698/fujii/datasets/mamba_ja_en/merged.jsonl"

if [ -f "$output_file" ]; then
  rm "$output_file"
fi


TARGET_FILES=""
TARGET_FILES="$TARGET_FILES /groups/gcd50698/fujii/datasets/pile/merged/merged.jsonl"
TARGET_FILES="$TARGET_FILES /groups/gcd50698/fujii/datasets/okazaki_lab_cc_02_sampled_405/merged/merged.jsonl"


for file in $TARGET_FILES; do
  echo "Merging $file into $output_file"
  cat "$file" >>"$output_file"
done

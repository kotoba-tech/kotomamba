#!/bin/bash
#$ -l rt_F=1
#$ -l h_rt=24:00:00
#$ -j y
#$ -o outputs/dataset/
#$ -cwd

OUTPUT_DIR=/groups/gcd50698/fujii/datasets/okazaki_lab_cc_02_sampled_405/merged
mkdir -p $OUTPUT_DIR

output_file="$OUTPUT_DIR/merged.jsonl"

if [ -f "$output_file" ]; then
  rm "$output_file"
fi

INPUT_DIR=/bb/llm/gaf51275/jalm/okazaki_lab_cc_02_sampled_405/
TARGET_FILES=$(find "$INPUT_DIR" -name "*.jsonl" | sort)

for file in $TARGET_FILES; do
  echo "Merging $file into $output_file"
  cat "$file" >>"$output_file"
done

#!/bin/bash
#$ -l rt_F=1
#$ -l h_rt=24:00:00
#$ -j y
#$ -o outputs/dataset/
#$ -cwd

CHUNK_INDEX=1

OUTPUT_FILE="/bb/grandchallenge/gaf51389/datasets/SlimPajama-627B/train/slimpajama-627b-${CHUNK_INDEX}.jsonl"

if [ -f "$OUTPUT_FILE" ]; then
  rm "$OUTPUT_FILE"
fi

TARGET_DIR="/bb/grandchallenge/gaf51389/datasets/SlimPajama-627B/train"

echo "Merging files in $TARGET_DIR to $OUTPUT_FILE"

for file in "$TARGET_DIR/chunk${CHUNK_INDEX}"/*.jsonl.zst; do
  zstd -d --stdout "$file" >> "$OUTPUT_FILE"
done

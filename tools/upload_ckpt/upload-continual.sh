#!/bin/bash

set -e

start=40000
end=50000
increment=10000

upload_base_dir=/bb/grandchallenge/gaf51389/converted_hf_checkpoints/mamba-2.8b/v-node

# for ループで指定された範囲と増分を使用
for ((i = start; i <= end; i += increment)); do
  upload_dir=$upload_base_dir/iter_$(printf "%07d" $i)

  python tools/upload_ckpt/upload.py \
    --ckpt-path $upload_dir \
    --repo-name kotoba-tech/kotomamba-2.8b-continual-ja180B-en20B-iter$(printf "%07d" $i)
done

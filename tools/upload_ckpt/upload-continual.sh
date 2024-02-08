#!/bin/bash

set -e

start=10000
end=15000
increment=5000

upload_base_dir=/bb/grandchallenge/gaf51389/converted_hf_checkpoints/mamba-2.8b/v-node_LR_6e-5_MINLR_6.6e-6_FP32

# for ループで指定された範囲と増分を使用
for ((i = start; i <= end; i += increment)); do
  upload_dir=$upload_base_dir/iter_$(printf "%07d" $i)

  python tools/upload_ckpt/upload.py \
    --ckpt-path $upload_dir \
    --repo-name kotoba-tech/kotomamba-2.8b-continual-ja180B-en20B-LR_6e-5_MINLR_6.6e-6_FP32-iter$(printf "%07d" $i)
done

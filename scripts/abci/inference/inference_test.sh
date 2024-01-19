#!/bin/bash
#$ -l rt_AG.small=1
#$ -l h_rt=00:30:00
#$ -j y
#$ -o outputs/inference/
#$ -cwd

# module load
source /etc/profile.d/modules.sh
module load cuda/11.8/11.8.0
module load cudnn/8.9/8.9.2
module load nccl/2.16/2.16.2-1
module load hpcx/2.12

# swich virtual env
source .env/bin/activate

# huggingface cache
export HF_HOME=/groups/gcd50698/fujii/work/mamba/mamba/.hf_cache

# mambda
python benchmarks/benchmark_generation_mamba_simple.py \
  --model-name "/bb/grandchallenge/gaf51389/converted_hf_checkpoints/mamba-2.8b/a-node/iter_0000500" \
  --tokenizer-path "/bb/llm/gaf51275/llm-jp/llm-ja-tokenizer/models/ver2/code20K_en40K_ja60K.ver2.2.model" \
  --prompt "東京工業大学のキャンパスは、" \
  --topp 0.9 --temperature 0.7 --repetition-penalty 1.2

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
  --model-name "state-spaces/mamba-2.8b" --batch 128

# pythia 2.8b
python benchmarks/benchmark_generation_mamba_simple.py \
  --model-name "EleutherAI/pythia-2.8b" --batch 128

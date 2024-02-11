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

CHECHPOINT_DIR=""

python benchmarks/benchmark_generation_mamba_simple.py \
  --model-name "${CHECHPOINT_DIR}/kotoba-tech/kotomamba-2.8B" \
  --tokenizer-path "${CHECHPOINT_DIR}/kotoba-tech/kotomamba-2.8B" \
  --tokenizer-model "${CHECHPOINT_DIR}/kotoba-tech/kotomamba-2.8B/tokenizer.model" \
  --tokenizer-type SentencePieceTokenizer \
  --use-sentencepiece \
  --prompt "東京工業大学は" \
  --topp 0.9 --temperature 0.7 --repetition-penalty 1.2

python benchmarks/benchmark_generation_mamba_simple.py \
  --model-name "${CHECHPOINT_DIR}/kotoba-tech/kotomamba-2.8B" \
  --tokenizer-path "${CHECHPOINT_DIR}/kotoba-tech/kotomamba-2.8B" \
  --tokenizer-model "${CHECHPOINT_DIR}/kotoba-tech/kotomamba-2.8B/tokenizer.model" \
  --tokenizer-type SentencePieceTokenizer \
  --use-sentencepiece \
  --prompt "東北大学は" \
  --topp 0.9 --temperature 0.7 --repetition-penalty 1.2

python benchmarks/benchmark_generation_mamba_simple.py \
  --model-name "${CHECHPOINT_DIR}/kotoba-tech/kotomamba-2.8B-CL" \
  --tokenizer-path "EleutherAI/gpt-neox-20b" \
  --prompt "東京工業大学は" \
  --topp 0.9 --temperature 0.7 --repetition-penalty 1.2


python benchmarks/benchmark_generation_mamba_simple.py \
  --model-name "${CHECHPOINT_DIR}/kotoba-tech/kotomamba-2.8B-CL" \
  --tokenizer-path "EleutherAI/gpt-neox-20b" \
  --prompt "東北大学は" \
  --topp 0.9 --temperature 0.7 --repetition-penalty 1.2

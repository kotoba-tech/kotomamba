#!/bin/bash
#$ -l rt_F=1
#$ -l h_rt=1:00:00
#$ -j y
#$ -o outputs/convert/ckpt/
#$ -cwd
# module load
source /etc/profile.d/modules.sh
module load cuda/11.8/11.8.0
module load cudnn/8.9/8.9.2
module load nccl/2.16/2.16.2-1
module load hpcx/2.12

set -e

# swich virtual env
source .env/bin/activate

# convert checkpoints
start=10000
end=15000
increment=5000

export HF_HOME=/bb/grandchallenge/gaf51389/hf_cache

for ((i = start; i <= end; i += increment)); do
  ITERATION=$i
  FORMATTED_ITERATION=$(printf "iter_%07d" $ITERATION)

  CHECK_POINT_PATH=/bb/llm/gaf51275/llama/checkpoints/mamba-2.8b-slimpj/v-node/BS_1024_LR_6e-5_MINLR_6e-6_WARMUP_2000_WD_0.1_GC_1_SEQ_2048_FP32/${FORMATTED_ITERATION}/model.pt
  OUTPUT_PATH=/bb/grandchallenge/gaf51389/converted_hf_checkpoints/mamba-2.8b/v-node_LR_6e-5_MINLR_6.6e-6_FP32/${FORMATTED_ITERATION}
  TOKNENIZER_PATH=EleutherAI/gpt-neox-20b

  echo "convert ${CHECK_POINT_PATH} to ${OUTPUT_PATH}"

  mkdir -p $OUTPUT_PATH

  BASE_MODEL_CHECKPOINT=/bb/grandchallenge/gaf51389/hf_checkpoints/mamba-2.8b-slimpj

  python tools/convert_ckpt/convert_mamba.py \
    --model $BASE_MODEL_CHECKPOINT \
    --ckpt $CHECK_POINT_PATH \
    --out $OUTPUT_PATH \
    --tokenizer-path $TOKNENIZER_PATH
done

#!/bin/bash
#$ -l rt_AF=1
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
start=1300
end=1300
increment=200

for ((i = start; i <= end; i += increment)); do
  ITERATION=$i
  FORMATTED_ITERATION=$(printf "iter_%07d" $ITERATION)

  CHECK_POINT_PATH=/groups/gcd50698/fujii/work/mamba/checkpoints/mamba-130m-pile-hpyer-parameter/${FORMATTED_ITERATION}/model.pt
  OUTPUT_PATH=/groups/gcd50698/fujii/work/mamba/checkpoints/mamba-130m-hf

  echo "convert ${CHECK_POINT_PATH} to ${OUTPUT_PATH}"

  mkdir -p $OUTPUT_PATH

  BASE_MODEL_CHECKPOINT=/groups/gcd50698/fujii/work/mamba/hf_checkpoints/mamba-130m

  python tools/convert_ckpt/convert_mamba.py \
    --model $BASE_MODEL_CHECKPOINT \
    --ckpt $CHECK_POINT_PATH \
    --out $OUTPUT_PATH
done

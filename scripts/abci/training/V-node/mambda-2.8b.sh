#!/bin/bash
#$ -l rt_F=64
#$ -l h_rt=00:30:00
#$ -j y
#$ -o outputs/v-node/mamba-2.8b/
#$ -cwd

# module load
source /etc/profile.d/modules.sh
module load cuda/11.8/11.8.0
module load cudnn/8.9/8.9.2
module load nccl/2.16/2.16.2-1
module load hpcx/2.12

# swich virtual env
source .env/bin/activate

# distributed settings
export MASTER_ADDR=$(/usr/sbin/ip a show dev bond0 | grep 'inet ' | awk '{ print $2 }' | cut -d "/" -f 1)
export MASTER_PORT=$((10000 + ($JOB_ID % 50000)))

echo "MASTER_ADDR=${MASTER_ADDR}"

# hostfile

if [[ "$SGE_RESOURCE_TYPE" == "rt_F" ]]; then
  export NUM_GPU_PER_NODE=4
  NODE_TYPE="v100"
elif [[ "$SGE_RESOURCE_TYPE" == "rt_AF" ]]; then
  export NUM_GPU_PER_NODE=8
  NODE_TYPE="a100"
else
  echo "Unrecognized SGE_RESOURCE_TYPE: $SGE_RESOURCE_TYPE"
fi

NUM_NODES=$NHOSTS
NUM_GPUS=$((${NUM_NODES} * ${NUM_GPU_PER_NODE}))

mkdir -p ./hostfile

HOSTFILE_NAME=./hostfile/hostfile_${JOB_ID}
while read -r line; do
  echo "${line} slots=${NUM_GPU_PER_NODE}"
done <"$SGE_JOB_HOSTLIST" >"$HOSTFILE_NAME"

# training settings
NUM_EPOCHS=1

# batch size
BATCH_SIZE=2
GLOBAL_BATCH_SIZE=512
GRADIENT_ACCUMULATION_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_SIZE * NUM_GPUS)))

if (($GRADIENT_ACCUMULATION_STEPS < 1)); then
  echo "Error: Gradient Accumulation Steps is less than 1. Exiting."
  exit 1
fi

# optimizer
LR=8e-4
LR_MIN=1e-5
LR_DECAY=0.80
LR_WARMUP=0.05
LR_DECAY_STYLE="cosine"
WEIGHT_DECAY=0.1

# seed
SEED=42

# dataset
NUM_WORKERS_DATALOADER=2
DATASET_DIR=/bb/grandchallenge/gaf51389/datasets/mamba_ja_en

# checkpoint path
CHECKPOINTS_PATH=/bb/grandchallenge/gaf51389/checkpoints/mamba-2.8b/v-node/pike-okazaki-cc
mkdir -p $CHECKPOINTS_PATH

# model dir
MODEL_DIR=/bb/grandchallenge/gaf51389/hf_checkpoints/mamba-2.8b

# huggingface cache
export HF_HOME=/bb/grandchallenge/gaf51389/hf_cache

# ldconfig
alias ldconfig=/usr/sbin/ldconfig

# run
mpirun -np $NUM_GPUS \
  --npernode $NUM_GPU_PER_NODE \
  -hostfile $HOSTFILE_NAME \
  -x MASTER_ADDR=$MASTER_ADDR \
  -x MASTER_PORT=$MASTER_PORT \
  -bind-to none -map-by slot \
  -x PATH \
  python pretrain.py \
  --enable_fsdp \
  --low_cpu_fsdp \
  --mixed_precision \
  --use_fp16 \
  --num_epochs $NUM_EPOCHS \
  --model_name $MODEL_DIR \
  --tokenizer_name /bb/llm/gaf51275/llm-jp/llm-ja-tokenizer/models/ver2/code10K_en20K_ja30K.ver2.2.model \
  --batch_size $BATCH_SIZE \
  --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
  --fsdp_activation_checkpointing \
  --lr $LR \
  --lr_min $LR_MIN \
  --lr_warmup $LR_WARMUP \
  --lr_decay $LR_DECAY \
  --lr_decay_style $LR_DECAY_STYLE \
  --weight_decay $WEIGHT_DECAY \
  --seed $SEED \
  --dataset "pile_dataset" \
  --train_data_path $DATASET_DIR/mamba-en-ja_text_document.bin \
  --num_workers_dataloader $NUM_WORKERS_DATALOADER \
  --save_model \
  --save_optimizer \
  --save_interval_iteration 500 \
  --context-size 2048 \
  --save_checkpoint_path $CHECKPOINTS_PATH \
  --load_checkpoint_path $CHECKPOINTS_PATH \
  --from_scratch \
  --use_mpi \
  --wandb-entity "fine-tuning-llm" \
  --wandb-project "mamba" \
  --wandb_name "2.8b-${NODE_TYPE}-${NUM_NODES}nodes-pile-okazaki-cc"

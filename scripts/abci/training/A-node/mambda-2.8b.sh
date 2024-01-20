#!/bin/bash
#$ -l rt_AF=2
#$ -l h_rt=10:00:00:00
#$ -j y
#$ -o outputs/a-node/mamba-2.8b/
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

# training config
SEQ_LENGTH=2048
DATA_PARALLEL_SIZE=$NUM_GPUS

MICRO_BATCH_SIZE=8
GLOBAL_BATCH_SIZE=1024
TRAIN_STEPS=191000

# optimizer config
LR=8e-4
MIN_LR=1e-5
LR_WARMUP_STEPS=2000
LR_DECAY_STEPS=$TRAIN_STEPS
WEIGHT_DECAY=0.1
GRAD_CLIP=1

# checkpoint & tokenizer
TOKENIZER_MODEL=/bb/llm/gaf51275/llm-jp/llm-ja-tokenizer/models/ver2/code20K_en40K_ja60K.ver2.2.model
CHECKPOINT_DIR=/bb/grandchallenge/gaf51389/hf_checkpoints/mamba-2.8b/
CHECKPOINT_SAVE_DIR=/bb/grandchallenge/gaf51389/checkpoints/mamba-2.8b/a-node/BS_${GLOBAL_BATCH_SIZE}_LR_${LR}_MINLR_${MIN_LR}_WARMUP_${LR_WARMUP_STEPS}_WD_${WEIGHT_DECAY}_GC_${GRAD_CLIP}

mkdir -p ${CHECKPOINT_SAVE_DIR}

# data config

DATASET_DIR=/bb/grandchallenge/gaf51389/datasets/abci-grand-challenge
DATA_PATH=""

# ja okazaki lab cc
DATA_PATH="${DATA_PATH} 1542288829 ${DATASET_DIR}/ja_wikipedia_text_document"

# job name
JOB_NAME="Mamba-2.8B-${NODE_TYPE}-${NUM_NODES}node-${NUM_GPUS}gpu-${SEQ_LENGTH}s-BS=${GLOBAL_BATCH_SIZE}-LR=${LR}-MINLR=${MIN_LR}-WARMUP=${LR_WARMUP_STEPS}-WD=${WEIGHT_DECAY}-GC=${GRAD_CLIP}"

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
  --seq-length ${SEQ_LENGTH} \
  --sliding-window-size ${SEQ_LENGTH} \
  --micro-batch-size ${MICRO_BATCH_SIZE} \
  --global-batch-size ${GLOBAL_BATCH_SIZE} \
  --train-iters ${TRAIN_STEPS} \
  --tokenizer-type SentencePieceTokenizer \
  --tokenizer-model ${TOKENIZER_MODEL} \
  --data-path ${DATA_PATH} \
  --split 949,50,1 \
  --lr ${LR} \
  --min-lr ${MIN_LR} \
  --lr-decay-style cosine \
  --lr-warmup-iters ${LR_WARMUP_STEPS} \
  --lr-decay-iters ${LR_DECAY_STEPS} \
  --weight-decay ${WEIGHT_DECAY} \
  --grad-clip-norm ${GRAD_CLIP} \
  --optimizer adam \
  --adam-beta1 0.9 \
  --adam-beta2 0.95 \
  --adam-eps 1e-6 \
  --save-interval 500 \
  --eval-interval 100 \
  --eval-iters 10 \
  --bf16 \
  --mixed-precision \
  --base-model ${CHECKPOINT_DIR} \
  --save ${CHECKPOINT_SAVE_DIR} \
  --load ${CHECKPOINT_SAVE_DIR} \
  --low-cpu-fsdp \
  --sharding-strategy FULL_SHARD \
  --checkpoint-type LOCAL_STATE_DICT \
  --fsdp-activation-checkpointing \
  --use-mpi \
  --from-scratch \
  --mamba \
  --wandb-entity "prj-jalm" \
  --wandb-project "ABCI-mamba" \
  --wandb-name "${JOB_NAME}"

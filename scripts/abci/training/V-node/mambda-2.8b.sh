#!/bin/bash
#$ -l rt_F=128
#$ -l h_rt=3:00:00
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

# training config
SEQ_LENGTH=2048
DATA_PARALLEL_SIZE=$NUM_GPUS

MICRO_BATCH_SIZE=2
GLOBAL_BATCH_SIZE=1024
TRAIN_STEPS=191000

# optimizer config
LR=1e-4
MIN_LR=3.3e-6
LR_WARMUP_STEPS=2000
LR_DECAY_STEPS=$TRAIN_STEPS
WEIGHT_DECAY=0.1
GRAD_CLIP=1

# checkpoint & tokenizer
TOKENIZER_MODEL=EleutherAI/gpt-neox-20b
CHECKPOINT_DIR=/bb/grandchallenge/gaf51389/hf_checkpoints/mamba-2.8b-slimpj
CHECKPOINT_SAVE_DIR=/bb/grandchallenge/gaf51389/checkpoints/mamba-2.8b-slimpj/v-node/BS_${GLOBAL_BATCH_SIZE}_LR_${LR}_MINLR_${MIN_LR}_WARMUP_${LR_WARMUP_STEPS}_WD_${WEIGHT_DECAY}_GC_${GRAD_CLIP}_SEQ_${SEQ_LENGTH}

mkdir -p ${CHECKPOINT_SAVE_DIR}

# data config

DATASET_DIR=/bb/grandchallenge/gaf51389/datasets/abci-grand-challenge-continual
DATA_PATH=""

# ja wikipedia
DATA_PATH="${DATA_PATH} 2445161607 ${DATASET_DIR}/ja_wikipedia_text_document"

# en slimpajama
DATA_PATH="${DATA_PATH} 1998733749 ${DATASET_DIR}/slimpajama_1_text_document"
DATA_PATH="${DATA_PATH} 1994431450 ${DATASET_DIR}/slimpajama_2_text_document"
DATA_PATH="${DATA_PATH} 1989453576 ${DATASET_DIR}/slimpajama_3_text_document"
DATA_PATH="${DATA_PATH} 2000446584 ${DATASET_DIR}/slimpajama_4_text_document"
DATA_PATH="${DATA_PATH} 2009165481 ${DATASET_DIR}/slimpajama_5_text_document"
DATA_PATH="${DATA_PATH} 1993229817 ${DATASET_DIR}/slimpajama_6_text_document"
DATA_PATH="${DATA_PATH} 1995173042 ${DATASET_DIR}/slimpajama_7_text_document"
DATA_PATH="${DATA_PATH} 2013004838 ${DATASET_DIR}/slimpajama_8_text_document"
DATA_PATH="${DATA_PATH} 1999418219 ${DATASET_DIR}/slimpajama_9_text_document"
DATA_PATH="${DATA_PATH} 2006943244 ${DATASET_DIR}/slimpajama_10_text_document"


DATASET_DIR="/bb/grandchallenge/gaf51389/datasets/abci-grand-challenge/okazaki_lab_cc_gpt-neox-20b"

# ja okazaki lab cc
DATA_PATH="${DATA_PATH} 8149250311 ${DATASET_DIR}/merged_0_text_document"
DATA_PATH="${DATA_PATH} 8184079697 ${DATASET_DIR}/merged_1_text_document"
DATA_PATH="${DATA_PATH} 8143099062 ${DATASET_DIR}/merged_2_text_document"
DATA_PATH="${DATA_PATH} 8459141078 ${DATASET_DIR}/merged_3_text_document"
DATA_PATH="${DATA_PATH} 7888082822 ${DATASET_DIR}/merged_4_text_document"
DATA_PATH="${DATA_PATH} 7898693365 ${DATASET_DIR}/merged_5_text_document"
DATA_PATH="${DATA_PATH} 7518110853 ${DATASET_DIR}/merged_6_text_document"
DATA_PATH="${DATA_PATH} 9901844874 ${DATASET_DIR}/merged_7_text_document"
DATA_PATH="${DATA_PATH} 9210340014 ${DATASET_DIR}/merged_8_text_document"
DATA_PATH="${DATA_PATH} 7951942872 ${DATASET_DIR}/merged_9_text_document"
DATA_PATH="${DATA_PATH} 8663600423 ${DATASET_DIR}/merged_10_text_document"
DATA_PATH="${DATA_PATH} 9679675407 ${DATASET_DIR}/merged_11_text_document"
DATA_PATH="${DATA_PATH} 9251081241 ${DATASET_DIR}/merged_12_text_document"
DATA_PATH="${DATA_PATH} 8037577256 ${DATASET_DIR}/merged_13_text_document"
DATA_PATH="${DATA_PATH} 9403629150 ${DATASET_DIR}/merged_14_text_document"
DATA_PATH="${DATA_PATH} 9183699689 ${DATASET_DIR}/merged_15_text_document"
DATA_PATH="${DATA_PATH} 8284421403 ${DATASET_DIR}/merged_16_text_document"
DATA_PATH="${DATA_PATH} 7661984764 ${DATASET_DIR}/merged_17_text_document"
DATA_PATH="${DATA_PATH} 8568060884 ${DATASET_DIR}/merged_18_text_document"
DATA_PATH="${DATA_PATH} 7977151963 ${DATASET_DIR}/merged_19_text_document"
DATA_PATH="${DATA_PATH} 7539371265 ${DATASET_DIR}/merged_20_text_document"

# job name
JOB_NAME="Mamba-2.8B-${NODE_TYPE}-${NUM_NODES}node-${NUM_GPUS}gpu-${SEQ_LENGTH}s-BS=${GLOBAL_BATCH_SIZE}-LR=${LR}-MINLR=${MIN_LR}-WARMUP=${LR_WARMUP_STEPS}-WD=${WEIGHT_DECAY}-GC=${GRAD_CLIP}"

# huggingface cache
export HF_HOME=/bb/grandchallenge/gaf51389/hf_cache
export TRITON_CACHE_DIR=/bb/grandchallenge/gaf51389/triton_cache
mkdir -p $TRITON_CACHE_DIR

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
  --micro-batch-size ${MICRO_BATCH_SIZE} \
  --global-batch-size ${GLOBAL_BATCH_SIZE} \
  --train-iters ${TRAIN_STEPS} \
  --tokenizer-type HuggingFaceTokenizer \
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
  --fp16 \
  --mixed-precision \
  --base-model ${CHECKPOINT_DIR} \
  --save ${CHECKPOINT_SAVE_DIR} \
  --load ${CHECKPOINT_SAVE_DIR} \
  --low-cpu-fsdp \
  --sharding-strategy FULL_SHARD \
  --checkpoint-type LOCAL_STATE_DICT \
  --fsdp-activation-checkpointing \
  --use-mpi \
  --mamba \
  --wandb-entity "prj-jalm" \
  --wandb-project "ABCI-mamba" \
  --wandb-name "${JOB_NAME}"

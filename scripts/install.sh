#!/bin/bash

module load cuda/11.8/11.8.0
module load cudnn/8.9/8.9.2
module load nccl/2.16/2.16.2-1
module load hpcx/2.12

# install

pip install -r requirements.txt

# mambda
pip install causal-conv1d>=1.1.0
pip install mamba-ssm

# multi-node
pip install mpi4py

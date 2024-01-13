#!/bin/bash
#$ -l rt_F=1
#$ -l h_rt=4:00:00:00
#$ -j y
#$ -o outputs/datasets/
#$ -cwd

cd /bb/grandchallenge/gaf51389/datasets

git clone git@hf.co:datasets/cerebras/SlimPajama-627B

#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=16:00:00
#$ -N gpu-job
#$ -j y -o gpu-job.out
module purge
module load cuda
source ~/.bashrc
conda activate diffpure
wandb agent maorong-wang/robustbench-ranpac-hira-rc-sweep/ioz9pqhp

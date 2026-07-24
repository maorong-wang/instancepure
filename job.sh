#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=24:00:00
#$ -N gpu-job
#$ -j y -o gpu-job3.out
module purge
module load cuda
source ~/.bashrc
conda activate robustbench

wandb agent maorong-wang/robustbench-ranpac-hira-apgdcw/do673tgq

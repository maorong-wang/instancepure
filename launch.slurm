#!/bin/bash
#SBATCH -p gpu
#SBATCH -J dfp 
#SBATCH -o dfp_%j.out
#SBATCH --nodes=1
#SBATCH --nodelist=elab-gpu-ubuntu04
#SBATCH --ntasks=8
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G
#SBATCH --time=48:00:00

source /home_fmg/maorong/miniconda3/etc/profile.d/conda.sh
conda activate diffpure 

SWEEP_ID="maorong-wang/robustbench-ranpac/zvq9nfch"

echo "Starting 8 W&B agents with srun"
echo "Job ID: $SLURM_JOB_ID"

srun --gpu-bind=single:1 \
     wandb agent $SWEEP_ID

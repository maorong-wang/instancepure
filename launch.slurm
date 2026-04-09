#!/bin/bash
#SBATCH -p long
#SBATCH -J wb_agent
#SBATCH -o wb_agent_%A_%a.out
#SBATCH --nodes=1
#SBATCH --nodelist=elab-gpu-ubuntu06
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=25G
#SBATCH --time=72:00:00


source /home_fmg/maorong/miniconda3/etc/profile.d/conda.sh
conda activate diffpure 

SWEEP_ID="maorong-wang/robustbench-ranpac-hira-rc-rp_normalization/b7dn8hu0"

echo "Job: $SLURM_JOB_ID  Array task: $SLURM_ARRAY_TASK_ID"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

wandb agent --forward-signals "$SWEEP_ID"

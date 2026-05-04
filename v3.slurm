#!/bin/bash
#SBATCH -p gpu
#SBATCH -J visualization
#SBATCH -o visualization_%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=25G
#SBATCH --time=24:00:00


source /home_fmg/maorong/miniconda3/etc/profile.d/conda.sh
conda activate diffpure 


python visualization/pgd_iteration_curves.py \
    --model-name Xu2024MIMIR_Swin-B \
    --data-dir ./dataset/imagenet \
    --model-dir ./robustbench_models \
    --device cuda:0 \
    --eps 4/255 \
    --max-steps 100 \
    --pgd-random-start true \
    --mask-pgd-logits false \
    --num-classes 1000 \
    --samples-per-class 5 \
    --batch-size 32 \
    --num-workers 4 \
    --soft-threshold-mode away_from_mean \
    --soft-threshold-alpha 0.9 \
    --soft-threshold-beta 4 \
    --hira-subspace-rank 1024 \
    --hira-subspace-shrink 0.5

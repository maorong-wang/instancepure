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

python visualization/hira_subspace_stability.py \
    --model-name Xu2024MIMIR_Swin-B \
    --data-dir ./dataset/imagenet \
    --model-dir ./robustbench_models \
    --device cuda:0 \
    --eps-list 0,2/255,4/255,6/255,8/255,10/255,12/255,14/255,16/255 \
    --pgd-steps 40 \
    --pgd-random-start true \
    --mask-pgd-logits false \
    --eval-examples 1000 \
    --batch-size 4 \
    --num-workers 4 \
    --cka-max-dims 2048 \
    --soft-threshold-mode away_from_mean \
    --soft-threshold-alpha 0.9 \
    --soft-threshold-beta 4 \
    --hira-subspace-rank 1024 \
    --hira-subspace-shrink 0.5

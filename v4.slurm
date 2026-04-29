#!/bin/bash
#SBATCH -p long
#SBATCH -J visualization
#SBATCH -o visualization_%A_%a.out
#SBATCH --nodes=1
#SBATCH --nodelist=elab-gpu-ubuntu06
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=25G
#SBATCH --time=72:00:00


source /home_fmg/maorong/miniconda3/etc/profile.d/conda.sh
conda activate diffpure 

python visualization/transferability_test.py \
    --model-name Xu2024MIMIR_Swin-B \
    --data-dir ./dataset/imagenet \
    --model-dir ./robustbench_models \
    --device cuda:0 \
    --eps-list 0,1/255,2/255,4/255,8/255,16/255 \
    --pgd-steps 40 \
    --pgd-random-start true \
    --mask-pgd-logits false \
    --num-classes 20 \
    --samples-per-class 50 \
    --batch-size 16 \
    --num-workers 4

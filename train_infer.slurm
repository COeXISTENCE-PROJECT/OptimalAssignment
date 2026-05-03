#!/bin/bash
#SBATCH --job-name=adttp
#SBATCH --partition=rknodes
#SBATCH --nodelist=bonk
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=slurm-%j.out
#SBATCH --qos=big_bonk

set -euo pipefail

export WANDB_MODE=online
export WANDB_DIR=/scratch/tmp/wandb
export WANDB_API_KEY="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"
mkdir -p "$WANDB_DIR"

cd /home/gorczyca/OptimalAssignment

/home/gorczyca/miniconda3/envs/wavenet_env/bin/python train_and_infer.py

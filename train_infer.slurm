#!/bin/bash
#SBATCH --job-name=test_hadamard
#SBATCH --partition=rknodes
#SBATCH --nodelist=bonk
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-%j.out
#SBATCH --qos=big_bonk

set -euo pipefail

export WANDB_MODE=online
export WANDB_DIR=/scratch/tmp/wandb
export WANDB_API_KEY="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"
mkdir -p "$WANDB_DIR"

cd /home/drozd/OptimalAssignment

/home/drozd/miniconda/envs/wavenet_env/bin/python train_and_infer.py

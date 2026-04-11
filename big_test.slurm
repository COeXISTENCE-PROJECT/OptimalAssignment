#!/bin/bash
#SBATCH --job-name=adttp_test
#SBATCH --partition=rknodes
#SBATCH --nodelist=bonk
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00

set -e
cd /home/drozd/OptimalAssignment

/home/drozd/miniconda/envs/wavenet_env/bin/python -V
/home/drozd/miniconda/envs/wavenet_env/bin/python -c "import sys; print(sys.executable)"
/home/drozd/miniconda/envs/wavenet_env/bin/python -c "import torch; print(torch.__version__)"

/home/drozd/miniconda/envs/wavenet_env/bin/python train.py \
  --device cuda:0 \
  --epochs 3 \
  --batch_size 32 \
  --num_workers 0 \
  --print_every 3000 \
  --gcn_bool \
  --addaptadj

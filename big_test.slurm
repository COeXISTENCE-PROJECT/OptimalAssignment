#!/bin/bash
#SBATCH --job-name=att_lstm
#SBATCH --partition=rknodes
#SBATCH --nodelist=bonk
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -e
cd /home/drozd/OptimalAssignment

/home/drozd/miniconda/envs/wavenet_env/bin/python -V
/home/drozd/miniconda/envs/wavenet_env/bin/python -c "import sys; print(sys.executable)"
/home/drozd/miniconda/envs/wavenet_env/bin/python -c "import torch; print(torch.__version__)"

python train.py \
  --q_dir /scratch/tmp/vec_flows_10s \
  --a_dir /scratch/tmp/vec_assignments_10s \
  --adjdata hex_adjacency_matrix.csv \
  --num_nodes 195 \
  --device cuda:0 \
  --epochs 200 \
  --batch_size 16 \
  --num_workers 4 \
  --print_every 1000 \
  --gcn_bool \
  --addaptadj

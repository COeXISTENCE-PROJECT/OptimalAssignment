#!/bin/bash
#SBATCH --job-name=adttp_infer
#SBATCH --partition=rknodes
#SBATCH --nodelist=bonk
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=slurm-%j.out

set -euo pipefail

cd /home/drozd/OptimalAssignment
python_bin=/home/drozd/miniconda/envs/wavenet_env/bin/python

# ===== dane =====
q_dir=/scratch/tmp/vec_flows_10s
a_dir=/scratch/tmp/vec_assignments_10s
adjdata=/home/drozd/OptimalAssignment/hex_adjacency_matrix.csv

# ===== checkpoint =====
checkpoint=/scratch/tmp/drozd_runs/att_lstm/long_training_test20260413_214315_exp1_best_0.4119.pth

# ===== output =====
out_root=/scratch/tmp/infer_runs/att_lstm_test
mkdir -p "$out_root"
run_name=infer_$(date +%Y%m%d_%H%M%S)
output_dir=$out_root/$run_name
mkdir -p "$output_dir"

# ===== split musi odpowiadać treningowi =====
seed=47
train_ratio=0.7
val_ratio=0.1

# ===== model hyperparams - muszą odpowiadać treningowi =====
device=cuda:0
num_nodes=195
seq_length_q=15
seq_length_a=30

sequence_model=lstm
fuse_method=Attention

nhid=32
dropout=0.3
learning_rate=0.001
weight_decay=0.0001

kernel_size=2
blocks=4
layers=2

a_embedding_size=32
a_hidden_size=32
q_rep_dim=32
fused_dim=32
mlp_hidden_dim=128
attention_num_heads=4
attention_ff_dim=32

# ===== wybór pliku testowego =====
# Opcja A: wybierz po indeksie w test split
test_index=1

# Opcja B: zamiast test_index możesz podać konkretną nazwę pliku:
# file_name=some_test_file.npy

$python_bin inference.py \
  --q_dir "$q_dir" \
  --a_dir "$a_dir" \
  --adjdata "$adjdata" \
  --checkpoint "$checkpoint" \
  --output_dir "$output_dir" \
  --seed "$seed" \
  --train_ratio "$train_ratio" \
  --val_ratio "$val_ratio" \
  --test_index "$test_index" \
  --seq_length_q "$seq_length_q" \
  --seq_length_a "$seq_length_a" \
  --device "$device" \
  --num_nodes "$num_nodes" \
  --nhid "$nhid" \
  --dropout "$dropout" \
  --learning_rate "$learning_rate" \
  --weight_decay "$weight_decay" \
  --kernel_size "$kernel_size" \
  --blocks "$blocks" \
  --layers "$layers" \
  --sequence_model "$sequence_model" \
  --fuse_method "$fuse_method" \
  --a_embedding_size "$a_embedding_size" \
  --a_hidden_size "$a_hidden_size" \
  --q_rep_dim "$q_rep_dim" \
  --fused_dim "$fused_dim" \
  --mlp_hidden_dim "$mlp_hidden_dim" \
  --attention_num_heads "$attention_num_heads" \
  --attention_ff_dim "$attention_ff_dim" \
  --addaptadj

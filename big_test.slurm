#!/bin/bash
#SBATCH --job-name=att_lstm
#SBATCH --partition=rknodes
#SBATCH --nodelist=bonk
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail

cd /home/drozd/OptimalAssignment
python_bin=/home/drozd/miniconda/envs/wavenet_env/bin/python

# ===== dane =====
q_dir=/scratch/tmp/vec_flows_10s
a_dir=/scratch/tmp/vec_assignments_10s
adjdata=hex_adjacency_matrix.csv
data_root=/scratch/tmp
save_dir=./garage/metr

# ===== sprzęt / split =====
device=cuda:0
seed=42
train_ratio=0.7
val_ratio=0.15

# ===== sekwencje =====
seq_length_q=15
seq_length_a=30
seq_length_y=1

# ===== graf =====
adjtype=doubletransition      # np. doubletransition / transition / identity
gcn_bool=1                    # 1=on, 0=off
aptonly=0                     # 1=on, 0=off
addaptadj=1                   # 1=on, 0=off
randomadj=0                   # 1=on, 0=off

# ===== model =====
in_dim=1
num_nodes=195
nhid=32
dropout=0.3
kernel_size=2
blocks=4
layers=2

target_dim=1
sequence_model=lstm           # lstm / gru
fuse_method=Attention         # Attention / Concat / MLP
a_embedding_size=32
a_hidden_size=64
q_rep_dim=32
fused_dim=64
mlp_hidden_dim=128
attention_num_heads=4
attention_ff_dim=128

# ===== trening =====
batch_size=16
learning_rate=0.001
weight_decay=0.0001
epochs=200
print_every=1000
num_workers=4
expid=1

echo "Python version:"
"$python_bin" -V
echo "Python executable:"
"$python_bin" -c "import sys; print(sys.executable)"
echo "Torch version:"
"$python_bin" -c "import torch; print(torch.__version__)"

args=()

[[ "$gcn_bool" -eq 1 ]]   && args+=(--gcn_bool)
[[ "$aptonly" -eq 1 ]]    && args+=(--aptonly)
[[ "$addaptadj" -eq 1 ]]  && args+=(--addaptadj)
[[ "$randomadj" -eq 1 ]]  && args+=(--randomadj)

"$python_bin" train.py \
  --device "$device" \
  --data "$data_root" \
  --q_dir "$q_dir" \
  --a_dir "$a_dir" \
  --seed "$seed" \
  --train_ratio "$train_ratio" \
  --val_ratio "$val_ratio" \
  --seq_length_q "$seq_length_q" \
  --seq_length_a "$seq_length_a" \
  --seq_length_y "$seq_length_y" \
  --adjdata "$adjdata" \
  --adjtype "$adjtype" \
  --in_dim "$in_dim" \
  --num_nodes "$num_nodes" \
  --nhid "$nhid" \
  --batch_size "$batch_size" \
  --learning_rate "$learning_rate" \
  --dropout "$dropout" \
  --weight_decay "$weight_decay" \
  --epochs "$epochs" \
  --print_every "$print_every" \
  --save "$save_dir" \
  --expid "$expid" \
  --kernel_size "$kernel_size" \
  --blocks "$blocks" \
  --layers "$layers" \
  --num_workers "$num_workers" \
  --target_dim "$target_dim" \
  --sequence_model "$sequence_model" \
  --fuse_method "$fuse_method" \
  --a_embedding_size "$a_embedding_size" \
  --a_hidden_size "$a_hidden_size" \
  --q_rep_dim "$q_rep_dim" \
  --fused_dim "$fused_dim" \
  --mlp_hidden_dim "$mlp_hidden_dim" \
  --attention_num_heads "$attention_num_heads" \
  --attention_ff_dim "$attention_ff_dim" \
  "${args[@]}"

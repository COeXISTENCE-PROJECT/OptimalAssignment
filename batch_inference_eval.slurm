#!/bin/bash
#SBATCH --job-name=adttp_batch_infer
#SBATCH --partition=rknodes
#SBATCH --nodelist=bonk
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=slurm-batch-infer-%j.out
#SBATCH --error=slurm-batch-infer-%j.err

set -euo pipefail

cd /home/drozd/OptimalAssignment

export PYTHONUNBUFFERED=1

python_bin=/home/drozd/miniconda/envs/wavenet_env/bin/python

# ============================================================
# Dane
# ============================================================

q_dir=/scratch/tmp/vec_flows_10s
a_dir=/scratch/tmp/vec_assignments_10s
adjdata=/home/drozd/OptimalAssignment/hex_adjacency_matrix.csv

# ============================================================
# Checkpoint
# ============================================================

checkpoint=/scratch/tmp/drozd_runs/att_lstm/long_training_test20260413_214315_exp1_best_0.4119.pth

# ============================================================
# Output
# ============================================================

out_root=/scratch/tmp/infer_runs/att_lstm_batch
mkdir -p "$out_root"

batch_name=batch_$(date +%Y%m%d_%H%M%S)

# ============================================================
# Split — musi odpowiadać treningowi
# ============================================================

seed=47
train_ratio=0.7
val_ratio=0.1

# ============================================================
# Model hyperparams — muszą odpowiadać checkpointowi
# ============================================================

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

# ============================================================
# Wybór plików
# ============================================================
# Opcja A: pierwsze N plików z test splitu
num_files=50
start_index=0

# Opcja B: zakres indeksów, np. 0:100 albo 0:200:2
# test_indices="0:100"

# Opcja C: wszystkie pliki z test splitu
# all_test_flag="--all_test"
all_test_flag=""

# Opcja D: tylko pliki pasujące regexem, np. zawierające "exp"
# file_pattern="exp"
file_pattern=""

# ============================================================
# Uruchomienie
# ============================================================

cmd=(
  "$python_bin" batch_inference_eval.py

  --inference_script /home/drozd/OptimalAssignment/inference.py
  --python_bin "$python_bin"

  --q_dir "$q_dir"
  --a_dir "$a_dir"
  --adjdata "$adjdata"
  --checkpoint "$checkpoint"

  --output_root "$out_root"
  --batch_name "$batch_name"

  --seed "$seed"
  --train_ratio "$train_ratio"
  --val_ratio "$val_ratio"

  --num_files "$num_files"
  --start_index "$start_index"

  --seq_length_q "$seq_length_q"
  --seq_length_a "$seq_length_a"

  --device "$device"
  --num_nodes "$num_nodes"

  --nhid "$nhid"
  --dropout "$dropout"
  --learning_rate "$learning_rate"
  --weight_decay "$weight_decay"

  --kernel_size "$kernel_size"
  --blocks "$blocks"
  --layers "$layers"

  --sequence_model "$sequence_model"
  --fuse_method "$fuse_method"

  --a_embedding_size "$a_embedding_size"
  --a_hidden_size "$a_hidden_size"
  --q_rep_dim "$q_rep_dim"
  --fused_dim "$fused_dim"
  --mlp_hidden_dim "$mlp_hidden_dim"
  --attention_num_heads "$attention_num_heads"
  --attention_ff_dim "$attention_ff_dim"

  --delta_t 10.0

  --addaptadj
  --continue_on_error
)

if [[ -n "$all_test_flag" ]]; then
  cmd+=("$all_test_flag")
fi

if [[ -n "$file_pattern" ]]; then
  cmd+=(--file_pattern "$file_pattern")
fi

# Jeśli chcesz użyć test_indices, odkomentuj:
# cmd+=(--test_indices "$test_indices")

echo "Running command:"
printf '%q ' "${cmd[@]}"
echo

"${cmd[@]}"
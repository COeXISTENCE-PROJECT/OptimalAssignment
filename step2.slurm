#!/bin/bash
#SBATCH --job-name=niemcy003
#SBATCH --partition=rknodes
#SBATCH --qos=normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=slurm-%j.out






# --gres=gpu:1

set -euo pipefail

cd /home/gorczyca/OptimalAssignment
python_bin=/home/gorczyca/miniconda3/envs/wavenet_env/bin/python

# ===== data =====
# /scratch/tmp/4k_grid/new_assignments_10s/
base_assignment=/scratch/tmp/100k_grid/new_assignments_10s/exp_50000.npy
q_seed=/scratch/tmp/100k_grid/new_flows_10s/exp_50000.npy
adjdata=/home/gorczyca/OptimalAssignment/hex_adjacency_matrix.csv

# ===== checkpoint =====
# Leave empty to use the simple ADTTP(A6) objective only.
# Set to a trained GenTTP checkpoint to score candidates with model rollout.
checkpoint=/scratch/tmp/drozd/g_ADTTP_tests_new/run_20260609_094422/checkpoint_best.pth

# ===== output =====
out_root=/scratch/tmp/grid_search_runs
mkdir -p "$out_root"
run_name=grid_search_$(date +%Y%m%d_%H%M%S)
output_dir=$out_root/$run_name
mkdir -p "$output_dir"

# ===== grid search setup =====
device=cuda:0
num_nodes=195
m=10
delta_t=1.0
max_candidates=20

# ===== penalties =====
# Legacy OD proxy. Keep 0.0 if using separate origin/destination penalties.
od_weight=0.0
origin_weight=100.0
destination_weight=100.0
admissibility_weight=1000.0
connectivity_weight=1000.0
projection_samples=16
projection_noise=0.7
preserve_total_by_t=true

# REQUIRED if origin/destination losses are active.
# Example only — replace with correct node ids.
origins="0"
destinations="194"

# Optional old OD proxy penalty. Leave empty to disable.
target_od=

# ===== model hyperparams - must match checkpoint =====
seq_length_q=15
seq_length_a=30
sequence_model=lstm
fuse_method=attention
nhid=128
dropout=0
learning_rate=0.001
weight_decay=0.01
kernel_size=2
blocks=5
layers=2
a_embedding_size=96
a_hidden_size=128
q_rep_dim=128
fused_dim=128
mlp_hidden_dim=256
attention_num_heads=2
attention_ff_dim=64
fuse_attention_num_heads=2
fuse_attention_ff_dim=128
fuse_gated_update=true

cmd=(
  "$python_bin" -u gradient_search.py
  --base_assignment "$base_assignment"
  --output_dir "$output_dir"
  --num_nodes "$num_nodes"
  --m "$m"
  --delta_t "$delta_t"
  --max_candidates "$max_candidates"
  --od_weight "$od_weight"
  --admissibility_weight "$admissibility_weight"
  --device "$device"
  --seq_length_q "$seq_length_q"
  --seq_length_a "$seq_length_a"
  --sequence_model "$sequence_model"
  --fuse_method "$fuse_method"
  --nhid "$nhid"
  --dropout "$dropout"
  --learning_rate "$learning_rate"
  --weight_decay "$weight_decay"
  --kernel_size "$kernel_size"
  --blocks "$blocks"
  --layers "$layers"
  --a_embedding_size "$a_embedding_size"
  --a_hidden_size "$a_hidden_size"
  --q_rep_dim "$q_rep_dim"
  --fused_dim "$fused_dim"
  --mlp_hidden_dim "$mlp_hidden_dim"
  --attention_num_heads "$attention_num_heads"
  --attention_ff_dim "$attention_ff_dim"
  --fuse_attention_ff_dim "$fuse_attention_ff_dim"
  --fuse_attention_num_heads "$fuse_attention_num_heads"
  --gcn_bool
  --origin_weight "$origin_weight"
  --destination_weight "$destination_weight"
  --admissibility_weight "$admissibility_weight"
  --connectivity_weight "$connectivity_weight"
  --adjdata "$adjdata"
  --projection_samples "$projection_samples"
  --projection_noise "$projection_noise"
  --use_genttp_gradient
  --checkpoint "$checkpoint"
  --q_seed "$q_seed"
)

if [[ "$preserve_total_by_t" == "true" ]]; then
  cmd+=(--preserve_total_by_t)
fi

if [[ "$fuse_gated_update" == "true" ]]; then
  cmd+=(--fuse_gated_update)
fi

if [[ -n "$origins" ]]; then
  cmd+=(--origins "$origins")
fi

if [[ -n "$destinations" ]]; then
  cmd+=(--destinations "$destinations")
fi

"${cmd[@]}"

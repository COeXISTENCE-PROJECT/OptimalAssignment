#!/bin/bash
#SBATCH --job-name=att_lstm
#SBATCH --partition=rknodes
#SBATCH --nodelist=bonk
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail

cd /home/drozd/OptimalAssignment

PYTHON=/home/drozd/miniconda/envs/wavenet_env/bin/python

# =========================
# ŚCIEŻKI / DANE
# =========================
Q_DIR=/scratch/tmp/vec_flows_10s
A_DIR=/scratch/tmp/vec_assignments_10s
ADJDATA=hex_adjacency_matrix.csv
DATA_ROOT=/scratch/tmp
SAVE_DIR=./garage/metr

# =========================
# URZĄDZENIE / SEED / SPLIT
# =========================
DEVICE=cuda:0
SEED=42
TRAIN_RATIO=0.7
VAL_RATIO=0.15

# =========================
# DŁUGOŚCI SEKWENCJI
# =========================
SEQ_LENGTH_Q=15
SEQ_LENGTH_A=30
SEQ_LENGTH_Y=1

# =========================
# GRAF
# =========================
ADJTYPE=doubletransition
GCN_BOOL=1
APTONLY=0
ADDAPTADJ=1
RANDOMADJ=0

# =========================
# PARAMETRY MODELU
# =========================
IN_DIM=1
NUM_NODES=195
NHID=32
DROPOUT=0.3
KERNEL_SIZE=2
BLOCKS=4
LAYERS=2

TARGET_DIM=1
SEQUENCE_MODEL=lstm
FUSE_METHOD=Attention
A_EMBEDDING_SIZE=32
A_HIDDEN_SIZE=64
Q_REP_DIM=32
FUSED_DIM=64
MLP_HIDDEN_DIM=128
ATTENTION_NUM_HEADS=4
ATTENTION_FF_DIM=128

# =========================
# TRENING
# =========================
BATCH_SIZE=16
LEARNING_RATE=0.001
WEIGHT_DECAY=0.0001
EPOCHS=200
PRINT_EVERY=1000
NUM_WORKERS=4
EXPID=1

echo "Python version:"
"$PYTHON" -V
echo "Python executable:"
"$PYTHON" -c "import sys; print(sys.executable)"
echo "Torch version:"
"$PYTHON" -c "import torch; print(torch.__version__)"

# Flagi boolowskie składamy warunkowo
ARGS=()

[[ "$GCN_BOOL" -eq 1 ]]   && ARGS+=(--gcn_bool)
[[ "$APTONLY" -eq 1 ]]    && ARGS+=(--aptonly)
[[ "$ADDAPTADJ" -eq 1 ]]  && ARGS+=(--addaptadj)
[[ "$RANDOMADJ" -eq 1 ]]  && ARGS+=(--randomadj)

"$PYTHON" train.py \
  --device "$DEVICE" \
  --data "$DATA_ROOT" \
  --q_dir "$Q_DIR" \
  --a_dir "$A_DIR" \
  --seed "$SEED" \
  --train_ratio "$TRAIN_RATIO" \
  --val_ratio "$VAL_RATIO" \
  --seq_length_q "$SEQ_LENGTH_Q" \
  --seq_length_a "$SEQ_LENGTH_A" \
  --seq_length_y "$SEQ_LENGTH_Y" \
  --adjdata "$ADJDATA" \
  --adjtype "$ADJTYPE" \
  --in_dim "$IN_DIM" \
  --num_nodes "$NUM_NODES" \
  --nhid "$NHID" \
  --batch_size "$BATCH_SIZE" \
  --learning_rate "$LEARNING_RATE" \
  --dropout "$DROPOUT" \
  --weight_decay "$WEIGHT_DECAY" \
  --epochs "$EPOCHS" \
  --print_every "$PRINT_EVERY" \
  --save "$SAVE_DIR" \
  --expid "$EXPID" \
  --kernel_size "$KERNEL_SIZE" \
  --blocks "$BLOCKS" \
  --layers "$LAYERS" \
  --num_workers "$NUM_WORKERS" \
  --target_dim "$TARGET_DIM" \
  --sequence_model "$SEQUENCE_MODEL" \
  --fuse_method "$FUSE_METHOD" \
  --a_embedding_size "$A_EMBEDDING_SIZE" \
  --a_hidden_size "$A_HIDDEN_SIZE" \
  --q_rep_dim "$Q_REP_DIM" \
  --fused_dim "$FUSED_DIM" \
  --mlp_hidden_dim "$MLP_HIDDEN_DIM" \
  --attention_num_heads "$ATTENTION_NUM_HEADS" \
  --attention_ff_dim "$ATTENTION_FF_DIM" \
  "${ARGS[@]}"

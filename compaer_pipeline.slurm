#!/bin/bash
#SBATCH --job-name=model_compare
#SBATCH --partition=rknodes
#SBATCH --nodelist=bonk
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=slurm-model-compare-%j.out
#SBATCH --error=slurm-model-compare-%j.err

set -euo pipefail

# ============================================================
# Ścieżki
# ============================================================

PROJECT_DIR=${PROJECT_DIR:-/home/drozd/OptimalAssignment}
PYTHON_BIN=${PYTHON_BIN:-/home/drozd/miniconda/envs/wavenet_env/bin/python}

SAVE_SCRIPT=${SAVE_SCRIPT:-$PROJECT_DIR/save_inference_run_for_comparison.py}
VIS_SCRIPT=${VIS_SCRIPT:-$PROJECT_DIR/aggregate_visualize_comparison.py}

# Folder całego porównania.
# Najlepiej ustawiaj jawnie przy sbatch:
#   COMPARISON_DIR=/scratch/tmp/model_comparisons/exp_001
COMPARISON_DIR=${COMPARISON_DIR:-/scratch/tmp/model_comparisons/compare_$(date +%Y%m%d_%H%M%S)}

# Tryby:
#   MODE=run  -> odpala jedną inferencję i zapisuje ją do comparison_dir
#   MODE=viz  -> robi agregację + wykresy z gotowego comparison_dir
#   MODE=both -> run + viz
MODE=${MODE:-run}

# Dla MODE=run/both:
MODEL_NAME=${MODEL_NAME:-}
CMD_FILE=${CMD_FILE:-}
SOURCE_DIR=${SOURCE_DIR:-}

# Dla MODE=viz/both:
BASE_MODEL=${BASE_MODEL:-}
ROLLING_WINDOW=${ROLLING_WINDOW:-25}
DPI=${DPI:-240}

# Opcjonalne:
EVAL_START=${EVAL_START:-}
COPY_MODE=${COPY_MODE:-copy}       # copy / symlink / hardlink
OVERWRITE=${OVERWRITE:-0}
REUSE_EXISTING=${REUSE_EXISTING:-0}
ALLOW_DIFFERENT_REAL=${ALLOW_DIFFERENT_REAL:-0}

# ============================================================
# Start
# ============================================================

cd "$PROJECT_DIR"
mkdir -p "$COMPARISON_DIR"

echo "[INFO] PROJECT_DIR:    $PROJECT_DIR"
echo "[INFO] PYTHON_BIN:     $PYTHON_BIN"
echo "[INFO] MODE:           $MODE"
echo "[INFO] COMPARISON_DIR: $COMPARISON_DIR"
echo "[INFO] MODEL_NAME:     ${MODEL_NAME:-<none>}"
echo "[INFO] CMD_FILE:       ${CMD_FILE:-<none>}"
echo "[INFO] SOURCE_DIR:     ${SOURCE_DIR:-<none>}"
echo "[INFO] BASE_MODEL:     ${BASE_MODEL:-<none>}"

# ============================================================
# MODE=run albo MODE=both
# ============================================================

if [[ "$MODE" == "run" || "$MODE" == "both" ]]; then
  if [[ -z "$MODEL_NAME" ]]; then
    echo "[ERROR] MODE=$MODE wymaga MODEL_NAME"
    exit 1
  fi

  save_args=(
    "$SAVE_SCRIPT"
    --comparison_dir "$COMPARISON_DIR"
    --model_name "$MODEL_NAME"
    --cwd "$PROJECT_DIR"
    --copy_mode "$COPY_MODE"
  )

  if [[ -n "$CMD_FILE" ]]; then
    save_args+=(--cmd_file "$CMD_FILE")
  fi

  if [[ -n "$SOURCE_DIR" ]]; then
    save_args+=(--source_dir "$SOURCE_DIR")
  fi

  if [[ "$OVERWRITE" == "1" ]]; then
    save_args+=(--overwrite)
  fi

  if [[ "$REUSE_EXISTING" == "1" ]]; then
    save_args+=(--reuse_existing)
  fi

  if [[ -n "$EVAL_START" ]]; then
    save_args+=(--seed_steps "$EVAL_START")
  fi

  echo "[INFO] Running save/inference step:"
  printf ' %q' "$PYTHON_BIN" "${save_args[@]}"
  echo

  "$PYTHON_BIN" "${save_args[@]}"

  echo "[OK] Save/inference step finished."
fi

# ============================================================
# MODE=viz albo MODE=both
# ============================================================

if [[ "$MODE" == "viz" || "$MODE" == "both" ]]; then
  if [[ -z "$BASE_MODEL" ]]; then
    echo "[ERROR] MODE=$MODE wymaga BASE_MODEL"
    exit 1
  fi

  vis_args=(
    "$VIS_SCRIPT"
    --comparison_dir "$COMPARISON_DIR"
    --base_model "$BASE_MODEL"
    --rolling_window "$ROLLING_WINDOW"
    --dpi "$DPI"
  )

  if [[ -n "$EVAL_START" ]]; then
    vis_args+=(--eval_start "$EVAL_START")
  fi

  if [[ "$ALLOW_DIFFERENT_REAL" == "1" ]]; then
    vis_args+=(--allow_different_real)
  fi

  echo "[INFO] Running visualization step:"
  printf ' %q' "$PYTHON_BIN" "${vis_args[@]}"
  echo

  "$PYTHON_BIN" "${vis_args[@]}"

  echo "[OK] Visualization step finished."
fi

echo "[OK] Done."
echo "[OK] Results: $COMPARISON_DIR"
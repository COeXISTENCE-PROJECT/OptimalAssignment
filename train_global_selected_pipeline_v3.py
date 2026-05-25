#!/usr/bin/env python3
"""
Pipeline:
1) train model and save checkpoint,
2) run GLOBAL GPU inference on the test/global set,
3) select simulations by global total travel time statistics:
   - lowest absolute percentage error of total TT,
   - highest absolute percentage error of total TT,
   - lowest real total TT,
   - highest real total TT,
4) run detailed inference + visualization for the selected simulations.

Important:
- This script does NOT run the SLURM wrapper `batch_inference_eval_gpu`.
- It runs the Python global-inference entrypoint directly: GLOBAL_INFER_SCRIPT.
- The checkpoint is taken from the training stage of THIS run, not hardcoded.
- Training data directories and inference data directories are separate.
- One RUN_DIR contains config, logs, training, global inference, selected detailed inference,
  visualizations, and summary files.
"""

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


# =============================================================================
# Paths / environment
# =============================================================================

PYTHON_BIN = "/home/drozd/miniconda/envs/wavenet_env/bin/python"
PROJECT_DIR = Path("/home/drozd/OptimalAssignment").resolve()

# ===== data used for TRAINING =====
# These directories are used only by train.py.
TRAIN_Q_DIR = "/scratch/tmp/10_grid/new_flows_10s"
TRAIN_A_DIR = "/scratch/tmp/10_grid/new_assignments_10s"
DATA_ROOT = "/scratch/tmp"

# ===== data used for GLOBAL + DETAILED INFERENCE =====
# These directories are intentionally independent from the training directories.
# Set them to the folder with the simulations/assignments you want to evaluate.
INFER_Q_DIR = "/scratch/tmp/10k_grid/new_flows_10s"
INFER_A_DIR = "/scratch/tmp/10k_grid/new_assignments_10s"

ADJDATA = "/scratch/tmp/21k_exps/21k_hex_adjacency_matrix.csv"

# ===== one root for the whole pipeline =====
PIPELINE_ROOT = Path("/scratch/tmp/g_ADTTP_tests_new")
RUN_STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_NAME = f"run_{RUN_STAMP}"
RUN_DIR = PIPELINE_ROOT / RUN_NAME

RESULTS_DIR = RUN_DIR / "results_summary"
LOGS_DIR = RUN_DIR / "logs"

# ===== subfolders inside this run =====
TRAIN_PREFIX = RUN_DIR / "training"
INFERENCE_ROOT = RUN_DIR / "inference"
GLOBAL_INFER_ROOT = INFERENCE_ROOT / "global"
DETAIL_INFER_ROOT = INFERENCE_ROOT / "selected_detailed"
VIS_DIR = RUN_DIR / "visual_selected"

# =============================================================================
# Hardware / split
# =============================================================================

# Training can stay on CPU if that is intentional.
# Global and detailed inference use separate GPU settings below.
DEVICE = "cuda:0"
SEED = 42
TRAIN_RATIO = 0.7
VAL_RATIO = 0.1

# ===== sequences =====
SEQ_LENGTH_Q = 15
SEQ_LENGTH_A = 30
SEQ_LENGTH_Y = 1

# ===== graph =====
ADJTYPE = "doubletransition"
GCN_BOOL = True
APTONLY = False
ADDAPTADJ = False
RANDOMADJ = False

# ===== model =====
IN_DIM = 1
NUM_NODES = 195
NHID = 128
DROPOUT = 0.1
KERNEL_SIZE = 2
BLOCKS = 4
LAYERS = 2

TARGET_DIM = 1
SEQUENCE_MODEL = "lstm"
FUSE_METHOD = "attention"  # "attention" / "concatenate" / "hadamard"

A_EMBEDDING_SIZE = 64
A_HIDDEN_SIZE = 128
Q_REP_DIM = 128
FUSED_DIM = 256
MLP_HIDDEN_DIM = 256

# To dotyczy SEQUENCE_MODEL="attention"
ATTENTION_NUM_HEADS = 8
ATTENTION_FF_DIM = 128

# To dotyczy FUSE_METHOD="attention"
FUSE_ATTENTION_NUM_HEADS = 4
FUSE_ATTENTION_FF_DIM = 256
FUSE_GATED_UPDATE = True

# ===== training =====
BATCH_SIZE = 128
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
EPOCHS = 100
PRINT_EVERY = 2000
NUM_WORKERS = 8
EXPID = 1

PREFETCH_FACTOR = 4
LOAD_TO_RAM = True

# =============================================================================
# Global GPU inference
# =============================================================================

# This is the Python entrypoint called by your SLURM wrapper.
# Do NOT point this at the SLURM file named `batch_inference_eval_gpu`.
# Point it at the underlying Python script, usually `batch_inference_eval_gpu.py`.
GLOBAL_INFER_SCRIPT = "batch_inference_eval_gpu.py"
GLOBAL_INFER_DEVICE = "cuda:0"
GLOBAL_BATCH_NAME = f"global_gpu_{RUN_STAMP}"

GLOBAL_NUM_FILES = 1000
GLOBAL_START_INDEX = 0
GLOBAL_BATCH_SIZE_FILES = 16
GLOBAL_DELTA_T = 10.0
GLOBAL_CONTINUE_ON_ERROR = True

# =============================================================================
# Selection for detailed inference
# =============================================================================

# Number of simulations per category. Duplicates are removed.
NUM_SELECT_EACH = 3
DETAIL_INFER_DEVICE = "cuda:0"

SELECTED_SIMULATIONS_CSV = RUN_DIR / "selected_simulations.csv"
SELECTED_SIMULATIONS_JSON = RUN_DIR / "selected_simulations.json"

# ===== detailed visualization =====
TOP_K_NODES = 20
NUM_BEST_NODES = 5
NUM_MIDDLE_NODES = 5
NUM_WORST_NODES = 5

# ===== Weights & Biases =====
WANDB_ENABLED = True
WANDB_PROJECT = "adttp"
WANDB_ENTITY = "lime-pps-uniwersytet-jagiello-ski-w-krakowie"
WANDB_RUN_NAME = RUN_NAME
WANDB_MODE = "online"  # "online", "offline" or "disabled"
WANDB_DIR = RUN_DIR / "wandb"


# =============================================================================
# Helpers
# =============================================================================


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def print_header(title: str):
    print("\n" + "=" * 90, flush=True)
    print(title, flush=True)
    print("=" * 90, flush=True)


def run_command(
    cmd: list[Any],
    cwd: Path | None = None,
    log_file: Path | None = None,
    env: dict[str, str] | None = None,
    check: bool = True,
):
    printable = " ".join(shlex.quote(str(x)) for x in cmd)
    print(f"\n[RUN] {printable}\n", flush=True)

    merged_env = os.environ.copy()
    merged_env["PYTHONUNBUFFERED"] = "1"
    if env is not None:
        merged_env.update(env)

    if log_file is None:
        return subprocess.run(cmd, cwd=cwd, check=check, env=merged_env)

    log_file.parent.mkdir(parents=True, exist_ok=True)

    with open(log_file, "w", encoding="utf-8", buffering=1) as f:
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=merged_env,
        )

        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            f.write(line)
            f.flush()

        ret = proc.wait()
        if check and ret != 0:
            raise subprocess.CalledProcessError(ret, cmd)
        return ret


def append_bool_flag(cmd: list[Any], flag_name: str, value: bool):
    if value:
        cmd.append(flag_name)


def build_wandb_env():
    if not WANDB_ENABLED:
        return None

    WANDB_DIR.mkdir(parents=True, exist_ok=True)
    return {
        "WANDB_DIR": str(WANDB_DIR),
        "WANDB_MODE": WANDB_MODE,
    }


def validate_project_entrypoint(script_name: str):
    script_path = PROJECT_DIR / script_name
    if not script_path.exists():
        raise FileNotFoundError(
            f"Nie znaleziono skryptu Python: {script_path}. "
            f"Zmień GLOBAL_INFER_SCRIPT albo dodaj ten plik do PROJECT_DIR."
        )


def validate_input_dirs():
    required_dirs = {
        "TRAIN_Q_DIR": TRAIN_Q_DIR,
        "TRAIN_A_DIR": TRAIN_A_DIR,
        "INFER_Q_DIR": INFER_Q_DIR,
        "INFER_A_DIR": INFER_A_DIR,
    }

    missing = [f"{name}={path}" for name, path in required_dirs.items() if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(
            "Nie znaleziono wymaganych katalogów danych:\n"
            + "\n".join(missing)
        )


def find_best_checkpoint(search_root: Path, expid: int | None = None) -> Path:
    """
    Looks recursively for checkpoints like:
        *_exp1_best_0.4119.pth
    If this pattern is not found, returns the newest .pth file.
    """
    candidates = sorted(search_root.rglob("*.pth"))
    if not candidates:
        raise FileNotFoundError(f"Nie znaleziono checkpointów .pth w: {search_root}")

    parsed: list[tuple[float, Path]] = []
    pattern = re.compile(r".*_exp(?P<expid>\d+)_best_(?P<score>[0-9.]+)\.pth$")

    for path in candidates:
        m = pattern.match(path.name)
        if not m:
            continue
        file_expid = int(m.group("expid"))
        score = float(m.group("score"))
        if expid is not None and file_expid != expid:
            continue
        parsed.append((score, path))

    if parsed:
        parsed.sort(key=lambda x: x[0])
        return parsed[0][1]

    newest = max(candidates, key=lambda p: p.stat().st_mtime)
    print("[WARN] Nie udało się sparsować checkpointu po wzorcu *_exp*_best_*.pth", flush=True)
    print(f"[WARN] Biorę najnowszy plik: {newest}", flush=True)
    return newest


def copy_if_exists(src: Path, dst_dir: Path, new_name: str | None = None):
    src = Path(src)
    if not src.exists():
        print(f"[WARN] Nie znaleziono pliku: {src}", flush=True)
        return

    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / (new_name if new_name else src.name)
    shutil.copy2(src, dst)
    print(f"[INFO] Skopiowano: {src} -> {dst}", flush=True)


def copy_tree_if_exists(src: Path, dst: Path):
    src = Path(src)
    dst = Path(dst)

    if not src.exists():
        print(f"[WARN] Nie znaleziono katalogu: {src}", flush=True)
        return

    if dst.exists():
        shutil.rmtree(dst)

    shutil.copytree(src, dst)
    print(f"[INFO] Skopiowano katalog: {src} -> {dst}", flush=True)


def copy_matching_files(src_root: Path, dst_root: Path, patterns: list[str]):
    src_root = Path(src_root)
    if not src_root.exists():
        print(f"[WARN] Nie znaleziono katalogu: {src_root}", flush=True)
        return

    for pattern in patterns:
        for src in src_root.rglob(pattern):
            if src.is_file():
                rel = src.relative_to(src_root)
                dst = dst_root / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                print(f"[INFO] Skopiowano: {src} -> {dst}", flush=True)


def copy_best_checkpoint_to_run_root(best_checkpoint: Path):
    target = RUN_DIR / "checkpoint_best.pth"
    shutil.copy2(best_checkpoint, target)
    print(f"[INFO] Skopiowano najlepszy checkpoint do: {target}", flush=True)


# =============================================================================
# Config
# =============================================================================


def save_pipeline_config(best_checkpoint: Path | None = None, selected_count: int | None = None):
    config = {
        "run_name": RUN_NAME,
        "run_dir": str(RUN_DIR),
        "python_bin": PYTHON_BIN,
        "project_dir": str(PROJECT_DIR),
        "folders": {
            "training": str(TRAIN_PREFIX),
            "inference_root": str(INFERENCE_ROOT),
            "global_inference": str(GLOBAL_INFER_ROOT),
            "selected_detailed_inference": str(DETAIL_INFER_ROOT),
            "visual_selected": str(VIS_DIR),
            "logs": str(LOGS_DIR),
            "results_summary": str(RESULTS_DIR),
        },
        "wandb": {
            "enabled": WANDB_ENABLED,
            "project": WANDB_PROJECT,
            "entity": WANDB_ENTITY,
            "run_name": WANDB_RUN_NAME,
            "mode": WANDB_MODE,
            "dir": str(WANDB_DIR),
        },
        "data": {
            "training": {
                "q_dir": TRAIN_Q_DIR,
                "a_dir": TRAIN_A_DIR,
                "data_root": DATA_ROOT,
            },
            "inference": {
                "q_dir": INFER_Q_DIR,
                "a_dir": INFER_A_DIR,
            },
            "adjdata": ADJDATA,
        },
        "split": {
            "seed": SEED,
            "train_ratio": TRAIN_RATIO,
            "val_ratio": VAL_RATIO,
        },
        "sequence": {
            "seq_length_q": SEQ_LENGTH_Q,
            "seq_length_a": SEQ_LENGTH_A,
            "seq_length_y": SEQ_LENGTH_Y,
        },
        "graph": {
            "adjtype": ADJTYPE,
            "gcn_bool": GCN_BOOL,
            "aptonly": APTONLY,
            "addaptadj": ADDAPTADJ,
            "randomadj": RANDOMADJ,
        },
        "model": {
            "in_dim": IN_DIM,
            "num_nodes": NUM_NODES,
            "nhid": NHID,
            "dropout": DROPOUT,
            "kernel_size": KERNEL_SIZE,
            "blocks": BLOCKS,
            "layers": LAYERS,
            "target_dim": TARGET_DIM,
            "sequence_model": SEQUENCE_MODEL,
            "fuse_method": FUSE_METHOD,
            "a_embedding_size": A_EMBEDDING_SIZE,
            "a_hidden_size": A_HIDDEN_SIZE,
            "q_rep_dim": Q_REP_DIM,
            "fused_dim": FUSED_DIM,
            "mlp_hidden_dim": MLP_HIDDEN_DIM,
            "attention_num_heads": ATTENTION_NUM_HEADS,
            "attention_ff_dim": ATTENTION_FF_DIM,
            "fuse_method": FUSE_METHOD,
            "fuse_attention_num_heads": FUSE_ATTENTION_NUM_HEADS,
            "fuse_attention_ff_dim": FUSE_ATTENTION_FF_DIM,
            "fuse_gated_update": FUSE_GATED_UPDATE,
        },
        "training": {
            "device": DEVICE,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "epochs": EPOCHS,
            "print_every": PRINT_EVERY,
            "num_workers": NUM_WORKERS,
            "expid": EXPID,
        },
        "global_inference": {
            "script": GLOBAL_INFER_SCRIPT,
            "q_dir": INFER_Q_DIR,
            "a_dir": INFER_A_DIR,
            "device": GLOBAL_INFER_DEVICE,
            "output_root": str(GLOBAL_INFER_ROOT),
            "batch_name": GLOBAL_BATCH_NAME,
            "num_files": GLOBAL_NUM_FILES,
            "start_index": GLOBAL_START_INDEX,
            "batch_size_files": GLOBAL_BATCH_SIZE_FILES,
            "delta_t": GLOBAL_DELTA_T,
            "continue_on_error": GLOBAL_CONTINUE_ON_ERROR,
        },
        "selection": {
            "num_select_each": NUM_SELECT_EACH,
            "selected_csv": str(SELECTED_SIMULATIONS_CSV),
            "selected_json": str(SELECTED_SIMULATIONS_JSON),
            "selected_count": selected_count,
        },
        "detailed_inference": {
            "q_dir": INFER_Q_DIR,
            "a_dir": INFER_A_DIR,
            "device": DETAIL_INFER_DEVICE,
            "output_root": str(DETAIL_INFER_ROOT),
        },
        "visualization": {
            "output_root": str(VIS_DIR),
            "top_k_nodes": TOP_K_NODES,
            "num_best_nodes": NUM_BEST_NODES,
            "num_middle_nodes": NUM_MIDDLE_NODES,
            "num_worst_nodes": NUM_WORST_NODES,
        },
        "best_checkpoint": None if best_checkpoint is None else str(best_checkpoint),
    }

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    with open(RUN_DIR / "pipeline_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


# =============================================================================
# Commands
# =============================================================================


def build_train_command():
    cmd: list[Any] = [
        PYTHON_BIN,
        "-u",
        "train.py",
        "--device",
        DEVICE,
        "--data_old",
        DATA_ROOT,
        "--q_dir",
        TRAIN_Q_DIR,
        "--a_dir",
        TRAIN_A_DIR,
        "--seed",
        str(SEED),
        "--train_ratio",
        str(TRAIN_RATIO),
        "--val_ratio",
        str(VAL_RATIO),
        "--seq_length_q",
        str(SEQ_LENGTH_Q),
        "--seq_length_a",
        str(SEQ_LENGTH_A),
        "--seq_length_y",
        str(SEQ_LENGTH_Y),
        "--adjdata",
        ADJDATA,
        "--adjtype",
        ADJTYPE,
        "--in_dim",
        str(IN_DIM),
        "--num_nodes",
        str(NUM_NODES),
        "--nhid",
        str(NHID),
        "--batch_size",
        str(BATCH_SIZE),
        "--learning_rate",
        str(LEARNING_RATE),
        "--dropout",
        str(DROPOUT),
        "--weight_decay",
        str(WEIGHT_DECAY),
        "--epochs",
        str(EPOCHS),
        "--print_every",
        str(PRINT_EVERY),
        "--save_dir",
        str(TRAIN_PREFIX),
        "--expid",
        str(EXPID),
        "--kernel_size",
        str(KERNEL_SIZE),
        "--blocks",
        str(BLOCKS),
        "--layers",
        str(LAYERS),
        "--num_workers",
        str(NUM_WORKERS),
        "--prefetch_factor",
        str(PREFETCH_FACTOR),
        "--target_dim",
        str(TARGET_DIM),
        "--sequence_model",
        str(SEQUENCE_MODEL),
        "--fuse_method",
        str(FUSE_METHOD),
        "--a_embedding_size",
        str(A_EMBEDDING_SIZE),
        "--a_hidden_size",
        str(A_HIDDEN_SIZE),
        "--q_rep_dim",
        str(Q_REP_DIM),
        "--fused_dim",
        str(FUSED_DIM),
        "--mlp_hidden_dim",
        str(MLP_HIDDEN_DIM),
        "--attention_num_heads",
        str(ATTENTION_NUM_HEADS),
        "--attention_ff_dim",
        str(ATTENTION_FF_DIM),
        "--fuse_attention_num_heads",
        str(FUSE_ATTENTION_NUM_HEADS),
        "--fuse_attention_ff_dim",
        str(FUSE_ATTENTION_FF_DIM),

    ]

    append_bool_flag(cmd, "--gcn_bool", GCN_BOOL)
    append_bool_flag(cmd, "--aptonly", APTONLY)
    append_bool_flag(cmd, "--addaptadj", ADDAPTADJ)
    append_bool_flag(cmd, "--randomadj", RANDOMADJ)
    append_bool_flag(cmd, "--load_to_ram", LOAD_TO_RAM)
    append_bool_flag(cmd, "--fuse_gated_update", FUSE_GATED_UPDATE)

    if WANDB_ENABLED:
        cmd.append("--wandb")
        cmd.extend(["--wandb_project", WANDB_PROJECT])
        cmd.extend(["--wandb_run_name", WANDB_RUN_NAME])
        if WANDB_ENTITY is not None:
            cmd.extend(["--wandb_entity", WANDB_ENTITY])

    return cmd


def build_global_inference_command(checkpoint_path: Path):
    """
    Direct Python equivalent of the SLURM wrapper logic.
    No hardcoded checkpoint. No SLURM wrapper call.
    """
    cmd: list[Any] = [
        PYTHON_BIN,
        "-u",
        GLOBAL_INFER_SCRIPT,
        "--q_dir",
        INFER_Q_DIR,
        "--a_dir",
        INFER_A_DIR,
        "--adjdata",
        ADJDATA,
        "--checkpoint",
        str(checkpoint_path),
        "--output_root",
        str(GLOBAL_INFER_ROOT),
        "--batch_name",
        GLOBAL_BATCH_NAME,
        "--seed",
        str(SEED),
        "--train_ratio",
        str(TRAIN_RATIO),
        "--val_ratio",
        str(VAL_RATIO),
        "--num_files",
        str(GLOBAL_NUM_FILES),
        "--start_index",
        str(GLOBAL_START_INDEX),
        "--seq_length_q",
        str(SEQ_LENGTH_Q),
        "--seq_length_a",
        str(SEQ_LENGTH_A),
        "--device",
        GLOBAL_INFER_DEVICE,
        "--num_nodes",
        str(NUM_NODES),
        "--nhid",
        str(NHID),
        "--dropout",
        str(DROPOUT),
        "--learning_rate",
        str(LEARNING_RATE),
        "--weight_decay",
        str(WEIGHT_DECAY),
        "--kernel_size",
        str(KERNEL_SIZE),
        "--blocks",
        str(BLOCKS),
        "--layers",
        str(LAYERS),
        "--sequence_model",
        str(SEQUENCE_MODEL),
        "--fuse_method",
        str(FUSE_METHOD),
        "--a_embedding_size",
        str(A_EMBEDDING_SIZE),
        "--a_hidden_size",
        str(A_HIDDEN_SIZE),
        "--q_rep_dim",
        str(Q_REP_DIM),
        "--fused_dim",
        str(FUSED_DIM),
        "--mlp_hidden_dim",
        str(MLP_HIDDEN_DIM),
        "--attention_num_heads",
        str(ATTENTION_NUM_HEADS),
        "--attention_ff_dim",
        str(ATTENTION_FF_DIM),
        "--fuse_attention_num_heads",
        str(FUSE_ATTENTION_NUM_HEADS),
        "--fuse_attention_ff_dim",
        str(FUSE_ATTENTION_FF_DIM),
        "--batch_size_files",
        str(GLOBAL_BATCH_SIZE_FILES),
        "--delta_t",
        str(GLOBAL_DELTA_T),

    ]

    append_bool_flag(cmd, "--gcn_bool", GCN_BOOL)
    append_bool_flag(cmd, "--aptonly", APTONLY)
    append_bool_flag(cmd, "--addaptadj", ADDAPTADJ)
    append_bool_flag(cmd, "--randomadj", RANDOMADJ)
    append_bool_flag(cmd, "--continue_on_error", GLOBAL_CONTINUE_ON_ERROR)
    append_bool_flag(cmd, "--fuse_gated_update", FUSE_GATED_UPDATE)

    return cmd


def build_detailed_inference_command(checkpoint_path: Path, selected: dict[str, Any]):
    out_dir = Path(selected["detail_output_dir"])

    cmd: list[Any] = [
        PYTHON_BIN,
        "-u",
        "inference.py",
        "--q_dir",
        INFER_Q_DIR,
        "--a_dir",
        INFER_A_DIR,
        "--adjdata",
        ADJDATA,
        "--checkpoint",
        str(checkpoint_path),
        "--output_dir",
        str(out_dir),
        "--seed",
        str(SEED),
        "--train_ratio",
        str(TRAIN_RATIO),
        "--val_ratio",
        str(VAL_RATIO),
        "--seq_length_q",
        str(SEQ_LENGTH_Q),
        "--seq_length_a",
        str(SEQ_LENGTH_A),
        "--device",
        DETAIL_INFER_DEVICE,
        "--num_nodes",
        str(NUM_NODES),
        "--nhid",
        str(NHID),
        "--dropout",
        str(DROPOUT),
        "--learning_rate",
        str(LEARNING_RATE),
        "--weight_decay",
        str(WEIGHT_DECAY),
        "--kernel_size",
        str(KERNEL_SIZE),
        "--blocks",
        str(BLOCKS),
        "--layers",
        str(LAYERS),
        "--sequence_model",
        str(SEQUENCE_MODEL),
        "--fuse_method",
        str(FUSE_METHOD),
        "--a_embedding_size",
        str(A_EMBEDDING_SIZE),
        "--a_hidden_size",
        str(A_HIDDEN_SIZE),
        "--q_rep_dim",
        str(Q_REP_DIM),
        "--fused_dim",
        str(FUSED_DIM),
        "--mlp_hidden_dim",
        str(MLP_HIDDEN_DIM),
        "--attention_num_heads",
        str(ATTENTION_NUM_HEADS),
        "--attention_ff_dim",
        str(ATTENTION_FF_DIM),
        "--fuse_attention_num_heads",
        str(FUSE_ATTENTION_NUM_HEADS),
        "--fuse_attention_ff_dim",
        str(FUSE_ATTENTION_FF_DIM),
    ]

    file_name = selected.get("file_name")
    test_index = selected.get("test_index")

    if file_name:
        cmd.extend(["--file_name", str(Path(str(file_name)).name)])
    elif test_index is not None:
        cmd.extend(["--test_index", str(int(test_index))])
    else:
        raise ValueError(
            f"Wybrana symulacja nie ma ani file_name, ani test_index: {selected}"
        )

    append_bool_flag(cmd, "--gcn_bool", GCN_BOOL)
    append_bool_flag(cmd, "--aptonly", APTONLY)
    append_bool_flag(cmd, "--addaptadj", ADDAPTADJ)
    append_bool_flag(cmd, "--randomadj", RANDOMADJ)
    append_bool_flag(cmd, "--fuse_gated_update", FUSE_GATED_UPDATE)

    return cmd


def build_visual_command(selected: dict[str, Any]):
    detail_dir = Path(selected["detail_output_dir"])
    visual_dir = Path(selected["visual_output_dir"])
    relative_output_subdir = os.path.relpath(visual_dir, start=detail_dir)

    return [
        PYTHON_BIN,
        "-u",
        "inference_visual.py",
        "--run_dir",
        str(detail_dir),
        "--output_subdir",
        relative_output_subdir,
        "--top_k_nodes",
        str(TOP_K_NODES),
        "--num_best_nodes",
        str(NUM_BEST_NODES),
        "--num_middle_nodes",
        str(NUM_MIDDLE_NODES),
        "--num_worst_nodes",
        str(NUM_WORST_NODES),
    ]


# =============================================================================
# Global result parsing and selection
# =============================================================================


def _norm_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def _find_column(df: pd.DataFrame, aliases: list[str]) -> str | None:
    norm_to_original = {_norm_name(col): col for col in df.columns}
    for alias in aliases:
        col = norm_to_original.get(_norm_name(alias))
        if col is not None:
            return col
    return None


def _read_csv_safely(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(f"[WARN] Nie mogę odczytać CSV {path}: {exc}", flush=True)
        return None


def find_global_results_csv() -> Path:
    """
    Finds a CSV under GLOBAL_INFER_ROOT containing enough columns to select simulations.
    Prefer files whose names suggest statistics/summary/metrics.
    """
    csv_files = list(GLOBAL_INFER_ROOT.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"Nie znaleziono plików CSV w: {GLOBAL_INFER_ROOT}")

    name_priority = ["global", "summary", "stat", "metric", "result", "eval"]

    def score(path: Path) -> tuple[int, float]:
        lower = path.name.lower()
        hit_score = sum(1 for token in name_priority if token in lower)
        return (hit_score, path.stat().st_mtime)

    candidates = sorted(csv_files, key=score, reverse=True)

    for path in candidates:
        df = _read_csv_safely(path)
        if df is None or df.empty:
            continue

        real_col = _find_column(df, REAL_TOTAL_TT_ALIASES)
        pred_col = _find_column(df, PRED_TOTAL_TT_ALIASES)
        err_col = _find_column(df, ABS_PCT_ERROR_ALIASES)

        if real_col is not None and (pred_col is not None or err_col is not None):
            print(f"[INFO] Wybrano CSV ze statystykami globalnymi: {path}", flush=True)
            return path

    raise ValueError(
        "Znaleziono CSV w globalnej inferencji, ale żaden nie ma wymaganych kolumn. "
        "Potrzebuję real total TT oraz pred total TT albo % błędu total TT."
    )


FILE_NAME_ALIASES = [
    "file_name",
    "filename",
    "file",
    "q_file",
    "q_filename",
    "flow_file",
    "assignment_file",
    "a_file",
    "simulation",
    "simulation_file",
    "sim_file",
    "case_id",
]

TEST_INDEX_ALIASES = [
    "test_index",
    "test_idx",
    "index",
    "idx",
    "file_index",
    "simulation_index",
    "sim_index",
    "global_index",
]

REAL_TOTAL_TT_ALIASES = [
    "real_total_tt",
    "actual_total_tt",
    "true_total_tt",
    "target_total_tt",
    "y_total_tt",
    "real_tt_total",
    "total_tt_real",
    "total_real_tt",
    "real_total_travel_time",
    "actual_total_travel_time",
    "true_total_travel_time",
    "target_total_travel_time",
]

PRED_TOTAL_TT_ALIASES = [
    "pred_total_tt",
    "predicted_total_tt",
    "prediction_total_tt",
    "yhat_total_tt",
    "forecast_total_tt",
    "pred_tt_total",
    "total_tt_pred",
    "total_pred_tt",
    "pred_total_travel_time",
    "predicted_total_travel_time",
]

ABS_PCT_ERROR_ALIASES = [
    "abs_pct_total_tt_error",
    "pct_total_tt_error_abs",
    "total_tt_abs_pct_error",
    "abs_percentage_error_total_tt",
    "absolute_percentage_error_total_tt",
    "abs_pct_error",
    "pct_abs_error",
    "pct_error_abs",
    "percentage_error_abs",
    "absolute_percentage_error",
    "pct_error",
    "percentage_error",
    "relative_error_pct",
    "total_tt_pct_error",
    "total_tt_percentage_error",
]


def prepare_global_stats_df(csv_path: Path) -> tuple[pd.DataFrame, dict[str, str | None]]:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"CSV ze statystykami jest pusty: {csv_path}")

    file_col = _find_column(df, FILE_NAME_ALIASES)
    index_col = _find_column(df, TEST_INDEX_ALIASES)
    real_col = _find_column(df, REAL_TOTAL_TT_ALIASES)
    pred_col = _find_column(df, PRED_TOTAL_TT_ALIASES)
    err_col = _find_column(df, ABS_PCT_ERROR_ALIASES)

    if real_col is None:
        raise ValueError(
            f"Nie znaleziono kolumny real total TT w {csv_path}. "
            f"Dostępne kolumny: {list(df.columns)}"
        )

    if pred_col is None and err_col is None:
        raise ValueError(
            f"Nie znaleziono ani pred total TT, ani % error w {csv_path}. "
            f"Dostępne kolumny: {list(df.columns)}"
        )

    out = df.copy()
    out["__row_position"] = range(len(out))
    out["__real_total_tt"] = pd.to_numeric(out[real_col], errors="coerce")

    if pred_col is not None:
        out["__pred_total_tt"] = pd.to_numeric(out[pred_col], errors="coerce")
    else:
        out["__pred_total_tt"] = pd.NA

    if err_col is not None:
        out["__abs_pct_total_tt_error"] = pd.to_numeric(out[err_col], errors="coerce").abs()
    else:
        denom = out["__real_total_tt"].abs().replace(0, pd.NA)
        out["__abs_pct_total_tt_error"] = (
            (out["__pred_total_tt"] - out["__real_total_tt"]).abs() / denom * 100.0
        )

    if file_col is not None:
        out["__file_name"] = out[file_col].astype(str)
    else:
        out["__file_name"] = pd.NA

    if index_col is not None:
        out["__test_index"] = pd.to_numeric(out[index_col], errors="coerce")
    else:
        # Fallback: if global inference processes the test split in order, this maps to test_index.
        out["__test_index"] = GLOBAL_START_INDEX + out["__row_position"]

    out = out.dropna(subset=["__real_total_tt", "__abs_pct_total_tt_error"])

    columns = {
        "file_col": file_col,
        "index_col": index_col,
        "real_col": real_col,
        "pred_col": pred_col,
        "err_col": err_col,
    }
    print(f"[INFO] Kolumny globalnych statystyk: {columns}", flush=True)
    return out, columns


def _stable_selection_key(row: pd.Series) -> str:
    file_name = row.get("__file_name")
    if pd.notna(file_name) and str(file_name).strip() and str(file_name).lower() != "nan":
        return f"file::{Path(str(file_name)).name}"
    return f"test_index::{int(row['__test_index'])}"


def _slug(text: str, max_len: int = 90) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")
    return text[:max_len] if len(text) > max_len else text


def select_simulations_from_global_stats() -> list[dict[str, Any]]:
    csv_path = find_global_results_csv()
    df, used_columns = prepare_global_stats_df(csv_path)

    if df.empty:
        raise ValueError("Po przetworzeniu globalnych statystyk nie zostały żadne poprawne wiersze.")

    selections: dict[str, dict[str, Any]] = {}

    categories = [
        ("lowest_pct_error", df.sort_values("__abs_pct_total_tt_error", ascending=True)),
        ("highest_pct_error", df.sort_values("__abs_pct_total_tt_error", ascending=False)),
        ("lowest_real_tt", df.sort_values("__real_total_tt", ascending=True)),
        ("highest_real_tt", df.sort_values("__real_total_tt", ascending=False)),
    ]

    for category, sorted_df in categories:
        for _, row in sorted_df.head(NUM_SELECT_EACH).iterrows():
            key = _stable_selection_key(row)

            if key not in selections:
                file_name = row.get("__file_name")
                if pd.isna(file_name) or str(file_name).lower() == "nan":
                    file_name = None
                else:
                    file_name = str(file_name)

                test_index = row.get("__test_index")
                if pd.isna(test_index):
                    test_index = None
                else:
                    test_index = int(test_index)

                base_label = Path(file_name).stem if file_name else f"test_index_{test_index}"
                selection_id = _slug(f"{len(selections) + 1:02d}_{category}_{base_label}")

                selections[key] = {
                    "selection_id": selection_id,
                    "categories": [],
                    "file_name": file_name,
                    "test_index": test_index,
                    "real_total_tt": float(row["__real_total_tt"]),
                    "pred_total_tt": None
                    if pd.isna(row["__pred_total_tt"])
                    else float(row["__pred_total_tt"]),
                    "abs_pct_total_tt_error": float(row["__abs_pct_total_tt_error"]),
                    "global_stats_csv": str(csv_path),
                    "used_columns": used_columns,
                }

            selections[key]["categories"].append(category)

    selected = list(selections.values())

    for item in selected:
        item["categories"] = sorted(set(item["categories"]))
        item["detail_output_dir"] = str(DETAIL_INFER_ROOT / item["selection_id"])
        item["visual_output_dir"] = str(VIS_DIR / item["selection_id"])

    selected_df = pd.DataFrame(selected)
    SELECTED_SIMULATIONS_CSV.parent.mkdir(parents=True, exist_ok=True)
    selected_df.to_csv(SELECTED_SIMULATIONS_CSV, index=False)

    with open(SELECTED_SIMULATIONS_JSON, "w", encoding="utf-8") as f:
        json.dump(selected, f, indent=2, ensure_ascii=False)

    print(f"[INFO] Wybrano {len(selected)} symulacji do szczegółowej inferencji", flush=True)
    print(f"[INFO] Zapisano: {SELECTED_SIMULATIONS_CSV}", flush=True)
    print(f"[INFO] Zapisano: {SELECTED_SIMULATIONS_JSON}", flush=True)

    for item in selected:
        print(
            "[SELECTED] "
            f"{item['selection_id']} | "
            f"categories={','.join(item['categories'])} | "
            f"file={item['file_name']} | "
            f"test_index={item['test_index']} | "
            f"real_total_tt={item['real_total_tt']:.4f} | "
            f"pred_total_tt={item['pred_total_tt']} | "
            f"abs_pct_error={item['abs_pct_total_tt_error']:.4f}%",
            flush=True,
        )

    return selected


# =============================================================================
# Summary collection
# =============================================================================


def collect_key_results():
    training_summary_dir = RESULTS_DIR / "training"
    inference_summary_dir = RESULTS_DIR / "inference"
    global_summary_dir = inference_summary_dir / "global"
    selected_summary_dir = inference_summary_dir / "selected_detailed"
    visual_summary_dir = RESULTS_DIR / "visual_selected"
    logs_summary_dir = RESULTS_DIR / "logs"

    training_summary_dir.mkdir(parents=True, exist_ok=True)
    global_summary_dir.mkdir(parents=True, exist_ok=True)
    selected_summary_dir.mkdir(parents=True, exist_ok=True)
    logs_summary_dir.mkdir(parents=True, exist_ok=True)

    # Training artifacts. Keep both old and newer possible locations.
    for p in RUN_DIR.rglob("learning_curves.png"):
        copy_if_exists(p, training_summary_dir)
    for p in RUN_DIR.rglob("training_metrics.csv"):
        copy_if_exists(p, training_summary_dir)

    # Global inference: copy lightweight stats, not necessarily every heavy array.
    copy_matching_files(GLOBAL_INFER_ROOT, global_summary_dir, ["*.csv", "*.json", "*.txt"])

    # Selected simulations table.
    copy_if_exists(SELECTED_SIMULATIONS_CSV, inference_summary_dir)
    copy_if_exists(SELECTED_SIMULATIONS_JSON, inference_summary_dir)

    # Detailed selected inference and visual outputs.
    copy_tree_if_exists(DETAIL_INFER_ROOT, selected_summary_dir)
    copy_tree_if_exists(VIS_DIR, visual_summary_dir)

    # Logs.
    for p in LOGS_DIR.glob("*.log"):
        copy_if_exists(p, logs_summary_dir)

    # Checkpoint + config.
    copy_if_exists(RUN_DIR / "checkpoint_best.pth", RESULTS_DIR)
    copy_if_exists(RUN_DIR / "pipeline_config.json", RESULTS_DIR)


# =============================================================================
# Main
# =============================================================================


def main():
    print_header("START PIPELINE: TRAIN -> GLOBAL GPU INFERENCE -> SELECTED DETAILED INFERENCE")

    ensure_dir(PIPELINE_ROOT)
    ensure_dir(RUN_DIR)
    ensure_dir(TRAIN_PREFIX.parent)
    ensure_dir(INFERENCE_ROOT)
    ensure_dir(GLOBAL_INFER_ROOT)
    ensure_dir(DETAIL_INFER_ROOT)
    ensure_dir(VIS_DIR)
    ensure_dir(RESULTS_DIR)
    ensure_dir(LOGS_DIR)

    validate_input_dirs()
    save_pipeline_config(best_checkpoint=None, selected_count=None)

    print_header("ENVIRONMENT")
    run_command([PYTHON_BIN, "-V"], cwd=PROJECT_DIR)
    run_command([PYTHON_BIN, "-c", "import sys; print(sys.executable)"], cwd=PROJECT_DIR)
    run_command([PYTHON_BIN, "-c", "import torch; print(torch.__version__)"], cwd=PROJECT_DIR)
    run_command(["nvidia-smi"], cwd=PROJECT_DIR, check=False)

    print_header("RUN DIRECTORY")
    print(f"RUN_DIR            = {RUN_DIR}", flush=True)
    print(f"TRAIN_Q_DIR        = {TRAIN_Q_DIR}", flush=True)
    print(f"TRAIN_A_DIR        = {TRAIN_A_DIR}", flush=True)
    print(f"INFER_Q_DIR        = {INFER_Q_DIR}", flush=True)
    print(f"INFER_A_DIR        = {INFER_A_DIR}", flush=True)
    print(f"TRAIN_PREFIX       = {TRAIN_PREFIX}", flush=True)
    print(f"INFERENCE_ROOT     = {INFERENCE_ROOT}", flush=True)
    print(f"GLOBAL_INFER_ROOT  = {GLOBAL_INFER_ROOT}", flush=True)
    print(f"DETAIL_INFER_ROOT  = {DETAIL_INFER_ROOT}", flush=True)
    print(f"VIS_DIR            = {VIS_DIR}", flush=True)
    print(f"LOGS_DIR           = {LOGS_DIR}", flush=True)

    # 1. Training.
    print_header("STAGE 1: TRAINING")
    train_cmd = build_train_command()
    wandb_env = build_wandb_env()
    run_command(train_cmd, cwd=PROJECT_DIR, log_file=LOGS_DIR / "train.log", env=wandb_env)

    # 2. Best checkpoint from this run.
    print_header("STAGE 2: BEST CHECKPOINT")
    best_checkpoint = find_best_checkpoint(RUN_DIR, expid=EXPID)
    print(f"[INFO] Best checkpoint: {best_checkpoint}", flush=True)
    copy_best_checkpoint_to_run_root(best_checkpoint)
    save_pipeline_config(best_checkpoint=best_checkpoint, selected_count=None)

    # 3. Global inference on GPU.
    print_header("STAGE 3: GLOBAL GPU INFERENCE")
    validate_project_entrypoint(GLOBAL_INFER_SCRIPT)
    global_cmd = build_global_inference_command(best_checkpoint)
    run_command(global_cmd, cwd=PROJECT_DIR, log_file=LOGS_DIR / "global_inference.log")

    # 4. Select simulations by global metrics.
    print_header("STAGE 4: SELECT SIMULATIONS FROM GLOBAL STATS")
    selected_simulations = select_simulations_from_global_stats()
    save_pipeline_config(best_checkpoint=best_checkpoint, selected_count=len(selected_simulations))

    # 5. Detailed inference + visualization for each selected simulation.
    print_header("STAGE 5: DETAILED GPU INFERENCE + VISUALIZATION FOR SELECTED SIMULATIONS")
    for selected in selected_simulations:
        selection_id = selected["selection_id"]
        print_header(f"DETAILED INFERENCE: {selection_id}")
        detail_cmd = build_detailed_inference_command(best_checkpoint, selected)
        run_command(
            detail_cmd,
            cwd=PROJECT_DIR,
            log_file=LOGS_DIR / f"detailed_inference_{selection_id}.log",
        )

        print_header(f"VISUALIZATION: {selection_id}")
        vis_cmd = build_visual_command(selected)
        run_command(
            vis_cmd,
            cwd=PROJECT_DIR,
            log_file=LOGS_DIR / f"visual_{selection_id}.log",
        )

    # 6. Summary.
    print_header("STAGE 6: COLLECT KEY RESULTS")
    collect_key_results()

    print_header("PIPELINE FINISHED")
    print(f"[OK] Whole run:              {RUN_DIR}", flush=True)
    print(f"[OK] Training:               {TRAIN_PREFIX}", flush=True)
    print(f"[OK] Global inference:       {GLOBAL_INFER_ROOT}", flush=True)
    print(f"[OK] Selected detailed infer: {DETAIL_INFER_ROOT}", flush=True)
    print(f"[OK] Selected visual:         {VIS_DIR}", flush=True)
    print(f"[OK] Best checkpoint:         {RUN_DIR / 'checkpoint_best.pth'}", flush=True)
    print(f"[OK] Config:                  {RUN_DIR / 'pipeline_config.json'}", flush=True)
    print(f"[OK] Summary:                 {RESULTS_DIR}", flush=True)


if __name__ == "__main__":
    main()

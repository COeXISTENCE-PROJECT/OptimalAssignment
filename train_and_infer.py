#!/usr/bin/env python3
import json
import re
import shlex
import shutil
import subprocess
from datetime import datetime
from pathlib import Path


PYTHON_BIN = "/home/drozd/miniconda/envs/wavenet_env/bin/python"
PROJECT_DIR = Path("/home/drozd/OptimalAssignment").resolve()

# ===== dane =====
Q_DIR = "/scratch/tmp/new_flows_10s"
A_DIR = "/scratch/tmp/new_assignments_10s"
DATA_ROOT = "/scratch/tmp"
ADJDATA = str(PROJECT_DIR / "new_hex_adjacency_matrix.csv")

# ===== główny katalog pipeline =====
PIPELINE_ROOT = Path("/scratch/tmp/ADTTP_tests_new")
RUN_STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_NAME = f"run_{RUN_STAMP}"
RUN_DIR = PIPELINE_ROOT / RUN_NAME

# ===== podfoldery jednego runu =====
TRAIN_PREFIX = RUN_DIR / "training"
INFER_DIR = RUN_DIR / "inference"
VIS_DIR = RUN_DIR / "visual"

# ===== sprzęt / split =====
DEVICE = "cuda:0"
SEED = 42
TRAIN_RATIO = 0.7
VAL_RATIO = 0.1

# ===== sekwencje =====
SEQ_LENGTH_Q = 15
SEQ_LENGTH_A = 30
SEQ_LENGTH_Y = 1

# ===== graf =====
ADJTYPE = "doubletransition"
GCN_BOOL = True
APTONLY = False
ADDAPTADJ = False
RANDOMADJ = False

# ===== model =====
IN_DIM = 1
NUM_NODES = 195
NHID = 64
DROPOUT = 0.1
KERNEL_SIZE = 2
BLOCKS = 4
LAYERS = 2

TARGET_DIM = 1
SEQUENCE_MODEL = "lstm"      # "lstm" / "gru"
FUSE_METHOD = "Attention"    # "Attention" / "Concat" / "MLP"

A_EMBEDDING_SIZE = 64
A_HIDDEN_SIZE = 64
Q_REP_DIM = 64
FUSED_DIM = 64
MLP_HIDDEN_DIM = 128
ATTENTION_NUM_HEADS = 4
ATTENTION_FF_DIM = 64

# ===== trening =====
BATCH_SIZE = 32
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
EPOCHS = 1
PRINT_EVERY = 1000
NUM_WORKERS = 4
EXPID = 1

# ===== inferencja =====
USE_TEST_INDEX = True
TEST_INDEX = 1

# Jeśli USE_TEST_INDEX = False, ustaw FILE_NAME
FILE_NAME = None
# np. FILE_NAME = "sample.npy"

# ===== wizualizacja =====
TOP_K_NODES = 20
NUM_BEST_NODES = 5
NUM_MIDDLE_NODES = 5
NUM_WORST_NODES = 5


#tools

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def print_header(title: str):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def run_command(cmd, cwd=None):
    printable = " ".join(shlex.quote(str(x)) for x in cmd)
    print(f"\n[RUN] {printable}\n")
    subprocess.run(cmd, cwd=cwd, check=True)


def append_bool_flag(cmd: list, flag_name: str, value: bool):
    if value:
        cmd.append(flag_name)


def find_best_checkpoint(train_dir: Path, expid: int | None = None) -> Path:
    """
    Szuka checkpointu typu:
    ..._exp1_best_0.4119.pth
    Jeśli nie znajdzie takiego wzorca, bierze najnowszy .pth.
    """
    candidates = sorted(train_dir.glob("*.pth"))
    if not candidates:
        raise FileNotFoundError(f"Nie znaleziono checkpointów .pth w: {train_dir}")

    parsed = []
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
        parsed.sort(key=lambda x: x[0])  # mniejszy score = lepszy
        return parsed[0][1]

    newest = max(candidates, key=lambda p: p.stat().st_mtime)
    print(f"[WARN] Nie udało się sparsować checkpointu po wzorcu *_best_*.pth")
    print(f"[WARN] Biorę najnowszy plik: {newest}")
    return newest


def save_pipeline_config(best_checkpoint: Path | None = None):
    config = {
        "run_name": RUN_NAME,
        "run_dir": str(RUN_DIR),
        "training_dir": str(TRAIN_PREFIX),
        "inference_dir": str(INFER_DIR),
        "visual_dir": str(VIS_DIR),
        "python_bin": PYTHON_BIN,
        "project_dir": str(PROJECT_DIR),
        "data": {
            "q_dir": Q_DIR,
            "a_dir": A_DIR,
            "data_root": DATA_ROOT,
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
        },
        "training": {
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "epochs": EPOCHS,
            "print_every": PRINT_EVERY,
            "num_workers": NUM_WORKERS,
            "expid": EXPID,
        },
        "inference": {
            "use_test_index": USE_TEST_INDEX,
            "test_index": TEST_INDEX if USE_TEST_INDEX else None,
            "file_name": FILE_NAME if not USE_TEST_INDEX else None,
        },
        "visualization": {
            "top_k_nodes": TOP_K_NODES,
            "num_best_nodes": NUM_BEST_NODES,
            "num_middle_nodes": NUM_MIDDLE_NODES,
            "num_worst_nodes": NUM_WORST_NODES,
        },
        "best_checkpoint": None if best_checkpoint is None else str(best_checkpoint),
    }

    with open(RUN_DIR / "pipeline_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def copy_best_checkpoint_to_run_root(best_checkpoint: Path):
    target = RUN_DIR / "checkpoint_best.pth"
    shutil.copy2(best_checkpoint, target)
    print(f"[INFO] Skopiowano najlepszy checkpoint do: {target}")


def build_train_command():
    cmd = [
        PYTHON_BIN, "train.py",
        "--device", DEVICE,
        "--data", DATA_ROOT,
        "--q_dir", Q_DIR,
        "--a_dir", A_DIR,
        "--seed", str(SEED),
        "--train_ratio", str(TRAIN_RATIO),
        "--val_ratio", str(VAL_RATIO),
        "--seq_length_q", str(SEQ_LENGTH_Q),
        "--seq_length_a", str(SEQ_LENGTH_A),
        "--seq_length_y", str(SEQ_LENGTH_Y),
        "--adjdata", ADJDATA,
        "--adjtype", ADJTYPE,
        "--in_dim", str(IN_DIM),
        "--num_nodes", str(NUM_NODES),
        "--nhid", str(NHID),
        "--batch_size", str(BATCH_SIZE),
        "--learning_rate", str(LEARNING_RATE),
        "--dropout", str(DROPOUT),
        "--weight_decay", str(WEIGHT_DECAY),
        "--epochs", str(EPOCHS),
        "--print_every", str(PRINT_EVERY),
        "--save", str(TRAIN_PREFIX),
        "--expid", str(EXPID),
        "--kernel_size", str(KERNEL_SIZE),
        "--blocks", str(BLOCKS),
        "--layers", str(LAYERS),
        "--num_workers", str(NUM_WORKERS),
        "--target_dim", str(TARGET_DIM),
        "--sequence_model", str(SEQUENCE_MODEL),
        "--fuse_method", str(FUSE_METHOD),
        "--a_embedding_size", str(A_EMBEDDING_SIZE),
        "--a_hidden_size", str(A_HIDDEN_SIZE),
        "--q_rep_dim", str(Q_REP_DIM),
        "--fused_dim", str(FUSED_DIM),
        "--mlp_hidden_dim", str(MLP_HIDDEN_DIM),
        "--attention_num_heads", str(ATTENTION_NUM_HEADS),
        "--attention_ff_dim", str(ATTENTION_FF_DIM),
    ]

    append_bool_flag(cmd, "--gcn_bool", GCN_BOOL)
    append_bool_flag(cmd, "--aptonly", APTONLY)
    append_bool_flag(cmd, "--addaptadj", ADDAPTADJ)
    append_bool_flag(cmd, "--randomadj", RANDOMADJ)

    return cmd


def build_inference_command(checkpoint_path: Path):
    cmd = [
        PYTHON_BIN, "inference.py",
        "--q_dir", Q_DIR,
        "--a_dir", A_DIR,
        "--adjdata", ADJDATA,
        "--checkpoint", str(checkpoint_path),
        "--output_dir", str(INFER_DIR),
        "--seed", str(SEED),
        "--train_ratio", str(TRAIN_RATIO),
        "--val_ratio", str(VAL_RATIO),
        "--seq_length_q", str(SEQ_LENGTH_Q),
        "--seq_length_a", str(SEQ_LENGTH_A),
        "--device", DEVICE,
        "--num_nodes", str(NUM_NODES),
        "--nhid", str(NHID),
        "--dropout", str(DROPOUT),
        "--learning_rate", str(LEARNING_RATE),
        "--weight_decay", str(WEIGHT_DECAY),
        "--kernel_size", str(KERNEL_SIZE),
        "--blocks", str(BLOCKS),
        "--layers", str(LAYERS),
        "--sequence_model", str(SEQUENCE_MODEL),
        "--fuse_method", str(FUSE_METHOD),
        "--a_embedding_size", str(A_EMBEDDING_SIZE),
        "--a_hidden_size", str(A_HIDDEN_SIZE),
        "--q_rep_dim", str(Q_REP_DIM),
        "--fused_dim", str(FUSED_DIM),
        "--mlp_hidden_dim", str(MLP_HIDDEN_DIM),
        "--attention_num_heads", str(ATTENTION_NUM_HEADS),
        "--attention_ff_dim", str(ATTENTION_FF_DIM),
    ]

    if USE_TEST_INDEX:
        cmd.extend(["--test_index", str(TEST_INDEX)])
    else:
        if not FILE_NAME:
            raise ValueError("Jeśli USE_TEST_INDEX=False, musisz ustawić FILE_NAME")
        cmd.extend(["--file_name", str(FILE_NAME)])

    append_bool_flag(cmd, "--gcn_bool", GCN_BOOL)
    append_bool_flag(cmd, "--aptonly", APTONLY)
    append_bool_flag(cmd, "--addaptadj", ADDAPTADJ)
    append_bool_flag(cmd, "--randomadj", RANDOMADJ)

    return cmd


def build_visual_command():
    """
    inference_visual.py zapisuje do:
        out_dir = run_dir / output_subdir

    Ponieważ run_dir = INFER_DIR, ustawiamy:
        output_subdir = "../visual"

    dzięki czemu wyniki trafią do:
        RUN_DIR / visual
    """
    return [
        PYTHON_BIN, "inference_visual.py",
        "--run_dir", str(INFER_DIR),
        "--output_subdir", "../visual",
        "--top_k_nodes", str(TOP_K_NODES),
        "--num_best_nodes", str(NUM_BEST_NODES),
        "--num_middle_nodes", str(NUM_MIDDLE_NODES),
        "--num_worst_nodes", str(NUM_WORST_NODES),
    ]



def main():
    print_header("START PIPELINE: TRAIN -> INFERENCE -> VISUAL")

    ensure_dir(PIPELINE_ROOT)
    ensure_dir(RUN_DIR)
    ensure_dir(INFER_DIR)
    ensure_dir(VIS_DIR)

    save_pipeline_config(best_checkpoint=None)

    print_header("ŚRODOWISKO")
    run_command([PYTHON_BIN, "-V"], cwd=PROJECT_DIR)
    run_command([PYTHON_BIN, "-c", "import sys; print(sys.executable)"], cwd=PROJECT_DIR)
    run_command([PYTHON_BIN, "-c", "import torch; print(torch.__version__)"], cwd=PROJECT_DIR)

    print_header("RUN DIRECTORY")
    print(f"RUN_DIR   = {RUN_DIR}")
    print(f"TRAIN_DIR = {TRAIN_PREFIX}")
    print(f"INFER_DIR = {INFER_DIR}")
    print(f"VIS_DIR   = {VIS_DIR}")

    # 1. trening
    print_header("ETAP 1: TRENING")
    train_cmd = build_train_command()
    run_command(train_cmd, cwd=PROJECT_DIR)

    # 2. najlepszy checkpoint
    print_header("ETAP 2: WYBÓR CHECKPOINTU")
    best_checkpoint = find_best_checkpoint(RUN_DIR, expid=EXPID)
    print(f"[INFO] Best checkpoint: {best_checkpoint}")

    copy_best_checkpoint_to_run_root(best_checkpoint)
    save_pipeline_config(best_checkpoint=best_checkpoint)

    # 3. inferencja
    print_header("ETAP 3: INFERENCJA")
    infer_cmd = build_inference_command(best_checkpoint)
    run_command(infer_cmd, cwd=PROJECT_DIR)

    # 4. wizualizacja
    print_header("ETAP 4: WIZUALIZACJA")
    vis_cmd = build_visual_command()
    run_command(vis_cmd, cwd=PROJECT_DIR)

    print_header("PIPELINE ZAKOŃCZONY")
    print(f"[OK] Cały run zapisany w: {RUN_DIR}")
    print(f"[OK] Trening:             {TRAIN_PREFIX}")
    print(f"[OK] Inferencja:          {INFER_DIR}")
    print(f"[OK] Wizualizacje:        {VIS_DIR}")
    print(f"[OK] Best checkpoint:     {RUN_DIR / 'checkpoint_best.pth'}")
    print(f"[OK] Konfiguracja runu:   {RUN_DIR / 'pipeline_config.json'}")


if __name__ == "__main__":
    main()

import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# =========================
# KONFIGURACJA TRENINGU
# =========================

@dataclass
class TrainingConfig:
    # co trenujemy
    dataset_name: str

    # ścieżki
    base_dir: Path
    processed_root: Path
    garage_root: Path

    # środowisko / uruchomienie
    device: str = "cpu"
    python_executable: str = sys.executable
    train_script_name: str = "train.py"

    # hiperparametry
    epochs: int = 2
    batch_size: int = 64
    learning_rate: float = 0.001
    nhid: int = 32
    kernel_size: int = 2
    blocks: int = 4
    layers: int = 2
    print_every: int = 10

    # opcje modelu
    use_gcn: bool = True
    addaptadj: bool = False

    # organizacja eksperymentu
    run_name: Optional[str] = None

    def resolved_base_dir(self) -> Path:
        return self.base_dir.resolve()

    def resolved_processed_root(self) -> Path:
        return self.processed_root.resolve()

    def resolved_garage_root(self) -> Path:
        return self.garage_root.resolve()

    def train_script_path(self) -> Path:
        return self.resolved_base_dir() / self.train_script_name

    def dataset_dir(self) -> Path:
        return self.resolved_processed_root() / self.dataset_name


# =========================
# NARZĘDZIA
# =========================

def load_dataset_metadata(dataset_dir: Path) -> dict:
    """
    Read dataset dimensions directly from train.npz.
    Expected x shape: (samples, seq_len, num_nodes, in_dim)
    """
    train_path = dataset_dir / "train.npz"
    adj_path = dataset_dir / "adjacency_matrices" / "adjacency_spatial.csv"

    if not train_path.exists():
        raise FileNotFoundError(f"Missing file: {train_path}")

    if not adj_path.exists():
        raise FileNotFoundError(f"Missing file: {adj_path}")

    with np.load(train_path) as data:
        x = data["x"]
        if x.ndim != 4:
            raise ValueError(
                f"Unexpected x shape in {train_path}: {x.shape}. "
                "Expected 4 dimensions: (samples, seq_len, num_nodes, in_dim)."
            )

        seq_length = int(x.shape[1])
        num_nodes = int(x.shape[2])
        in_dim = int(x.shape[3])

    return {
        "seq_length": seq_length,
        "num_nodes": num_nodes,
        "in_dim": in_dim,
        "adj_path": adj_path.resolve(),
        "data_dir": dataset_dir.resolve(),
    }


def validate_config(config: TrainingConfig) -> None:
    if not config.dataset_name:
        raise ValueError("dataset_name cannot be empty.")

    if config.epochs <= 0:
        raise ValueError("epochs must be > 0.")

    if config.batch_size <= 0:
        raise ValueError("batch_size must be > 0.")

    if config.learning_rate <= 0:
        raise ValueError("learning_rate must be > 0.")

    if config.nhid <= 0:
        raise ValueError("nhid must be > 0.")

    if config.kernel_size <= 0:
        raise ValueError("kernel_size must be > 0.")

    if config.blocks <= 0:
        raise ValueError("blocks must be > 0.")

    if config.layers <= 0:
        raise ValueError("layers must be > 0.")

    train_script = config.train_script_path()
    if not train_script.exists():
        raise FileNotFoundError(f"train.py not found: {train_script}")

    dataset_dir = config.dataset_dir()
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")


def build_training_command(config: TrainingConfig, metadata: dict, model_save_prefix: Path) -> list[str]:
    command = [
        config.python_executable,
        str(config.train_script_path()),
        "--device", str(config.device),
        "--data", str(metadata["data_dir"]),
        "--adjdata", str(metadata["adj_path"]),
        "--adjtype", "doubletransition",
        "--num_nodes", str(metadata["num_nodes"]),
        "--in_dim", str(metadata["in_dim"]),
        "--seq_length", str(metadata["seq_length"]),
        "--nhid", str(config.nhid),
        "--epochs", str(config.epochs),
        "--print_every", str(config.print_every),
        "--batch_size", str(config.batch_size),
        "--learning_rate", str(config.learning_rate),
        "--save", str(model_save_prefix),
        "--kernel_size", str(config.kernel_size),
        "--blocks", str(config.blocks),
        "--layers", str(config.layers),
    ]

    if config.use_gcn:
        command.append("--gcn_bool")

    if config.addaptadj:
        command.append("--addaptadj")

    return command


def _make_json_safe_dict(config: TrainingConfig) -> dict:
    raw = asdict(config)
    safe = {}
    for key, value in raw.items():
        if isinstance(value, Path):
            safe[key] = str(value)
        else:
            safe[key] = value
    return safe


def run_training(config: TrainingConfig) -> Path:
    """
    Main entrypoint for training exactly one dataset.
    Returns path to the created experiment directory.
    """
    validate_config(config)

    metadata = load_dataset_metadata(config.dataset_dir())

    job_id = os.environ.get("SLURM_JOB_ID", datetime.now().strftime("%Y%m%d_%H%M%S"))
    run_suffix = config.run_name if config.run_name else f"experiment_{job_id}"

    save_dir = config.resolved_garage_root() / config.dataset_name / run_suffix
    save_dir.mkdir(parents=True, exist_ok=True)

    model_save_prefix = save_dir / "model"
    command = build_training_command(config, metadata, model_save_prefix)

    # zapis konfiguracji i komendy - bardzo przydatne na chmurze
    with open(save_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(_make_json_safe_dict(config), f, indent=2, ensure_ascii=False)

    with open(save_dir / "resolved_dataset_metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "seq_length": metadata["seq_length"],
                "num_nodes": metadata["num_nodes"],
                "in_dim": metadata["in_dim"],
                "adj_path": str(metadata["adj_path"]),
                "data_dir": str(metadata["data_dir"]),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    with open(save_dir / "command.sh", "w", encoding="utf-8") as f:
        f.write(" ".join(map(str, command)) + "\n")

    print("=" * 100)
    print(f"Starting training for dataset: {config.dataset_name}")
    print(f"Data directory: {metadata['data_dir']}")
    print(f"Adjacency path: {metadata['adj_path']}")
    print(
        f"num_nodes={metadata['num_nodes']}, "
        f"in_dim={metadata['in_dim']}, "
        f"seq_length={metadata['seq_length']}"
    )
    print(f"Save directory: {save_dir}")
    print("=" * 100)

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(config.resolved_base_dir()),
        bufsize=1,
    )

    if process.stdout is None:
        raise RuntimeError("Failed to capture training process stdout.")

    log_file = save_dir / "train.log"
    with open(log_file, "w", encoding="utf-8") as log:
        for line in process.stdout:
            print(line, end="")
            log.write(line)

    process.wait()

    if process.returncode != 0:
        raise RuntimeError(
            f"Training failed for dataset {config.dataset_name}. "
            f"Exit code: {process.returncode}. "
            f"See log: {log_file}"
        )

    print(f"\nTraining completed successfully for dataset: {config.dataset_name}")
    print(f"Logs and artifacts saved in: {save_dir}")
    return save_dir


# =========================
# ANALIZA CSV
# =========================

def analyze_training_csv(csv_path: str | Path) -> None:
    """
    Analyze training_metrics.csv.
    """
    csv_path = Path(csv_path)

    if not csv_path.exists():
        print(f"Error: File {csv_path} does not exist.")
        sys.exit(1)

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        sys.exit(1)

    required_columns = [
        "epoch",
        "train_loss",
        "valid_loss",
        "valid_rmse",
        "valid_mape",
        "train_time",
        "val_time",
    ]

    if not all(col in df.columns for col in required_columns):
        print("Error: CSV file does not contain all required columns.")
        print(f"Required columns: {required_columns}")
        sys.exit(1)

    best_loss_idx = df["valid_loss"].idxmin()
    best_rmse_idx = df["valid_rmse"].idxmin()
    best_mape_idx = df["valid_mape"].idxmin()

    best_val_loss = df.loc[best_loss_idx, "valid_loss"]
    corresponding_train_loss = df.loc[best_loss_idx, "train_loss"]
    gen_gap = best_val_loss - corresponding_train_loss

    total_train_time = df["train_time"].sum()
    total_val_time = df["val_time"].sum()
    mean_epoch_time = df["train_time"].mean()

    print("=" * 60)
    print(f"training report: {csv_path.parent.name}")
    print("=" * 60)

    print("\n[1] time summary")
    print(f"  • Recorded epochs:       {len(df)}")
    print(f"  • Total training time:   {total_train_time / 60:.2f} min")
    print(f"  • Total validation time: {total_val_time / 60:.2f} min")
    print(f"  • Average epoch time:    {mean_epoch_time:.2f} s")

    print("\n[2] optimal training stopping points")
    print(f"  • Minimum Loss (MAE): {best_val_loss:.4f} (epoch {df.loc[best_loss_idx, 'epoch']})")
    print(f"  • Minimum RMSE:       {df.loc[best_rmse_idx, 'valid_rmse']:.4f} (epoch {df.loc[best_rmse_idx, 'epoch']})")
    print(f"  • Minimum MAPE:       {df.loc[best_mape_idx, 'valid_mape']:.4f} (epoch {df.loc[best_mape_idx, 'epoch']})")

    print(f"\n[3] overfitting analysis for epoch {df.loc[best_loss_idx, 'epoch']}")
    print(f"  • Train Loss (MAE):   {corresponding_train_loss:.4f}")
    print(f"  • Valid Loss (MAE):   {best_val_loss:.4f}")
    print(f"  • Generalization Gap: {gen_gap:.4f}")

    if corresponding_train_loss != 0 and gen_gap > (0.2 * corresponding_train_loss):
        print("    Significant difference between training and validation error (>20%).")
        print("    This may indicate early overfitting.")

    print("\n[4] last 5 epochs")
    recent_df = df.tail(5)[["epoch", "train_loss", "valid_loss", "valid_rmse"]].copy()
    print(recent_df.to_string(index=False))
    print("=" * 60)


# =========================
# PRZYKŁADOWE UŻYCIE
# =========================

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent

    config = TrainingConfig(
        dataset_name="ingolstadt_770",

        base_dir=BASE_DIR,
        processed_root=BASE_DIR / "data" / "processed_networks",
        garage_root=BASE_DIR / "garage",

        device="cpu",       # albo "cpu"
        epochs=1,
        batch_size=64,
        learning_rate=0.001,
        nhid=32,
        kernel_size=2,
        blocks=4,
        layers=2,
        print_every=10,
        use_gcn=True,
        addaptadj=False,

        # opcjonalnie własna nazwa runa:
        # run_name="test_run_01",
    )

    run_training(config)

    # przykład analizy po treningu:
    # analyze_training_csv(BASE_DIR / "garage" / "ingolstadt_770" / "experiment_xxx" / "training_metrics.csv")
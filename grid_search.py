import itertools
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


# =========================
# KONFIGURACJA GRID SEARCH
# =========================

@dataclass
class HyperparameterGrid:
    learning_rate: List[float] = field(default_factory=lambda: [1e-3])
    batch_size: List[int] = field(default_factory=lambda: [32])
    dropout: List[float] = field(default_factory=lambda: [0.2])
    weight_decay: List[float] = field(default_factory=lambda: [1e-4])
    nhid: List[int] = field(default_factory=lambda: [32])
    kernel_size: List[int] = field(default_factory=lambda: [2, 3])
    blocks: List[int] = field(default_factory=lambda: [3, 4])
    layers: List[int] = field(default_factory=lambda: [2, 3])
    gcn_bool: List[bool] = field(default_factory=lambda: [True])
    addaptadj: List[bool] = field(default_factory=lambda: [True, False])
    randomadj: List[bool] = field(default_factory=lambda: [False, True])


@dataclass
class GridSearchConfig:
    dataset_name: str

    # ścieżki
    base_dir: Path
    processed_root: Path
    garage_root: Path

    # środowisko / uruchomienie
    device: str = "cpu"
    python_executable: str = sys.executable
    train_script_name: str = "train.py"

    # parametry stałe
    epochs: int = 20
    print_every: int = 10
    adjtype: str = "doubletransition"
    aptonly: bool = False

    # organizacja eksperymentu
    run_name: Optional[str] = None

    # grid
    grid: HyperparameterGrid = field(default_factory=HyperparameterGrid)

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

def receptive_field(kernel_size: int, blocks: int, layers: int) -> int:
    """
    Approximate receptive field used in Graph WaveNet-like temporal stack:
    R = 1 + blocks * (kernel_size - 1) * (2^layers - 1)
    """
    return 1 + blocks * (kernel_size - 1) * (2 ** layers - 1)


def load_dataset_metadata(dataset_dir: Path) -> Dict[str, Any]:
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
                "Expected (samples, seq_len, num_nodes, in_dim)."
            )

        seq_length = int(x.shape[1])
        num_nodes = int(x.shape[2])
        in_dim = int(x.shape[3])

    return {
        "data_dir": dataset_dir.resolve(),
        "adj_path": adj_path.resolve(),
        "seq_length": seq_length,
        "num_nodes": num_nodes,
        "in_dim": in_dim,
    }


def validate_config(config: GridSearchConfig) -> None:
    if not config.dataset_name:
        raise ValueError("dataset_name cannot be empty.")

    if config.epochs <= 0:
        raise ValueError("epochs must be > 0.")

    if config.print_every <= 0:
        raise ValueError("print_every must be > 0.")

    train_script = config.train_script_path()
    if not train_script.exists():
        raise FileNotFoundError(f"train.py not found: {train_script}")

    dataset_dir = config.dataset_dir()
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")

    grid_dict = asdict(config.grid)
    for key, values in grid_dict.items():
        if not isinstance(values, list) or len(values) == 0:
            raise ValueError(f"Grid field '{key}' must be a non-empty list.")


def generate_grid(grid: HyperparameterGrid) -> List[Dict[str, Any]]:
    """
    Generate all hyperparameter combinations.
    """
    grid_dict = asdict(grid)
    keys = list(grid_dict.keys())
    values = [grid_dict[k] for k in keys]

    configs = []
    for combo in itertools.product(*values):
        cfg = dict(zip(keys, combo))

        # randomadj only makes sense if addaptadj=True
        if cfg["randomadj"] and not cfg["addaptadj"]:
            continue

        configs.append(cfg)

    return configs


def build_command(
    config: Dict[str, Any],
    base_dir: Path,
    train_script_path: Path,
    python_executable: str,
    save_prefix: Path,
) -> List[str]:
    """
    Build command line for train.py based on config.
    """
    command = [
        python_executable,
        str(train_script_path),
        "--device", str(config["device"]),
        "--data", str(config["data"]),
        "--adjdata", str(config["adjdata"]),
        "--adjtype", str(config["adjtype"]),
        "--num_nodes", str(config["num_nodes"]),
        "--in_dim", str(config["in_dim"]),
        "--seq_length", str(config["seq_length"]),
        "--nhid", str(config["nhid"]),
        "--epochs", str(config["epochs"]),
        "--print_every", str(config["print_every"]),
        "--batch_size", str(config["batch_size"]),
        "--learning_rate", str(config["learning_rate"]),
        "--dropout", str(config["dropout"]),
        "--weight_decay", str(config["weight_decay"]),
        "--save", str(save_prefix),
        "--kernel_size", str(config["kernel_size"]),
        "--blocks", str(config["blocks"]),
        "--layers", str(config["layers"]),
    ]

    if config.get("gcn_bool", False):
        command.append("--gcn_bool")
    if config.get("aptonly", False):
        command.append("--aptonly")
    if config.get("addaptadj", False):
        command.append("--addaptadj")
    if config.get("randomadj", False):
        command.append("--randomadj")

    return command


def summarize_training_csv(csv_path: Path) -> Optional[Dict[str, Any]]:
    """
    Read training_metrics.csv and extract best metrics.
    """
    if not csv_path.exists():
        return None

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return {"status": f"csv_read_error: {e}"}

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
        return {"status": "csv_missing_columns"}

    best_loss_idx = df["valid_loss"].idxmin()
    best_rmse_idx = df["valid_rmse"].idxmin()
    best_mape_idx = df["valid_mape"].idxmin()

    best_valid_loss = float(df.loc[best_loss_idx, "valid_loss"])
    best_valid_rmse = float(df.loc[best_rmse_idx, "valid_rmse"])
    best_valid_mape = float(df.loc[best_mape_idx, "valid_mape"])

    best_epoch_loss = int(df.loc[best_loss_idx, "epoch"])
    best_epoch_rmse = int(df.loc[best_rmse_idx, "epoch"])
    best_epoch_mape = int(df.loc[best_mape_idx, "epoch"])

    train_loss_at_best = float(df.loc[best_loss_idx, "train_loss"])
    generalization_gap = best_valid_loss - train_loss_at_best

    total_train_time = float(df["train_time"].sum())
    total_val_time = float(df["val_time"].sum())
    mean_epoch_time = float(df["train_time"].mean())

    return {
        "status": "ok",
        "best_valid_loss": best_valid_loss,
        "best_valid_rmse": best_valid_rmse,
        "best_valid_mape": best_valid_mape,
        "best_epoch_loss": best_epoch_loss,
        "best_epoch_rmse": best_epoch_rmse,
        "best_epoch_mape": best_epoch_mape,
        "train_loss_at_best": train_loss_at_best,
        "generalization_gap": generalization_gap,
        "total_train_time": total_train_time,
        "total_val_time": total_val_time,
        "mean_epoch_time": mean_epoch_time,
        "epochs_recorded": len(df),
    }


def _make_json_safe(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_json_safe(v) for v in obj]
    return obj


def run_single_experiment(
    config: GridSearchConfig,
    fixed_config: Dict[str, Any],
    variable_config: Dict[str, Any],
    dataset_name: str,
    experiment_root: Path,
    run_idx: int,
) -> Dict[str, Any]:
    """
    Run one grid-search experiment and return summary dict.
    """
    merged_config = {**fixed_config, **variable_config}

    rf = receptive_field(
        kernel_size=merged_config["kernel_size"],
        blocks=merged_config["blocks"],
        layers=merged_config["layers"],
    )

    if rf < merged_config["seq_length"]:
        return {
            **merged_config,
            "dataset": dataset_name,
            "run_idx": run_idx,
            "receptive_field": rf,
            "status": f"skipped_rf_lt_seq_length_{merged_config['seq_length']}",
        }

    run_name = (
        f"run_{run_idx:03d}"
        f"_lr{merged_config['learning_rate']}"
        f"_bs{merged_config['batch_size']}"
        f"_do{merged_config['dropout']}"
        f"_wd{merged_config['weight_decay']}"
        f"_h{merged_config['nhid']}"
        f"_k{merged_config['kernel_size']}"
        f"_b{merged_config['blocks']}"
        f"_l{merged_config['layers']}"
        f"_gcn{int(merged_config['gcn_bool'])}"
        f"_adapt{int(merged_config['addaptadj'])}"
        f"_rand{int(merged_config['randomadj'])}"
    )

    save_dir = experiment_root / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    save_prefix = save_dir / "model"

    command = build_command(
        config=merged_config,
        base_dir=config.resolved_base_dir(),
        train_script_path=config.train_script_path(),
        python_executable=config.python_executable,
        save_prefix=save_prefix,
    )

    with open(save_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(_make_json_safe(merged_config), f, indent=2, ensure_ascii=False)

    with open(save_dir / "command.sh", "w", encoding="utf-8") as f:
        f.write(" ".join(map(str, command)) + "\n")

    print("=" * 100)
    print(f"[{dataset_name}] RUN {run_idx}: {run_name}")
    print(f"Save dir: {save_dir}")
    print(f"Receptive field: {rf}")
    print("Command:")
    print(" ".join(command))
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

    log_path = save_dir / "train_stdout.log"
    with open(log_path, "w", encoding="utf-8") as log_file:
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)

    process.wait()

    result = {
        **merged_config,
        "dataset": dataset_name,
        "run_idx": run_idx,
        "run_name": run_name,
        "save_dir": str(save_dir),
        "receptive_field": rf,
        "return_code": process.returncode,
    }

    if process.returncode != 0:
        result["status"] = f"train_failed_code_{process.returncode}"
        return result

    metrics_csv = save_dir / "training_metrics.csv"
    summary = summarize_training_csv(metrics_csv)

    if summary is None:
        result["status"] = "missing_training_metrics_csv"
        return result

    result.update(summary)
    result["metrics_csv"] = str(metrics_csv)
    return result


def run_grid_search(config: GridSearchConfig) -> Path:
    """
    Main entrypoint for grid search on exactly one dataset.
    Returns path to the created experiment directory.
    """
    validate_config(config)

    dataset_dir = config.dataset_dir()
    metadata = load_dataset_metadata(dataset_dir)

    job_id = os.environ.get("SLURM_JOB_ID", datetime.now().strftime("%Y%m%d_%H%M%S"))
    run_suffix = config.run_name if config.run_name else f"gridsearch_{job_id}"

    experiment_root = config.resolved_garage_root() / config.dataset_name / run_suffix
    experiment_root.mkdir(parents=True, exist_ok=True)

    fixed_config = {
        "device": config.device,
        "data": str(metadata["data_dir"]),
        "adjdata": str(metadata["adj_path"]),
        "adjtype": config.adjtype,
        "num_nodes": metadata["num_nodes"],
        "in_dim": metadata["in_dim"],
        "seq_length": metadata["seq_length"],
        "epochs": config.epochs,
        "print_every": config.print_every,
        "aptonly": config.aptonly,
    }

    candidate_configs = generate_grid(config.grid)

    with open(experiment_root / "grid_config.json", "w", encoding="utf-8") as f:
        json.dump(_make_json_safe(asdict(config)), f, indent=2, ensure_ascii=False)

    with open(experiment_root / "resolved_dataset_metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "data_dir": str(metadata["data_dir"]),
                "adj_path": str(metadata["adj_path"]),
                "seq_length": metadata["seq_length"],
                "num_nodes": metadata["num_nodes"],
                "in_dim": metadata["in_dim"],
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    with open(experiment_root / "grid_candidates.json", "w", encoding="utf-8") as f:
        json.dump(_make_json_safe(candidate_configs), f, indent=2, ensure_ascii=False)

    print("=" * 100)
    print(f"GRID SEARCH FOR DATASET: {config.dataset_name}")
    print(f"Data dir: {metadata['data_dir']}")
    print(f"Adjacency: {metadata['adj_path']}")
    print(
        f"num_nodes={metadata['num_nodes']}, "
        f"in_dim={metadata['in_dim']}, "
        f"seq_length={metadata['seq_length']}"
    )
    print(f"Total candidate configurations: {len(candidate_configs)}")
    print(f"Experiment root: {experiment_root}")
    print("=" * 100)

    all_results = []

    for run_idx, variable_config in enumerate(candidate_configs, start=1):
        result = run_single_experiment(
            config=config,
            fixed_config=fixed_config,
            variable_config=variable_config,
            dataset_name=config.dataset_name,
            experiment_root=experiment_root,
            run_idx=run_idx,
        )
        all_results.append(result)

        partial_df = pd.DataFrame(all_results)
        partial_csv = experiment_root / "grid_results.csv"
        partial_df.to_csv(partial_csv, index=False)
        print(f"\n[INFO] Partial results saved to: {partial_csv}\n")

    results_df = pd.DataFrame(all_results)

    if "best_valid_loss" in results_df.columns:
        results_df = results_df.sort_values(
            by=["status", "best_valid_loss"],
            ascending=[True, True],
            na_position="last",
        )

    final_csv = experiment_root / "grid_results_sorted.csv"
    results_df.to_csv(final_csv, index=False)

    print("=" * 100)
    print(f"GRID SEARCH FINISHED FOR DATASET: {config.dataset_name}")
    print(f"Results: {final_csv}")

    if "status" in results_df.columns:
        successful = results_df[results_df["status"] == "ok"]
    else:
        successful = pd.DataFrame()

    if not successful.empty:
        best = successful.sort_values("best_valid_loss").iloc[0]
        print("\nBest configuration:")
        print(best.to_string())
    else:
        print("\nNo successful runs found.")

    print("=" * 100)
    return experiment_root


# =========================
# PRZYKŁADOWE UŻYCIE
# =========================

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent

    config = GridSearchConfig(
        dataset_name="ingolstadt_770",

        base_dir=BASE_DIR,
        processed_root=BASE_DIR / "data" / "processed_networks",
        garage_root=BASE_DIR / "garage",

        device="cuda:0",
        epochs=20,
        print_every=10,
        adjtype="doubletransition",
        aptonly=False,

        # opcjonalnie:
        # run_name="grid_ingolstadt_770_test",

        grid=HyperparameterGrid(
            learning_rate=[1e-3],
            batch_size=[32],
            dropout=[0.2],
            weight_decay=[1e-4],
            nhid=[32],
            kernel_size=[2, 3],
            blocks=[3, 4],
            layers=[2, 3],
            gcn_bool=[True],
            addaptadj=[True, False],
            randomadj=[False, True],
        ),
    )

    run_grid_search(config)
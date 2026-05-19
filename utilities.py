import torch
import numpy as np
from DataLoader import SumoFolderDataset
from torch.utils.data import DataLoader
from pathlib import Path
import random
import pandas as pd
import matplotlib.pyplot as plt


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_learning_curves(history, output_path):
    """Save training and validation curves."""

    epochs = history["epoch"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, history["train_loss"], label="Train Loss")
    axes[0].plot(epochs, history["valid_loss"], label="Valid Loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.7)

    axes[1].plot(epochs, history["train_mae"], label="Train MAE")
    axes[1].plot(epochs, history["valid_mae"], label="Valid MAE")
    axes[1].set_title("MAE")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.7)

    axes[2].plot(epochs, history["train_rmse"], label="Train RMSE")
    axes[2].plot(epochs, history["valid_rmse"], label="Valid RMSE")
    axes[2].set_title("RMSE")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()
    axes[2].grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def load_csv_adj(csv_path, num_nodes, device):
    df = pd.read_csv(csv_path, index_col=0)

    A = df.to_numpy(dtype=np.float32)

    if A.shape[0] != A.shape[1]:
        raise ValueError(f"Adjacency must be square, got {A.shape}")

    if A.shape[0] != num_nodes:
        raise ValueError(
            f"Adjacency size {A.shape[0]} does not match num_nodes={num_nodes}"
        )

    A = torch.tensor(A, dtype=torch.float32, device=device)
    AT = A.transpose(0, 1).contiguous()
    I = torch.eye(num_nodes, dtype=torch.float32, device=device)

    return [A, AT, I]


def split_file_names(q_dir, a_dir, train_ratio=0.7, val_ratio=0.15, seed=42):
    q_dir = Path(q_dir)
    a_dir = Path(a_dir)

    q_files = {p.name for p in q_dir.glob("*.npy")}
    a_files = {p.name for p in a_dir.glob("*.npy")}
    common_files = sorted(q_files & a_files)

    if not common_files:
        raise RuntimeError(
            f"Brak wspólnych plików .npy między {q_dir} i {a_dir}"
        )

    rng = random.Random(seed)
    rng.shuffle(common_files)

    n = len(common_files)
    n_train = max(1, int(n * train_ratio))
    n_val = max(1, int(n * val_ratio))
    n_test = n - n_train - n_val

    if n_test <= 0:
        n_test = 1
        if n_train > n_val:
            n_train -= 1
        else:
            n_val -= 1

    train_files = common_files[:n_train]
    val_files = common_files[n_train:n_train + n_val]
    test_files = common_files[n_train + n_val:]

    return train_files, val_files, test_files


def make_subset_loader(
    q_dir,
    a_dir,
    selected_files,
    batch_size,
    shuffle,
    num_workers,
    seq_length_q,
    seq_length_a,
    seq_length_y,
    target_nodes,
):
    dataset = SumoFolderDataset(
        flow_dir=q_dir,
        assign_dir=a_dir,
        file_names=selected_files,
        seq_length_q=seq_length_q,
        seq_length_a=seq_length_a,
        seq_length_y=seq_length_y,
        target_nodes=target_nodes,
    )

    if len(dataset) == 0:
        raise RuntimeError("Subset datasetu jest pusty.")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=4 if num_workers > 0 else None,
    )


def infer_target_dim_from_batch(batch):
    y = batch["y"]
    if y.dim() == 2:
        # (B, N) -> one-step prediction
        return 1
    if y.dim() == 3:
        # (B, H, N)
        return y.shape[1]
    raise ValueError(f"Unsupported y shape: {tuple(y.shape)}")


def maybe_inverse_transform(scaler, x):
    if scaler is None:
        return x
    return scaler.inverse_transform(x)

def init_metric_acc():
    return {
        "loss": 0.0,
        "mae": 0.0,
        "rmse": 0.0,
        "n": 0,
    }


def update_metric_acc(acc, metrics, batch):
    bs = batch["y"].size(0)

    acc["loss"] += metrics["loss"] * bs
    acc["mae"] += metrics["mae"] * bs
    acc["rmse"] += metrics["rmse"] * bs
    acc["n"] += bs


def finalize_metric_acc(acc):
    n = max(acc["n"], 1)
    return {
        "loss": acc["loss"] / n,
        "mae": acc["mae"] / n,
        "rmse": acc["rmse"] / n,
    }

def evaluate_loader(engine, loader):
    acc = init_metric_acc()
    for batch in loader:
        metrics = engine.eval(batch)
        update_metric_acc(acc, metrics, batch)
    return finalize_metric_acc(acc)


def _ensure_batch_sequence(x: torch.Tensor, name: str) -> tuple[torch.Tensor, bool]:
    if x.dim() == 2:
        return x.unsqueeze(0), True
    if x.dim() == 3:
        return x, False
    raise ValueError(
        f"{name} must have shape (B, T, N) or (T, N), got {tuple(x.shape)}"
    )

def _normalize_lengths(
    lengths: torch.Tensor | None,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor | None:
    if lengths is None:
        return None

    if not torch.is_tensor(lengths):
        lengths = torch.as_tensor(lengths, device=device)

    lengths = lengths.to(device=device, dtype=torch.long)

    if lengths.dim() == 0:
        lengths = lengths.unsqueeze(0)

    if lengths.dim() != 1:
        raise ValueError(f"lengths must have shape (B,), got {tuple(lengths.shape)}")

    if lengths.numel() != batch_size:
        raise ValueError(
            f"lengths must contain {batch_size} elements, got {lengths.numel()}"
        )

    return lengths


def mae(pred, real):
    return torch.mean(torch.abs(pred - real))


def rmse(pred, real):
    return torch.sqrt(torch.mean((pred - real) ** 2))


def compute_metrics(pred, real, loss_value):
    return {
        "loss": loss_value.item(),
        "mae": mae(pred, real).item(),
        "rmse": rmse(pred, real).item(),
    }


def unpack_batch(batch_or_q, a=None, real_val=None, lengths=None):
    if isinstance(batch_or_q, dict) and "x" in batch_or_q and "y" in batch_or_q:
        x = batch_or_q["x"]
        q = x["q"]
        a = x["a"]
        real_val = batch_or_q["y"]

        if lengths is None:
            lengths = x.get("lengths", batch_or_q.get("lengths", None))

        return q, a, real_val, lengths

    return batch_or_q, a, real_val, lengths
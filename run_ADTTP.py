import os
import time
import math
import random
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

import util
import engine as engine_module
from engine import TrainerADTTP
from dataset_utils.DataLoader import SumoFolderDataset


Q_DIR = "/scratch/tmp/vec_flows_10s"
A_DIR = "/scratch/tmp/vec_assignments_10s"
SAVE_DIR = "/scratch/tmp/checkpoints/adttp_sumo"




def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def infer_target_dim_from_batch(batch):
    y = batch["y"]
    if y.dim() == 2:   # (B, N)
        return 1
    if y.dim() == 3:   # (B, H, N)
        return y.shape[1]
    raise ValueError(f"Unsupported y shape: {tuple(y.shape)}")


def list_common_npy_files(q_dir: str | Path, a_dir: str | Path):
    q_dir = Path(q_dir)
    a_dir = Path(a_dir)

    q_files = {p.name for p in q_dir.glob("*.npy")}
    a_files = {p.name for p in a_dir.glob("*.npy")}
    common = sorted(q_files & a_files)

    if not common:
        raise RuntimeError(f"Brak wspólnych plików .npy w:\nQ_DIR={q_dir}\nA_DIR={a_dir}")

    return common


def split_files(file_names, train_ratio=0.7, val_ratio=0.15, seed=42):
    if not math.isclose(train_ratio + val_ratio, 0.85, rel_tol=1e-6):
        # test ratio będzie 1 - train - val
        pass

    rng = random.Random(seed)
    file_names = list(file_names)
    rng.shuffle(file_names)

    n = len(file_names)
    n_train = max(1, int(n * train_ratio))
    n_val = max(1, int(n * val_ratio))
    n_test = n - n_train - n_val

    if n_test <= 0:
        n_test = 1
        if n_train > n_val:
            n_train -= 1
        else:
            n_val -= 1

    train_files = file_names[:n_train]
    val_files = file_names[n_train:n_train + n_val]
    test_files = file_names[n_train + n_val:]

    return train_files, val_files, test_files


class TorchGumbelScaler:
    """
    Scaler for Gumbel-max distributed data.

    mode="standard_gumbel":
        x -> (x - loc) / scale
        Result should follow approximately Gumbel(0, 1).

    mode="zscore":
        x -> (x - mean) / std
        Result has approximately mean 0 and std 1.

    mode="standard_normal":
        x -> Phi^{-1}(F_gumbel(x))
        Result should be approximately N(0, 1).
    """

    def __init__(self, mode="standard_gumbel", eps=1e-6):
        self.mode = mode
        self.eps = eps

        self.mean = None
        self.std = None
        self.loc = None
        self.scale = None

    def fit(self, data):
        if isinstance(data, torch.Tensor):
            x = data.detach().float().reshape(-1)
        else:
            x = torch.tensor(data, dtype=torch.float32).reshape(-1)

        self.mean = x.mean()
        self.std = x.std(unbiased=True)

        if self.std <= 0:
            raise ValueError("Cannot fit scaler: data has zero variance.")

        gamma = 0.5772156649015329

        # Method-of-moments estimates for Gumbel-max
        self.scale = self.std * math.sqrt(6.0) / math.pi
        self.loc = self.mean - gamma * self.scale

        return self

    def transform(self, data):
        is_numpy = isinstance(data, np.ndarray)

        if is_numpy:
            x = torch.tensor(data, dtype=torch.float32)
        else:
            x = data.float()

        loc = self.loc.to(x.device)
        scale = self.scale.to(x.device)
        mean = self.mean.to(x.device)
        std = self.std.to(x.device)

        y = (x - loc) / scale

        if self.mode == "standard_gumbel":
            out = y

        elif self.mode == "zscore":
            out = (x - mean) / std

        elif self.mode == "standard_normal":
            u = torch.exp(-torch.exp(-y))
            u = torch.clamp(u, self.eps, 1.0 - self.eps)

            normal = torch.distributions.Normal(
                torch.tensor(0.0, device=x.device),
                torch.tensor(1.0, device=x.device),
            )
            out = normal.icdf(u)

        else:
            raise ValueError(f"Unknown scaler mode: {self.mode}")

        return out.cpu().numpy() if is_numpy else out

    def inverse_transform(self, data):
        is_numpy = isinstance(data, np.ndarray)

        if is_numpy:
            y = torch.tensor(data, dtype=torch.float32)
        else:
            y = data.float()

        loc = self.loc.to(y.device)
        scale = self.scale.to(y.device)
        mean = self.mean.to(y.device)
        std = self.std.to(y.device)

        if self.mode == "standard_gumbel":
            out = loc + scale * y

        elif self.mode == "zscore":
            out = mean + std * y

        elif self.mode == "standard_normal":
            normal = torch.distributions.Normal(
                torch.tensor(0.0, device=y.device),
                torch.tensor(1.0, device=y.device),
            )
            u = normal.cdf(y)
            u = torch.clamp(u, self.eps, 1.0 - self.eps)
            out = loc - scale * torch.log(-torch.log(u))

        else:
            raise ValueError(f"Unknown scaler mode: {self.mode}")

        return out.cpu().numpy() if is_numpy else out


class FileFilteredSumoDataset(Dataset):
    """
    Wrapper na SumoFolderDataset, ale ograniczony do wybranych nazw plików.
    Dzięki temu nie musimy ruszać DataLoader.py.
    """
    def __init__(
        self,
        flow_dir,
        assign_dir,
        selected_files,
        seq_length_q=15,
        seq_length_a=30,
        seq_length_y=1,
        target_nodes=195,
        dtype=torch.float32,
    ):
        self.base = SumoFolderDataset(
            flow_dir=flow_dir,
            assign_dir=assign_dir,
            seq_length_q=seq_length_q,
            seq_length_a=seq_length_a,
            seq_length_y=seq_length_y,
            target_nodes=target_nodes,
            dtype=dtype,
        )
        selected_files = set(selected_files)
        self.base.samples = [s for s in self.base.samples if s[0] in selected_files]
        self.base.exp_files = sorted(selected_files)

        if len(self.base.samples) == 0:
            raise RuntimeError("Po filtrowaniu zbiór jest pusty.")

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        return self.base[idx]


def build_loader(
    q_dir,
    a_dir,
    file_names,
    batch_size,
    shuffle,
    num_workers,
    seq_length_q,
    seq_length_a,
    seq_length_y,
    target_nodes,
):
    dataset = FileFilteredSumoDataset(
        flow_dir=q_dir,
        assign_dir=a_dir,
        selected_files=file_names,
        seq_length_q=seq_length_q,
        seq_length_a=seq_length_a,
        seq_length_y=seq_length_y,
        target_nodes=target_nodes,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
    )


def evaluate_model_on_loader(engine, loader):
    losses, mapes, rmses = [], [], []

    for batch in loader:
        loss, mape, rmse = engine.eval(batch)
        losses.append(loss)
        mapes.append(mape)
        rmses.append(rmse)

    return float(np.mean(losses)), float(np.mean(mapes)), float(np.mean(rmses))


def collect_predictions(engine, loader, device):
    preds_all = []
    reals_all = []

    engine.model.eval()
    with torch.no_grad():
        for batch in loader:
            q = batch["x"]["q"].to(device)
            a = batch["x"]["a"].to(device)
            y = batch["y"].to(device)

            pred = engine.model(q, a)

            preds_all.append(pred.detach().cpu())
            reals_all.append(y.detach().cpu())

    return torch.cat(preds_all, dim=0), torch.cat(reals_all, dim=0)


def save_learning_curves(history, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    df = pd.DataFrame(history)
    csv_path = os.path.join(out_dir, "training_metrics.csv")
    df.to_csv(csv_path, index=False)

    epochs_range = history["epoch"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(epochs_range, history["train_loss"], label="Train Loss (MAE)")
    axes[0].plot(epochs_range, history["valid_loss"], label="Valid Loss (MAE)")
    axes[0].set_title("Loss (MAE)")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.7)

    axes[1].plot(epochs_range, history["train_mape"], label="Train MAPE")
    axes[1].plot(epochs_range, history["valid_mape"], label="Valid MAPE")
    axes[1].set_title("MAPE")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.7)

    axes[2].plot(epochs_range, history["train_rmse"], label="Train RMSE")
    axes[2].plot(epochs_range, history["valid_rmse"], label="Valid RMSE")
    axes[2].set_title("RMSE")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()
    axes[2].grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    plot_path = os.path.join(out_dir, "learning_curves.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"Metrics CSV zapisane do: {csv_path}")
    print(f"Wykresy zapisane do: {plot_path}")


def fit_gumbel_scaler_from_loader(loader, use_q=True, use_y=True):
    values = []

    for batch in loader:
        if use_q:
            values.append(batch["x"]["q"].reshape(-1).float())

        if use_y:
            values.append(batch["y"].reshape(-1).float())

    values = torch.cat(values, dim=0)

    scaler = TorchGumbelScaler(mode="standard_gumbel")
    scaler.fit(values)

    print("Fitted Gumbel scaler:")
    print(f"  mean  = {float(scaler.mean):.6f}")
    print(f"  std   = {float(scaler.std):.6f}")
    print(f"  loc   = {float(scaler.loc):.6f}")
    print(f"  scale = {float(scaler.scale):.6f}")

    return scaler


def main():
    parser = argparse.ArgumentParser()

    # dane
    parser.add_argument("--q_dir", type=str, default=Q_DIR)
    parser.add_argument("--a_dir", type=str, default=A_DIR)
    parser.add_argument("--save_dir", type=str, default=SAVE_DIR)

    # zewnętrzna macierz sąsiedztwa
    parser.add_argument("--adjdata", type=str, required=True, help="ścieżka do pliku adjacency")
    parser.add_argument("--adjtype", type=str, default="doubletransition")

    # device
    parser.add_argument("--device", type=str, default="cuda:0")

    # architektura / trening
    parser.add_argument("--gcn_bool", action="store_true")
    parser.add_argument("--aptonly", action="store_true")
    parser.add_argument("--addaptadj", action="store_true")
    parser.add_argument("--randomadj", action="store_true")

    parser.add_argument("--in_dim", type=int, default=1, help="dla ADTTP musi być 1")
    parser.add_argument("--num_nodes", type=int, default=195)
    parser.add_argument("--nhid", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--print_every", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--kernel_size", type=int, default=2)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)

    # okna czasowe z DataLoader.py
    parser.add_argument("--seq_length_q", type=int, default=15)
    parser.add_argument("--seq_length_a", type=int, default=30)
    parser.add_argument("--seq_length_y", type=int, default=1)

    # split po plikach
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # adjacency zewnętrznie
    sensor_ids, sensor_id_to_ind, adj_mx = util.load_adj(args.adjdata, args.adjtype)
    supports = [torch.tensor(i, dtype=torch.float32, device=device) for i in adj_mx]

    if args.randomadj:
        adjinit = None
    else:
        adjinit = supports[0]

    if args.aptonly:
        supports = None

    # split po nazwach plików
    common_files = list_common_npy_files(args.q_dir, args.a_dir)
    train_files, val_files, test_files = split_files(
        common_files,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    print(f"Liczba wspólnych plików: {len(common_files)}")
    print(f"train/val/test files: {len(train_files)}/{len(val_files)}/{len(test_files)}")

    train_loader = build_loader(
        q_dir=args.q_dir,
        a_dir=args.a_dir,
        file_names=train_files,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        seq_length_q=args.seq_length_q,
        seq_length_a=args.seq_length_a,
        seq_length_y=args.seq_length_y,
        target_nodes=args.num_nodes,
    )

    val_loader = build_loader(
        q_dir=args.q_dir,
        a_dir=args.a_dir,
        file_names=val_files,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        seq_length_q=args.seq_length_q,
        seq_length_a=args.seq_length_a,
        seq_length_y=args.seq_length_y,
        target_nodes=args.num_nodes,
    )

    test_loader = build_loader(
        q_dir=args.q_dir,
        a_dir=args.a_dir,
        file_names=test_files,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        seq_length_q=args.seq_length_q,
        seq_length_a=args.seq_length_a,
        seq_length_y=args.seq_length_y,
        target_nodes=args.num_nodes,
    )

    first_batch = next(iter(train_loader))
    target_dim = infer_target_dim_from_batch(first_batch)

    scaler = fit_gumbel_scaler_from_loader(
        train_loader,
        use_q=True,
        use_y=True,
    )

    engine = TrainerADTTP(
        scaler=scaler,
        in_dim=1,
        num_nodes=args.num_nodes,
        nhid=args.nhid,
        dropout=args.dropout,
        lrate=args.learning_rate,
        wdecay=args.weight_decay,
        device=device,
        supports=supports,
        gcn_bool=args.gcn_bool,
        addaptadj=args.addaptadj,
        aptinit=adjinit,
        kernel_size=args.kernel_size,
        blocks=args.blocks,
        layers=args.layers,
        target_dim=target_dim,
    )

    # dodatkowy patch: ADTTP ma domyślnie device="cuda", więc ustawiamy realne device po zbudowaniu
    if hasattr(engine.model, "q_encoder") and hasattr(engine.model.q_encoder, "device"):
        engine.model.q_encoder.device = device

    print("start training...", flush=True)

    history = {
        "epoch": [],
        "train_loss": [], "train_mape": [], "train_rmse": [],
        "valid_loss": [], "valid_mape": [], "valid_rmse": [],
        "train_time": [], "val_time": []
    }

    best_val_loss = float("inf")
    best_epoch = -1
    best_ckpt_path = None

    for epoch in range(1, args.epochs + 1):
        train_loss, train_mape, train_rmse = [], [], []

        t1 = time.time()
        for iter_idx, batch in enumerate(train_loader):
            metrics = engine.train(batch)
            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])

            if iter_idx % args.print_every == 0:
                print(
                    "Iter: {:04d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}".format(
                        iter_idx, train_loss[-1], train_mape[-1], train_rmse[-1]
                    ),
                    flush=True
                )
        t2 = time.time()

        s1 = time.time()
        mvalid_loss, mvalid_mape, mvalid_rmse = evaluate_model_on_loader(engine, val_loader)
        s2 = time.time()

        mtrain_loss = float(np.mean(train_loss))
        mtrain_mape = float(np.mean(train_mape))
        mtrain_rmse = float(np.mean(train_rmse))

        history["epoch"].append(epoch)
        history["train_loss"].append(mtrain_loss)
        history["train_mape"].append(mtrain_mape)
        history["train_rmse"].append(mtrain_rmse)
        history["valid_loss"].append(mvalid_loss)
        history["valid_mape"].append(mvalid_mape)
        history["valid_rmse"].append(mvalid_rmse)
        history["train_time"].append(t2 - t1)
        history["val_time"].append(s2 - s1)

        print(
            "Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, "
            "Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, "
            "Train Time: {:.2f}s, Val Time: {:.2f}s".format(
                epoch, mtrain_loss, mtrain_mape, mtrain_rmse,
                mvalid_loss, mvalid_mape, mvalid_rmse,
                (t2 - t1), (s2 - s1)
            ),
            flush=True
        )

        ckpt_path = os.path.join(args.save_dir, f"epoch_{epoch:03d}_val_{mvalid_loss:.4f}.pth")
        torch.save(engine.model.state_dict(), ckpt_path)

        if mvalid_loss < best_val_loss:
            best_val_loss = mvalid_loss
            best_epoch = epoch
            best_ckpt_path = os.path.join(args.save_dir, "best_model.pth")
            torch.save(engine.model.state_dict(), best_ckpt_path)

    print(f"Average Training Time: {np.mean(history['train_time']):.4f} sec/epoch")
    print(f"Average Validation Time: {np.mean(history['val_time']):.4f} sec")
    print(f"Best epoch: {best_epoch}, best val loss: {best_val_loss:.4f}")

    stats_dir = os.path.join(args.save_dir, "stats")
    save_learning_curves(history, stats_dir)

    # test na best model
    if best_ckpt_path is not None:
        engine.model.load_state_dict(torch.load(best_ckpt_path, map_location=device))

    test_loss, test_mape, test_rmse = evaluate_model_on_loader(engine, test_loader)
    print(
        "TEST | Loss: {:.4f}, MAPE: {:.4f}, RMSE: {:.4f}".format(
            test_loss, test_mape, test_rmse
        )
    )

    preds, reals = collect_predictions(engine, test_loader, device)
    torch.save(
        {
            "preds": preds,
            "reals": reals,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
        },
        os.path.join(args.save_dir, "test_predictions.pt")
    )

    print("Training finished.")


if __name__ == "__main__":
    main()

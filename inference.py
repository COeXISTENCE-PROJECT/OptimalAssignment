import argparse
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch

from engine import TrainerADTTP


#helpers

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
        raise RuntimeError(f"Brak wspólnych plików .npy między {q_dir} i {a_dir}")

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




def load_flow_TN(flow_path, target_nodes):
    flow = np.load(flow_path, mmap_mode="r")

    if flow.ndim != 2:
        raise ValueError(f"flow file must have shape (N, T), got {flow.shape}")

    current_nodes = flow.shape[0]
    n_timesteps = flow.shape[1]
    nodes_to_copy = min(current_nodes, target_nodes)

    out = np.zeros((n_timesteps, target_nodes), dtype=np.float32)
    out[:, :nodes_to_copy] = flow[:nodes_to_copy, :].T
    return out, current_nodes, nodes_to_copy


def load_assign_TN(assign_path, current_nodes, target_nodes):
    assign = np.load(assign_path, mmap_mode="r")
    nodes_to_copy = min(current_nodes, target_nodes)

    if assign.ndim == 2:
        if assign.shape[0] == current_nodes:
            # (N, T) -> (T, N)
            arr = assign[:nodes_to_copy, :].T
        elif assign.shape[1] == current_nodes:
            # (T, N)
            arr = assign[:, :nodes_to_copy]
        else:
            raise ValueError(f"Unsupported assign shape: {assign.shape}")

    elif assign.ndim == 3 and assign.shape[-1] == 1:
        if assign.shape[0] == current_nodes:
            # (N, T, 1) -> (T, N)
            arr = assign[:nodes_to_copy, :, 0].T
        elif assign.shape[1] == current_nodes:
            # (T, N, 1) -> (T, N)
            arr = assign[:, :nodes_to_copy, 0]
        else:
            raise ValueError(f"Unsupported assign shape: {assign.shape}")
    else:
        raise ValueError(
            f"assign must have shape (N, T), (T, N), (N, T, 1) or (T, N, 1), got {assign.shape}"
        )

    out = np.zeros((arr.shape[0], target_nodes), dtype=np.float32)
    out[:, :nodes_to_copy] = arr
    return out


def build_trainer(args, device):
    supports = load_csv_adj(args.adjdata, args.num_nodes, device)
    adjinit = None if args.randomadj else supports[0]

    trainer = TrainerADTTP(
        scaler=None,
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
        target_dim=1,
        sequence_model=args.sequence_model,
        fuse_method=args.fuse_method,
        a_embedding_size=args.a_embedding_size,
        a_hidden_size=args.a_hidden_size,
        q_rep_dim=args.q_rep_dim,
        fused_dim=args.fused_dim,
        mlp_hidden_dim=args.mlp_hidden_dim,
        attention_num_heads=args.attention_num_heads,
        attention_ff_dim=args.attention_ff_dim,
    )

    state = torch.load(args.checkpoint, map_location=device)
    trainer.model.load_state_dict(state)
    trainer.model.eval()
    return trainer


@torch.no_grad()
def rollout_one_sequence(model, device, real_q_TN, assign_TN, seq_length_q, seq_length_a):
    """
    real_q_TN: (T, N)   -- pełny prawdziwy flow, używany do seedu i ewaluacji
    assign_TN: (T, N)   -- pełny assign
    """
    seed_steps = max(seq_length_q, seq_length_a)
    T, N = real_q_TN.shape

    if T <= seed_steps:
        raise ValueError(
            f"Za krótka sekwencja: T={T}, a potrzeba > {seed_steps}"
        )

    generated_q = np.zeros((T, N), dtype=np.float32)

    # seed: pierwsze max(Lq, La) kroków prawdziwego q
    generated_q[:seed_steps] = real_q_TN[:seed_steps]

    # przewidujemy q_t dla t = seed_steps, ..., T-1
    for t in range(seed_steps, T):
        q_window = generated_q[t - seq_length_q:t]      # (Lq, N)
        a_window = assign_TN[t - seq_length_a:t]        # (La, N)

        q_tensor = torch.from_numpy(q_window).unsqueeze(0).to(device)  # (1, Lq, N)
        a_tensor = torch.from_numpy(a_window).unsqueeze(0).to(device)  # (1, La, N)

        pred = model(q_tensor, a_tensor)

        if pred.dim() == 2:
            pred_step = pred[0]             # (N,)
        elif pred.dim() == 3 and pred.shape[1] == 1:
            pred_step = pred[0, 0]          # (N,)
        else:
            raise ValueError(f"Nieoczekiwany shape predykcji: {tuple(pred.shape)}")

        generated_q[t] = pred_step.detach().cpu().numpy()

    return generated_q, seed_steps


def main():
    parser = argparse.ArgumentParser()

    # ===== pliki =====
    parser.add_argument("--q_dir", type=str, required=True)
    parser.add_argument("--a_dir", type=str, required=True)
    parser.add_argument("--adjdata", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./infer_out")

    # ===== wybór pliku testowego =====
    parser.add_argument("--file_name", type=str, default=None,
                        help="konkretna nazwa pliku .npy; jeśli pusta, biorę test_files[test_index]")
    parser.add_argument("--test_index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)

    # ===== długości sekwencji =====
    parser.add_argument("--seq_length_q", type=int, default=15)
    parser.add_argument("--seq_length_a", type=int, default=30)

    # ===== model =====
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_nodes", type=int, default=195)
    parser.add_argument("--nhid", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0001)

    parser.add_argument("--gcn_bool", action="store_true")
    parser.add_argument("--addaptadj", action="store_true")
    parser.add_argument("--randomadj", action="store_true")

    parser.add_argument("--kernel_size", type=int, default=2)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)

    parser.add_argument("--sequence_model", type=str, default="lstm",
                        choices=["lstm", "gru", "attention"])
    parser.add_argument("--fuse_method", type=str, default="Attention",
                        choices=["concatenate", "Attention"])
    parser.add_argument("--a_embedding_size", type=int, default=32)
    parser.add_argument("--a_hidden_size", type=int, default=64)
    parser.add_argument("--q_rep_dim", type=int, default=32)
    parser.add_argument("--fused_dim", type=int, default=64)
    parser.add_argument("--mlp_hidden_dim", type=int, default=128)
    parser.add_argument("--attention_num_heads", type=int, default=4)
    parser.add_argument("--attention_ff_dim", type=int, default=128)

    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        args.device if torch.cuda.is_available() else "cpu"
    )

    # odtwórz test split dokładnie jak w treningu
    _, _, test_files = split_file_names(
        q_dir=args.q_dir,
        a_dir=args.a_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    if args.file_name is not None:
        file_name = args.file_name
        if file_name not in test_files:
            raise ValueError(
                f"Plik {file_name} nie należy do test splitu dla podanych seed/ratio."
            )
    else:
        if args.test_index < 0 or args.test_index >= len(test_files):
            raise IndexError(
                f"test_index={args.test_index} poza zakresem [0, {len(test_files)-1}]"
            )
        file_name = test_files[args.test_index]

    flow_path = Path(args.q_dir) / file_name
    assign_path = Path(args.a_dir) / file_name

    print(f"Wybrany plik testowy: {file_name}")
    print(f"Flow path:   {flow_path}")
    print(f"Assign path: {assign_path}")

    real_q_TN, current_nodes, nodes_to_copy = load_flow_TN(
        flow_path, target_nodes=args.num_nodes
    )
    assign_TN = load_assign_TN(
        assign_path, current_nodes=current_nodes, target_nodes=args.num_nodes
    )

    if assign_TN.shape[0] != real_q_TN.shape[0]:
        raise ValueError(
            f"Niezgodna liczba timestepów: flow={real_q_TN.shape[0]}, assign={assign_TN.shape[0]}"
        )

    trainer = build_trainer(args, device)

    generated_q, seed_steps = rollout_one_sequence(
        model=trainer.model,
        device=device,
        real_q_TN=real_q_TN,
        assign_TN=assign_TN,
        seq_length_q=args.seq_length_q,
        seq_length_a=args.seq_length_a,
    )

    # ewaluacja tylko na części generowanej, bez seedu
    pred_eval = generated_q[seed_steps:, :nodes_to_copy]
    real_eval = real_q_TN[seed_steps:, :nodes_to_copy]

    mae = np.mean(np.abs(pred_eval - real_eval))
    rmse = np.sqrt(np.mean((pred_eval - real_eval) ** 2))

    print(f"Seed steps: {seed_steps}")
    print(f"Eval horizon: {pred_eval.shape[0]} steps")
    print(f"MAE:  {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")

    np.save(out_dir / f"{Path(file_name).stem}_pred_q.npy", generated_q[:, :nodes_to_copy])
    np.save(out_dir / f"{Path(file_name).stem}_real_q.npy", real_q_TN[:, :nodes_to_copy])
    np.save(out_dir / f"{Path(file_name).stem}_assign.npy", assign_TN[:, :nodes_to_copy])

    meta = {
        "file_name": file_name,
        "seed_steps": int(seed_steps),
        "nodes": int(nodes_to_copy),
        "timesteps": int(real_q_TN.shape[0]),
        "mae": float(mae),
        "rmse": float(rmse),
        "checkpoint": args.checkpoint,
    }
    pd.DataFrame([meta]).to_csv(out_dir / f"{Path(file_name).stem}_metrics.csv", index=False)
    print(f"Zapisano wyniki do: {out_dir}")


if __name__ == "__main__":
    main()
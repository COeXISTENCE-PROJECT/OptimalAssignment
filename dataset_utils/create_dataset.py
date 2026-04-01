from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch


ReduceMode = Literal[None, "sum", "max", "any"]
AssignmentMode = Literal["window", "prefix"]


def safe_mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def is_valid_temporal_csv(csv_path: str | Path) -> bool:
    try:
        header = pd.read_csv(csv_path, nrows=0)
    except Exception:
        return False

    columns = set(header.columns)
    has_coords = {"coord_x", "coord_y"}.issubset(columns)
    has_steps = any(str(col).startswith("Step ") for col in header.columns)
    return has_coords and has_steps


def infer_dataset_name_from_file(file_path: Path) -> str:
    stem = file_path.stem
    stem = re.sub(r"_?ep\d+(\([^)]*\))?$", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"\s+", "_", stem)
    stem = re.sub(r"[^A-Za-z0-9_\-]", "", stem)
    return stem or "dataset"


def load_torch_tensor(path: str | Path) -> torch.Tensor:
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        obj = torch.load(path, map_location="cpu")

    if isinstance(obj, dict):
        if "assignment" in obj:
            obj = obj["assignment"]
        else:
            raise ValueError(
                f"Unsupported dict format in {path}. Expected key 'assignment'."
            )

    if not isinstance(obj, torch.Tensor):
        raise TypeError(f"Loaded object from {path} is not a torch.Tensor.")

    return obj


STEP_RE = re.compile(r"^Step\s+(\d+)$")


def extract_step_columns(df: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    step_pairs = []

    for col in df.columns:
        m = STEP_RE.fullmatch(str(col).strip())
        if m:
            step_pairs.append((int(m.group(1)), col))

    if not step_pairs:
        raise ValueError("No valid 'Step <int>' columns found.")

    step_pairs.sort(key=lambda x: x[0])

    step_ids = np.array([k for k, _ in step_pairs], dtype=np.int64)
    step_columns = [col for _, col in step_pairs]

    expected = np.arange(step_ids[0], step_ids[0] + len(step_ids))
    if not np.array_equal(step_ids, expected):
        raise ValueError(
            f"Step columns are not consecutive. Found range {step_ids.tolist()[:5]} ... {step_ids.tolist()[-5:]}"
        )

    return step_columns, step_ids


def load_sumo_temporal_tensor(
    csv_path: str | Path,
    output_filename: str | Path,
) -> tuple[np.ndarray, pd.Index, np.ndarray]:
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path, index_col=0)

    step_columns, step_ids = extract_step_columns(df)

    X_ep = df[step_columns].to_numpy()   # (N, T)
    X = np.expand_dims(X_ep.T, axis=-1)  # (T, N, 1)
    nodes = df.index

    safe_mkdir(Path(output_filename).parent)
    np.savez_compressed(
        output_filename,
        X=X,
        nodes=nodes.to_numpy(),
        step_ids=step_ids,
    )

    print(f"[tensor] processed: {csv_path.name}, shape={X.shape}")
    print(f"[tensor] saved: {output_filename}, shape={X.shape}")
    return X, nodes, step_ids


def load_assignment_timeline(
    assignment_pt: str | Path,
    output_filename: str | Path,
) -> dict[str, np.ndarray]:
    """
    Wczytuje pojedynczy tensor assignmentów i zapisuje go w formie sparse timeline.

    Input tensor shape:
        (N, N, M, T)

    Saved keys:
    - a_i, a_j, a_m, a_t, a_values
    - a_time_ptr
    - assignment_shape_step = [N, N, M]
    - T_total
    """
    assignment_pt = Path(assignment_pt)
    A = load_torch_tensor(assignment_pt)

    if A.layout != torch.sparse_coo:
        A = A.to_sparse()

    A = A.coalesce()

    if A.ndim != 4:
        raise ValueError(
            f"Assignment tensor must be 4D (N, N, M, T). Got shape={tuple(A.shape)} in {assignment_pt}"
        )

    N1, N2, M, T_total = map(int, A.shape)
    if N1 != N2:
        raise ValueError(
            f"Assignment tensor first two dims must match. Got {A.shape} in {assignment_pt}"
        )

    if A._nnz() == 0:
        a_i = np.empty((0,), dtype=np.int64)
        a_j = np.empty((0,), dtype=np.int64)
        a_m = np.empty((0,), dtype=np.int64)
        a_t = np.empty((0,), dtype=np.int64)
        a_values = np.empty((0,), dtype=np.float32)
        a_time_ptr = np.zeros(T_total + 1, dtype=np.int64)

        print(f"[assignment] processed: {assignment_pt.name}, shape={tuple(A.shape)}, nnz=0")
    else:
        idx = A.indices().cpu().numpy()    # (4, nnz)
        vals = A.values().cpu().numpy()    # (nnz,)

        if vals.ndim > 1:
            vals = vals.reshape(-1)

        order = np.argsort(idx[3], kind="stable")
        idx = idx[:, order]
        vals = vals[order]

        a_i = idx[0].astype(np.int64, copy=False)
        a_j = idx[1].astype(np.int64, copy=False)
        a_m = idx[2].astype(np.int64, copy=False)
        a_t = idx[3].astype(np.int64, copy=False)
        a_values = vals.astype(np.float32, copy=False)

        counts_per_t = np.bincount(a_t, minlength=T_total)
        a_time_ptr = np.zeros(T_total + 1, dtype=np.int64)
        a_time_ptr[1:] = np.cumsum(counts_per_t)

        print(
            f"[assignment] processed: {assignment_pt.name}, "
            f"shape={tuple(A.shape)}, nnz={A._nnz()}"
        )

    result = {
        "a_i": a_i,
        "a_j": a_j,
        "a_m": a_m,
        "a_t": a_t,
        "a_values": a_values,
        "a_time_ptr": a_time_ptr,
        "assignment_shape_step": np.array([N1, N2, M], dtype=np.int64),
        "T_total": np.array([T_total], dtype=np.int64),
    }

    safe_mkdir(Path(output_filename).parent)
    np.savez_compressed(output_filename, **result)

    print(
        f"[assignment] saved: {output_filename}, "
        f"T_total={T_total}, nnz_total={len(a_values)}, shape_step=({N1}, {N2}, {M})"
    )

    return result


def left_pad_time_axis(
    arr: np.ndarray,
    target_len: int,
    pad_value: float = 0.0,
) -> np.ndarray:
    """
    Padding po pierwszej osi (czas).
    """
    current_len = arr.shape[0]
    if current_len > target_len:
        raise ValueError(f"current_len={current_len} > target_len={target_len}")

    if current_len == target_len:
        return arr

    pad_shape = (target_len - current_len, *arr.shape[1:])
    pad = np.full(pad_shape, pad_value, dtype=arr.dtype)
    return np.concatenate([pad, arr], axis=0)


def build_dense_assignment_block(
    assignment_data: dict[str, np.ndarray],
    start_t: int,
    end_t: int,
    reduce_agents: ReduceMode = None,
    dtype=np.float32,
) -> np.ndarray:
    """
    Odtwarza gęsty blok assignmentów dla zakresu [start_t, end_t].

    Bez redukcji:
        shape = (L, N, N, M)
    Po redukcji po agentach:
        shape = (L, N, N)
    """
    if end_t < start_t:
        raise ValueError(f"end_t={end_t} < start_t={start_t}")

    N1, N2, M = map(int, assignment_data["assignment_shape_step"])
    L = end_t - start_t + 1

    A = np.zeros((L, N1, N2, M), dtype=dtype)

    a_i = assignment_data["a_i"]
    a_j = assignment_data["a_j"]
    a_m = assignment_data["a_m"]
    a_values = assignment_data["a_values"]
    a_time_ptr = assignment_data["a_time_ptr"]

    for tau in range(start_t, end_t + 1):
        ptr_start = int(a_time_ptr[tau])
        ptr_end = int(a_time_ptr[tau + 1])

        if ptr_start == ptr_end:
            continue

        local_t = tau - start_t
        i = a_i[ptr_start:ptr_end]
        j = a_j[ptr_start:ptr_end]
        m = a_m[ptr_start:ptr_end]
        v = a_values[ptr_start:ptr_end].astype(dtype, copy=False)

        A[local_t, i, j, m] = v

    if reduce_agents is None:
        return A

    if reduce_agents == "sum":
        return A.sum(axis=-1)  # (L, N, N)

    if reduce_agents == "max":
        return A.max(axis=-1)  # (L, N, N)

    if reduce_agents == "any":
        return (A > 0).any(axis=-1).astype(dtype)

    raise ValueError(f"Unsupported reduce_agents={reduce_agents}")


def generate_qA_supervised_dataset(
    X: np.ndarray,
    assignment_data: dict[str, np.ndarray],
    output_dir: str | Path,
    intermediate_dir: str | Path,
    seq_length_q: int = 12,
    seq_length_a: int = 12,
    seq_length_y: int = 1,
    y_start: int = 1,
    assignment_mode: AssignmentMode = "window",
    a_prefix_pad_to: int | None = None,
    reduce_agents: ReduceMode = "sum",
    split_ratio: tuple[float, float, float] = (0.7, 0.1, 0.2),
) -> None:
    """
    Tworzy zbiór:
        (q_{t-Lq+1}, ..., q_t, A_hist) -> (q_{t+y_start}, ..., q_{t+y_start+Ly-1})

    Parametry:
    - seq_length_q: długość okna dla q
    - seq_length_a: długość okna dla A w trybie "window"
    - seq_length_y: długość targetu
    - y_start: przesunięcie targetu względem t
    - assignment_mode:
        * "window" -> A_{t-La+1} ... A_t (z paddingiem na początku)
        * "prefix" -> A_0 ... A_t (z paddingiem do a_prefix_pad_to)
    - reduce_agents:
        * None  -> zachowaj M, x_a shape = (S, L, N, N, M)
        * "sum" -> redukcja po M, x_a shape = (S, L, N, N)
    """
    if seq_length_q <= 0:
        raise ValueError("seq_length_q must be > 0")
    if seq_length_a <= 0:
        raise ValueError("seq_length_a must be > 0")
    if seq_length_y <= 0:
        raise ValueError("seq_length_y must be > 0")
    if y_start <= 0:
        raise ValueError("y_start must be > 0")

    T_x, N_x, C = X.shape
    T_a = int(assignment_data["T_total"][0])

    if int(assignment_data["assignment_shape_step"][0]) != N_x:
        raise ValueError(
            f"Node mismatch: X has N={N_x}, assignment has N={assignment_data['assignment_shape_step'][0]}"
        )

    if T_x not in (T_a, T_a + 1):
        raise ValueError(
            f"Unsupported time mismatch: X has T={T_x}, assignment has T={T_a}. "
            "Expected T_x == T_a or T_x == T_a + 1."
        )

    min_t = seq_length_q - 1
    max_t_from_q = T_x - y_start - seq_length_y
    max_t_from_a = T_a - 1
    max_t = min(max_t_from_q, max_t_from_a)
    max_t_exclusive = max_t + 1

    T = T_x

    if assignment_mode == "prefix":
        if a_prefix_pad_to is None:
            a_prefix_pad_to = T
        if a_prefix_pad_to <= 0:
            raise ValueError("a_prefix_pad_to must be > 0 for prefix mode.")
        a_output_len = a_prefix_pad_to
    elif assignment_mode == "window":
        a_output_len = seq_length_a
    else:
        raise ValueError(f"Unsupported assignment_mode={assignment_mode}")

    # t = koniec wejściowego okna q
    min_t = seq_length_q - 1
    max_t_exclusive = T - y_start - seq_length_y + 1

    if max_t_exclusive <= min_t:
        raise ValueError(
            f"Series too short: T={T}, seq_length_q={seq_length_q}, "
            f"seq_length_y={seq_length_y}, y_start={y_start}"
        )

    x_q_list = []
    x_a_list = []
    y_list = []

    q_window_start_t = []
    q_window_end_t = []
    a_window_start_t = []
    a_window_end_t = []

    for t in range(min_t, max_t_exclusive):
        # q history
        q_hist = X[t - seq_length_q + 1 : t + 1]  # (Lq, N, C)

        # target
        y = X[t + y_start : t + y_start + seq_length_y]  # (Ly, N, C)

        # A history
        if assignment_mode == "window":
            a_start = max(0, t - seq_length_a + 1)
            a_end = t

            A_hist = build_dense_assignment_block(
                assignment_data=assignment_data,
                start_t=a_start,
                end_t=a_end,
                reduce_agents=reduce_agents,
                dtype=np.float32,
            )
            A_hist = left_pad_time_axis(A_hist, seq_length_a)

        else:  # prefix
            a_start = 0
            a_end = t

            A_hist = build_dense_assignment_block(
                assignment_data=assignment_data,
                start_t=a_start,
                end_t=a_end,
                reduce_agents=reduce_agents,
                dtype=np.float32,
            )

            if A_hist.shape[0] > a_prefix_pad_to:
                raise ValueError(
                    f"Prefix length {A_hist.shape[0]} exceeds a_prefix_pad_to={a_prefix_pad_to}. "
                    "Increase a_prefix_pad_to or use assignment_mode='window'."
                )

            A_hist = left_pad_time_axis(A_hist, a_prefix_pad_to)

        x_q_list.append(q_hist.astype(np.float32, copy=False))
        x_a_list.append(A_hist.astype(np.float32, copy=False))
        y_list.append(y.astype(np.float32, copy=False))

        q_window_start_t.append(t - seq_length_q + 1)
        q_window_end_t.append(t)

        a_window_start_t.append(a_start)
        a_window_end_t.append(a_end)

    x_q = np.stack(x_q_list, axis=0)
    x_a = np.stack(x_a_list, axis=0)
    y = np.stack(y_list, axis=0)

    q_window_start_t = np.asarray(q_window_start_t, dtype=np.int64)
    q_window_end_t = np.asarray(q_window_end_t, dtype=np.int64)
    a_window_start_t = np.asarray(a_window_start_t, dtype=np.int64)
    a_window_end_t = np.asarray(a_window_end_t, dtype=np.int64)

    print(f"[dataset] x_q shape = {x_q.shape}")
    print(f"[dataset] x_a shape = {x_a.shape}")
    print(f"[dataset] y   shape = {y.shape}")

    train_ratio, val_ratio, test_ratio = split_ratio
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("split_ratio must sum to 1.0")

    S = x_q.shape[0]
    num_train = int(np.floor(S * train_ratio))
    num_val = int(np.floor(S * val_ratio))
    num_test = S - num_train - num_val

    idx_train = slice(0, num_train)
    idx_val = slice(num_train, num_train + num_val)
    idx_test = slice(num_train + num_val, S)

    safe_mkdir(Path(output_dir))
    safe_mkdir(Path(intermediate_dir))

    common_meta = {
        "seq_length_q": np.array([seq_length_q], dtype=np.int64),
        "seq_length_a": np.array([seq_length_a], dtype=np.int64),
        "a_output_len": np.array([a_output_len], dtype=np.int64),
        "seq_length_y": np.array([seq_length_y], dtype=np.int64),
        "y_start": np.array([y_start], dtype=np.int64),
        "assignment_mode_code": np.array(
            [0 if assignment_mode == "window" else 1],
            dtype=np.int64,
        ),
        "reduce_agents_code": np.array(
            [
                -1 if reduce_agents is None else
                0 if reduce_agents == "sum" else
                1 if reduce_agents == "max" else
                2
            ],
            dtype=np.int64,
        ),
        "assignment_shape_step": assignment_data["assignment_shape_step"],
        "T_total": np.array([T], dtype=np.int64),
    }

    # save all
    np.savez_compressed(
        Path(intermediate_dir) / "windows_dense_all_qA.npz",
        x_q=x_q,
        x_a=x_a,
        y=y,
        q_window_start_t=q_window_start_t,
        q_window_end_t=q_window_end_t,
        a_window_start_t=a_window_start_t,
        a_window_end_t=a_window_end_t,
        **common_meta,
    )
    print(f"[dataset] saved all: {Path(intermediate_dir) / 'windows_dense_all_qA.npz'}")

    splits = {
        "train": (
            x_q[idx_train],
            x_a[idx_train],
            y[idx_train],
            q_window_start_t[idx_train],
            q_window_end_t[idx_train],
            a_window_start_t[idx_train],
            a_window_end_t[idx_train],
        ),
        "val": (
            x_q[idx_val],
            x_a[idx_val],
            y[idx_val],
            q_window_start_t[idx_val],
            q_window_end_t[idx_val],
            a_window_start_t[idx_val],
            a_window_end_t[idx_val],
        ),
        "test": (
            x_q[idx_test],
            x_a[idx_test],
            y[idx_test],
            q_window_start_t[idx_test],
            q_window_end_t[idx_test],
            a_window_start_t[idx_test],
            a_window_end_t[idx_test],
        ),
    }

    for split_name, payload in splits.items():
        x_q_split, x_a_split, y_split, q_s, q_e, a_s, a_e = payload

        filepath = Path(output_dir) / f"{split_name}.npz"
        np.savez_compressed(
            filepath,
            x_q=x_q_split,
            x_a=x_a_split,
            y=y_split,
            q_window_start_t=q_s,
            q_window_end_t=q_e,
            a_window_start_t=a_s,
            a_window_end_t=a_e,
            **common_meta,
        )

        print(
            f"[{split_name}] saved: {filepath} -> "
            f"x_q={x_q_split.shape}, x_a={x_a_split.shape}, y={y_split.shape}, samples={x_q_split.shape[0]}"
        )


def process_explicit_files_qA(
    network_csv: str | Path,
    assignment_pt: str | Path,
    output_root: str | Path = "data/processed_networks_with_assignments",
    dataset_name: str = "manual_dataset",
    seq_length_q: int = 12,
    seq_length_a: int = 12,
    seq_length_y: int = 1,
    y_start: int = 1,
    assignment_mode: AssignmentMode = "window",
    a_prefix_pad_to: int | None = None,
    reduce_agents: ReduceMode = "sum",
    split_ratio: tuple[float, float, float] = (0.7, 0.1, 0.2),
) -> None:
    network_csv = Path(network_csv)
    assignment_pt = Path(assignment_pt)

    if not network_csv.exists():
        raise FileNotFoundError(f"Network CSV not found: {network_csv}")
    if not assignment_pt.exists():
        raise FileNotFoundError(f"Assignment PT not found: {assignment_pt}")

    dataset_output_dir = safe_mkdir(Path(output_root) / dataset_name)
    intermediate_dir = safe_mkdir(dataset_output_dir / "intermediate_tensors")

    print(f"Processing single-episode dataset: {dataset_name}")
    print(f"Network CSV: {network_csv}")
    print(f"Assignment PT: {assignment_pt}")
    print(f"Output directory: {dataset_output_dir}")

    X, nodes, _ = load_sumo_temporal_tensor(
        csv_path=network_csv,
        output_filename=intermediate_dir / "temporal_tensor_full.npz",
    )
    np.savez_compressed(intermediate_dir / "nodes.npz", nodes=nodes.to_numpy())

    assignment_data = load_assignment_timeline(
        assignment_pt=assignment_pt,
        output_filename=intermediate_dir / "assignment_timeline_full.npz",
    )

    generate_qA_supervised_dataset(
        X=X,
        assignment_data=assignment_data,
        output_dir=dataset_output_dir,
        intermediate_dir=intermediate_dir,
        seq_length_q=seq_length_q,
        seq_length_a=seq_length_a,
        seq_length_y=seq_length_y,
        y_start=y_start,
        assignment_mode=assignment_mode,
        a_prefix_pad_to=a_prefix_pad_to,
        reduce_agents=reduce_agents,
        split_ratio=split_ratio,
    )


if __name__ == "__main__":
    process_explicit_files_qA(
        network_csv="data/networks/ingolstadt_770_assignments.csv",
        assignment_pt="data/assignments/assignment_M10_T1032.pt",
        output_root="data/processed_networks_with_assignments",
        dataset_name="manual_dataset",

        # q-window
        seq_length_q=12,

        # A-window
        seq_length_a=20,

        # target
        seq_length_y=1,
        y_start=1,

        # "window" albo "prefix"
        assignment_mode="window",

        # używane tylko gdy assignment_mode="prefix"
        a_prefix_pad_to=None,

        # None / "sum" / "max" / "any"
        reduce_agents="sum",

        split_ratio=(0.7, 0.1, 0.2),
    )
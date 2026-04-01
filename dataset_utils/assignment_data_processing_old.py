from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import kneighbors_graph


def safe_mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def is_valid_temporal_csv(csv_path: str | Path) -> bool:
    """
    Check whether CSV has coordinates and at least one temporal column.
    """
    try:
        header = pd.read_csv(csv_path, nrows=0)
    except Exception:
        return False

    columns = set(header.columns)
    has_coords = {"coord_x", "coord_y"}.issubset(columns)
    has_steps = any(str(col).startswith("Step ") for col in header.columns)
    return has_coords and has_steps


def infer_dataset_name_from_file(file_path: Path) -> str:
    """
    Fallback naming if files are placed directly in root instead of subfolders.
    Example:
    traffic_heatmap_data_ep1(in).csv -> traffic_heatmap_data
    assignment_ep1.pt -> assignment
    """
    stem = file_path.stem
    stem = re.sub(r"_?ep\d+(\([^)]*\))?$", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"\s+", "_", stem)
    stem = re.sub(r"[^A-Za-z0-9_\-]", "", stem)
    return stem or "dataset"


def discover_dataset_inputs(networks_root: str | Path) -> dict[str, list[Path]]:
    """
    Discover datasets in two layouts:

    1) Preferred:
       data/networks/<dataset_name>/*.csv

    2) Fallback:
       data/networks/*.csv
       then files are grouped by filename prefix before ep-number
    """
    root = Path(networks_root)
    if not root.exists():
        raise FileNotFoundError(f"Networks directory does not exist: {root}")

    datasets: dict[str, list[Path]] = {}

    # Preferred layout: one folder = one dataset
    for subdir in sorted(p for p in root.iterdir() if p.is_dir()):
        csv_files = sorted(subdir.glob("*.csv"))
        valid_files = [p for p in csv_files if is_valid_temporal_csv(p)]
        if valid_files:
            datasets[subdir.name] = valid_files

    # Fallback layout: raw CSVs directly inside networks/
    root_csvs = sorted(p for p in root.glob("*.csv") if p.is_file())
    grouped: dict[str, list[Path]] = {}

    for csv_path in root_csvs:
        if not is_valid_temporal_csv(csv_path):
            continue
        dataset_name = infer_dataset_name_from_file(csv_path)
        grouped.setdefault(dataset_name, []).append(csv_path)

    for dataset_name, files in grouped.items():
        datasets.setdefault(dataset_name, []).extend(sorted(files))

    if not datasets:
        raise ValueError(
            f"No valid temporal CSV files found in: {root}. "
            "Expected columns: coord_x, coord_y and at least one 'Step ...' column."
        )

    return datasets


def discover_assignment_inputs(assignments_root: str | Path) -> dict[str, list[Path]]:
    """
    Discover assignment tensors (.pt) in two layouts:

    1) Preferred:
       data/assignments/<dataset_name>/*.pt

    2) Fallback:
       data/assignments/*.pt
       then files are grouped by filename prefix before ep-number
    """
    root = Path(assignments_root)
    if not root.exists():
        raise FileNotFoundError(f"Assignments directory does not exist: {root}")

    datasets: dict[str, list[Path]] = {}

    # Preferred layout: one folder = one dataset
    for subdir in sorted(p for p in root.iterdir() if p.is_dir()):
        pt_files = sorted(subdir.glob("*.pt"))
        if pt_files:
            datasets[subdir.name] = pt_files

    # Fallback layout: raw PTs directly inside assignments/
    root_pts = sorted(p for p in root.glob("*.pt") if p.is_file())
    grouped: dict[str, list[Path]] = {}

    for pt_path in root_pts:
        dataset_name = infer_dataset_name_from_file(pt_path)
        grouped.setdefault(dataset_name, []).append(pt_path)

    for dataset_name, files in grouped.items():
        datasets.setdefault(dataset_name, []).extend(sorted(files))

    if not datasets:
        raise ValueError(f"No .pt assignment files found in: {root}")

    return datasets

def process_explicit_files(
    network_csv: str | Path,
    assignment_pt: str | Path,
    output_root: str | Path = "data/processed_networks_with_assignments",
    dataset_name: str = "manual_dataset",
    epsilon: float = 500.0,
    sigma: float | None = None,
    use_knn: bool = False,
    knn_k: int = 10,
    seq_length_x: int = 12,
    seq_length_y: int = 1,
    y_start: int = 1,
    split_ratio: tuple[float, float, float] = (0.7, 0.1, 0.2),
) -> None:
    network_csv = Path(network_csv)
    assignment_pt = Path(assignment_pt)

    if not network_csv.exists():
        raise FileNotFoundError(f"Network CSV not found: {network_csv}")
    if not assignment_pt.exists():
        raise FileNotFoundError(f"Assignment PT not found: {assignment_pt}")

    dataset_output_dir = safe_mkdir(Path(output_root) / dataset_name)
#    adjacency_dir = safe_mkdir(dataset_output_dir / "adjacency_matrices")
    intermediate_dir = safe_mkdir(dataset_output_dir / "intermediate_tensors")

    print(f"Processing explicit dataset: {dataset_name}")
    print(f"Network CSV: {network_csv}")
    print(f"Assignment PT: {assignment_pt}")
    print(f"Output directory: {dataset_output_dir}")

    # create_spatial_adjacency_matrix(
    #     data_path=network_csv,
    #     output_path=adjacency_dir / "adjacency_spatial.csv",
    #     sigma=sigma,
    #     epsilon=epsilon,
    # )
    #
    # if use_knn:
    #     create_knn_adjacency_matrix(
    #         data_path=network_csv,
    #         output_path=adjacency_dir / f"adjacency_knn_k{knn_k}.csv",
    #         k=knn_k,
    #     )

    X_full, nodes = compile_sumo_temporal_tensor(
        filepaths=[network_csv],
        output_filename=intermediate_dir / "temporal_tensor_full.npz",
    )
    np.savez_compressed(intermediate_dir / "nodes.npz", nodes=nodes.to_numpy())

    assignment_data = compile_assignment_timeline(
        filepaths=[assignment_pt],
        output_filename=intermediate_dir / "assignment_timeline_full.npz",
    )

    generate_wavenet_assignment_tensors(
        X=X_full,
        assignment_data=assignment_data,
        output_dir=dataset_output_dir,
        intermediate_dir=intermediate_dir,
        seq_length_x=seq_length_x,
        seq_length_y=seq_length_y,
        y_start=y_start,
        split_ratio=split_ratio,
    )


def create_spatial_adjacency_matrix(
    data_path: str | Path,
    output_path: str | Path,
    sigma: float | None = None,
    epsilon: float = 500.0,
) -> pd.DataFrame:
    """
    Create weighted adjacency matrix from spatial coordinates.
    """
    df = pd.read_csv(data_path, index_col=0)

    coords = df[["coord_x", "coord_y"]].values
    dist_matrix = squareform(pdist(coords, metric="euclidean"))

    if sigma is None:
        sigma = np.std(dist_matrix)
        if sigma == 0:
            sigma = 1e-6

    adjacency = np.exp(-((dist_matrix ** 2) / (sigma ** 2)))
    adjacency[dist_matrix > epsilon] = 0.0
    np.fill_diagonal(adjacency, 0.0)

    adj_df = pd.DataFrame(adjacency, index=df.index, columns=df.index)

    safe_mkdir(Path(output_path).parent)
    adj_df.to_csv(output_path)

    print(f"[adjacency] saved: {output_path}")
    print(f"[adjacency] nodes={adjacency.shape[0]}, non_zero={np.count_nonzero(adjacency)}")

    return adj_df


def create_knn_adjacency_matrix(
    data_path: str | Path,
    output_path: str | Path,
    k: int = 10,
):
    """
    Create symmetric binary k-NN adjacency matrix.
    """
    df = pd.read_csv(data_path, index_col=0)
    coords = df[["coord_x", "coord_y"]].values
    n_nodes = coords.shape[0]

    adjacency_directed = kneighbors_graph(
        X=coords,
        n_neighbors=k,
        mode="connectivity",
        include_self=False,
        n_jobs=-1,
    )

    adjacency_symmetric = adjacency_directed.maximum(adjacency_directed.T)

    adj_knn_df = pd.DataFrame(
        adjacency_symmetric.toarray(),
        index=df.index,
        columns=df.index,
    )

    safe_mkdir(Path(output_path).parent)
    adj_knn_df.to_csv(output_path)

    degrees = np.array(adjacency_symmetric.sum(axis=1)).flatten()
    density = adjacency_symmetric.nnz / (n_nodes * (n_nodes - 1))

    print(f"[knn] saved: {output_path}")
    print(f"[knn] nodes={n_nodes}, edges={adjacency_symmetric.nnz}, sparsity={1.0 - density:.4f}")
    print(f"[knn] mean_degree={np.mean(degrees):.2f}, min_degree={np.min(degrees)}, max_degree={np.max(degrees)}")

    return adjacency_symmetric


def compile_sumo_temporal_tensor(
    filepaths: Iterable[str | Path],
    output_filename: str | Path,
) -> tuple[np.ndarray, pd.Index]:
    """
    Build one temporal tensor from multiple CSV files.
    Output shape: (T_total, N, C)
    """
    filepaths = sorted(Path(p) for p in filepaths)
    if not filepaths:
        raise ValueError("No input CSV files provided.")

    tensor_list: list[np.ndarray] = []
    canonical_nodes: pd.Index | None = None

    for path in filepaths:
        df = pd.read_csv(path, index_col=0)
        step_columns = [col for col in df.columns if str(col).startswith("Step ")]
        if not step_columns:
            raise ValueError(f"No 'Step ...' columns found in file: {path}")

        if canonical_nodes is None:
            canonical_nodes = df.index
        else:
            df = df.reindex(canonical_nodes)
            if df[step_columns].isna().any(axis=None):
                raise ValueError(
                    f"Node mismatch after reindex for file: {path}. "
                    "At least one node is missing compared to canonical ordering."
                )

        X_ep = df[step_columns].to_numpy()             # (N, T)
        X_ep_tensor = np.expand_dims(X_ep.T, axis=-1) # (T, N, 1)

        tensor_list.append(X_ep_tensor)
        print(f"[tensor] processed: {path.name}, shape={X_ep_tensor.shape}")

    X_full = np.concatenate(tensor_list, axis=0)

    safe_mkdir(Path(output_filename).parent)
    np.savez_compressed(output_filename, X=X_full, nodes=canonical_nodes.to_numpy())

    print(f"[tensor] saved: {output_filename}, shape={X_full.shape}")

    return X_full, canonical_nodes


def load_torch_tensor(path: str | Path) -> torch.Tensor:
    """
    Load tensor from .pt file.
    Supports plain tensor or dict with key 'assignment'.
    """
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


def compile_assignment_timeline(
    filepaths: Iterable[str | Path],
    output_filename: str | Path,
) -> dict[str, np.ndarray]:
    """
    Compile one sparse assignment timeline from one or more .pt files.

    Input tensors must have shape: (N, N, M, T_ep)
    and preferably be sparse COO.

    Output is saved as compressed NPZ with:
    - a_i, a_j, a_m, a_t, a_values
    - a_time_ptr  (prefix pointer per time step)
    - assignment_shape_step = [N, N, M]
    - T_total

    Semantics:
    All non-zero assignment entries across time are flattened and sorted by time.
    For any sample ending at time t, history A_0 ... A_t is simply the prefix
    [: a_time_ptr[t + 1]].
    """
    filepaths = sorted(Path(p) for p in filepaths)
    if not filepaths:
        raise ValueError("No assignment .pt files provided.")

    all_i = []
    all_j = []
    all_m = []
    all_t = []
    all_v = []

    N_ref = None
    M_ref = None
    time_offset = 0

    for path in filepaths:
        A = load_torch_tensor(path)

        if A.layout != torch.sparse_coo:
            A = A.to_sparse()

        A = A.coalesce()

        if A.ndim != 4:
            raise ValueError(
                f"Assignment tensor must be 4D (N, N, M, T). Got shape={tuple(A.shape)} in {path}"
            )

        N1, N2, M, T_ep = map(int, A.shape)
        if N1 != N2:
            raise ValueError(f"Assignment tensor first two dims must match. Got {A.shape} in {path}")

        if N_ref is None:
            N_ref = N1
            M_ref = M
        else:
            if N1 != N_ref or M != M_ref:
                raise ValueError(
                    f"Assignment shape mismatch in {path}. Expected N={N_ref}, M={M_ref}, "
                    f"got N={N1}, M={M}."
                )

        if A._nnz() == 0:
            print(f"[assignment] processed: {path.name}, shape={tuple(A.shape)}, nnz=0")
            time_offset += T_ep
            continue

        idx = A.indices().cpu().numpy()    # shape: (4, nnz)
        vals = A.values().cpu().numpy()    # shape: (nnz,)

        if vals.ndim > 1:
            vals = vals.reshape(-1)

        # sort by local time index
        order = np.argsort(idx[3], kind="stable")
        idx = idx[:, order]
        vals = vals[order]

        i = idx[0].astype(np.int64, copy=False)
        j = idx[1].astype(np.int64, copy=False)
        m = idx[2].astype(np.int64, copy=False)
        t = idx[3].astype(np.int64, copy=False) + time_offset
        v = vals.astype(np.uint8, copy=False)

        all_i.append(i)
        all_j.append(j)
        all_m.append(m)
        all_t.append(t)
        all_v.append(v)

        print(f"[assignment] processed: {path.name}, shape={tuple(A.shape)}, nnz={A._nnz()}")
        time_offset += T_ep

    T_total = time_offset

    if all_t:
        a_i = np.concatenate(all_i, axis=0)
        a_j = np.concatenate(all_j, axis=0)
        a_m = np.concatenate(all_m, axis=0)
        a_t = np.concatenate(all_t, axis=0)
        a_values = np.concatenate(all_v, axis=0)

        order = np.argsort(a_t, kind="stable")
        a_i = a_i[order]
        a_j = a_j[order]
        a_m = a_m[order]
        a_t = a_t[order]
        a_values = a_values[order]

        counts_per_t = np.bincount(a_t, minlength=T_total)
        a_time_ptr = np.zeros(T_total + 1, dtype=np.int64)
        a_time_ptr[1:] = np.cumsum(counts_per_t)
    else:
        a_i = np.empty((0,), dtype=np.int64)
        a_j = np.empty((0,), dtype=np.int64)
        a_m = np.empty((0,), dtype=np.int64)
        a_t = np.empty((0,), dtype=np.int64)
        a_values = np.empty((0,), dtype=np.uint8)
        a_time_ptr = np.zeros(T_total + 1, dtype=np.int64)

    result = {
        "a_i": a_i,
        "a_j": a_j,
        "a_m": a_m,
        "a_t": a_t,
        "a_values": a_values,
        "a_time_ptr": a_time_ptr,
        "assignment_shape_step": np.array([N_ref, N_ref, M_ref], dtype=np.int64),
        "T_total": np.array([T_total], dtype=np.int64),
    }

    safe_mkdir(Path(output_filename).parent)
    np.savez_compressed(output_filename, **result)

    print(
        f"[assignment] saved: {output_filename}, "
        f"T_total={T_total}, nnz_total={len(a_values)}, shape_step=({N_ref}, {N_ref}, {M_ref})"
    )

    return result


def trim_assignment_timeline(
    assignment_data: dict[str, np.ndarray],
    T_keep: int,
) -> dict[str, np.ndarray]:
    """
    Trim assignment timeline to first T_keep time steps.
    """
    a_time_ptr = assignment_data["a_time_ptr"]
    if T_keep < 0 or T_keep > len(a_time_ptr) - 1:
        raise ValueError(f"Invalid T_keep={T_keep}. Must be in [0, {len(a_time_ptr) - 1}]")

    end_ptr = int(a_time_ptr[T_keep])

    trimmed = {
        "a_i": assignment_data["a_i"][:end_ptr],
        "a_j": assignment_data["a_j"][:end_ptr],
        "a_m": assignment_data["a_m"][:end_ptr],
        "a_t": assignment_data["a_t"][:end_ptr],
        "a_values": assignment_data["a_values"][:end_ptr],
        "a_time_ptr": assignment_data["a_time_ptr"][: T_keep + 1],
        "assignment_shape_step": assignment_data["assignment_shape_step"],
        "T_total": np.array([T_keep], dtype=np.int64),
    }
    return trimmed


def generate_wavenet_assignment_tensors(
    X: np.ndarray,
    assignment_data: dict[str, np.ndarray],
    output_dir: str | Path,
    intermediate_dir: str | Path,
    seq_length_x: int = 12,
    seq_length_y: int = 12,
    y_start: int = 1,
    split_ratio: tuple[float, float, float] = (0.7, 0.1, 0.2),
) -> None:
    """
    Generate Graph WaveNet windows augmented with assignment history.

    Input:
      X shape: (T, N, C)

    Assignment representation:
      global sparse timeline arrays:
        a_i, a_j, a_m, a_t, a_values, a_time_ptr
      For sample ending at time t, assignment history A_0...A_t is:
        prefix [: a_time_ptr[t + 1]]

    Saved NPZ files contain:
      - x, y
      - x_offsets, y_offsets
      - a_i, a_j, a_m, a_t, a_values, a_time_ptr
      - a_sample_end_t
      - a_sample_end_ptr
      - assignment_shape_step
    """
    T_x, N_x, C = X.shape
    T_a = int(assignment_data["T_total"][0])

    if assignment_data["assignment_shape_step"][0] != N_x:
        raise ValueError(
            f"Node dimension mismatch: X has N={N_x}, "
            f"assignment has N={assignment_data['assignment_shape_step'][0]}"
        )

    T = min(T_x, T_a)

    if T_x != T_a:
        print(f"[align] warning: X has T={T_x}, assignments have T={T_a}. Using first T={T} steps.")

    if T <= 0:
        raise ValueError("Aligned time dimension is zero. Check input tensors.")

    X = X[:T]
    assignment_data = trim_assignment_timeline(assignment_data, T_keep=T)

    print(f"[windows] input tensor shape: T={T}, N={N_x}, C={C}")
    print(
        f"[windows] assignment timeline: nnz={len(assignment_data['a_values'])}, "
        f"M={assignment_data['assignment_shape_step'][2]}"
    )

    x_offsets = np.sort(np.arange(-(seq_length_x - 1), 1, 1))
    y_offsets = np.sort(np.arange(y_start, seq_length_y + 1, 1))

    min_t = abs(min(x_offsets))
    max_t = T - abs(max(y_offsets))

    if max_t <= min_t:
        raise ValueError(
            f"Time series length T={T} is too short for "
            f"seq_length_x={seq_length_x}, seq_length_y={seq_length_y}, y_start={y_start}."
        )

    x_windows = []
    y_windows = []
    a_sample_end_t = []
    a_sample_end_ptr = []

    for t in range(min_t, max_t):
        x_windows.append(X[t + x_offsets, ...])
        y_windows.append(X[t + y_offsets, ...])

        # history is A_0 ... A_t
        a_sample_end_t.append(t)
        a_sample_end_ptr.append(int(assignment_data["a_time_ptr"][t + 1]))

    X_out = np.stack(x_windows, axis=0)
    Y_out = np.stack(y_windows, axis=0)
    a_sample_end_t = np.asarray(a_sample_end_t, dtype=np.int64)
    a_sample_end_ptr = np.asarray(a_sample_end_ptr, dtype=np.int64)

    print(f"[windows] all_x={X_out.shape}, all_y={Y_out.shape}, samples={X_out.shape[0]}")

    train_ratio, val_ratio, test_ratio = split_ratio
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("split_ratio must sum to 1.0")

    S = X_out.shape[0]
    num_train = int(np.round(S * train_ratio))
    num_test = int(np.round(S * test_ratio))
    num_val = S - num_train - num_test

    indices_train = slice(0, num_train)
    indices_val = slice(num_train, num_train + num_val)
    indices_test = slice(num_train + num_val, S)

    safe_mkdir(Path(output_dir))
    safe_mkdir(Path(intermediate_dir))

    x_offsets_reshaped = x_offsets.reshape(-1, 1)
    y_offsets_reshaped = y_offsets.reshape(-1, 1)

    common_payload = {
        "a_i": assignment_data["a_i"],
        "a_j": assignment_data["a_j"],
        "a_m": assignment_data["a_m"],
        "a_t": assignment_data["a_t"],
        "a_values": assignment_data["a_values"],
        "a_time_ptr": assignment_data["a_time_ptr"],
        "assignment_shape_step": assignment_data["assignment_shape_step"],
        "x_offsets": x_offsets_reshaped,
        "y_offsets": y_offsets_reshaped,
    }

    # Save all windows as intermediate tensors
    np.savez_compressed(
        Path(intermediate_dir) / "windows_all_with_assignments.npz",
        x=X_out,
        y=Y_out,
        a_sample_end_t=a_sample_end_t,
        a_sample_end_ptr=a_sample_end_ptr,
        **common_payload,
    )
    print(f"[windows] saved: {Path(intermediate_dir) / 'windows_all_with_assignments.npz'}")

    splits = {
        "train": (
            X_out[indices_train],
            Y_out[indices_train],
            a_sample_end_t[indices_train],
            a_sample_end_ptr[indices_train],
        ),
        "val": (
            X_out[indices_val],
            Y_out[indices_val],
            a_sample_end_t[indices_val],
            a_sample_end_ptr[indices_val],
        ),
        "test": (
            X_out[indices_test],
            Y_out[indices_test],
            a_sample_end_t[indices_test],
            a_sample_end_ptr[indices_test],
        ),
    }

    for split_name, (x_split, y_split, a_end_t_split, a_end_ptr_split) in splits.items():
        filepath = Path(output_dir) / f"{split_name}.npz"
        np.savez_compressed(
            filepath,
            x=x_split,
            y=y_split,
            a_sample_end_t=a_end_t_split,
            a_sample_end_ptr=a_end_ptr_split,
            **common_payload,
        )
        print(
            f"[{split_name}] saved: {filepath} -> "
            f"x={x_split.shape}, y={y_split.shape}, samples={x_split.shape[0]}"
        )


def process_single_dataset(
    dataset_name: str,
    csv_files: list[Path],
    assignment_files: list[Path],
    output_root: str | Path,
    epsilon: float = 500.0,
    sigma: float | None = None,
    use_knn: bool = False,
    knn_k: int = 10,
    seq_length_x: int = 12,
    seq_length_y: int = 12,
    y_start: int = 1,
    split_ratio: tuple[float, float, float] = (0.7, 0.1, 0.2),
) -> None:
    """
    Process one dataset and save everything to:
    output_root / dataset_name /
    """
    dataset_output_dir = safe_mkdir(Path(output_root) / dataset_name)
    adjacency_dir = safe_mkdir(dataset_output_dir / "adjacency_matrices")
    intermediate_dir = safe_mkdir(dataset_output_dir / "intermediate_tensors")

    reference_csv = csv_files[0]

    print(f"Processing dataset: {dataset_name}")
    print(f"Input CSV files: {len(csv_files)}")
    print(f"Input assignment files: {len(assignment_files)}")
    print(f"Reference CSV for adjacency: {reference_csv}")
    print(f"Output directory: {dataset_output_dir}")

    # Spatial adjacency
    create_spatial_adjacency_matrix(
        data_path=reference_csv,
        output_path=adjacency_dir / "adjacency_spatial.csv",
        sigma=sigma,
        epsilon=epsilon,
    )

    # Optional k-NN adjacency
    if use_knn:
        create_knn_adjacency_matrix(
            data_path=reference_csv,
            output_path=adjacency_dir / f"adjacency_knn_k{knn_k}.csv",
            k=knn_k,
        )

    # Full temporal tensor
    X_full, nodes = compile_sumo_temporal_tensor(
        filepaths=csv_files,
        output_filename=intermediate_dir / "temporal_tensor_full.npz",
    )
    np.savez_compressed(intermediate_dir / "nodes.npz", nodes=nodes.to_numpy())

    # Full assignment timeline
    assignment_data = compile_assignment_timeline(
        filepaths=assignment_files,
        output_filename=intermediate_dir / "assignment_timeline_full.npz",
    )

    # Train / val / test tensors with assignment history
    generate_wavenet_assignment_tensors(
        X=X_full,
        assignment_data=assignment_data,
        output_dir=dataset_output_dir,
        intermediate_dir=intermediate_dir,
        seq_length_x=seq_length_x,
        seq_length_y=seq_length_y,
        y_start=y_start,
        split_ratio=split_ratio,
    )


def process_all_networks_with_assignments(
    networks_root: str | Path = "data/networks",
    assignments_root: str | Path = "data/assignments",
    output_root: str | Path = "data/processed_networks_with_assignments",
    epsilon: float = 500.0,
    sigma: float | None = None,
    use_knn: bool = False,
    knn_k: int = 10,
    seq_length_x: int = 12,
    seq_length_y: int = 12,
    y_start: int = 1,
    split_ratio: tuple[float, float, float] = (0.7, 0.1, 0.2),
) -> None:
    """
    Process every dataset found in networks_root, matching it with assignment tensors
    found in assignments_root.

    Expected preferred layout:
        data/networks/<dataset_name>/*.csv
        data/assignments/<dataset_name>/*.pt

    Supported assignment counts per dataset:
    - 1 assignment file total
    - same number as CSV files (episode-wise)
    """
    datasets = discover_dataset_inputs(networks_root)
    assignment_sets = discover_assignment_inputs(assignments_root)

    print(f"Found {len(datasets)} dataset(s) in {networks_root}")
    print(f"Found {len(assignment_sets)} assignment dataset(s) in {assignments_root}")

    for dataset_name, csv_files in datasets.items():
        if dataset_name not in assignment_sets:
            raise ValueError(
                f"Missing assignments for dataset '{dataset_name}'. "
                f"Expected folder or files under: {Path(assignments_root) / dataset_name}"
            )

        assignment_files = assignment_sets[dataset_name]

        if len(assignment_files) not in (1, len(csv_files)):
            raise ValueError(
                f"Dataset '{dataset_name}' has {len(csv_files)} CSV files but "
                f"{len(assignment_files)} assignment files. Supported cases: "
                f"1 total assignment file or exactly one assignment per CSV."
            )

        process_single_dataset(
            dataset_name=dataset_name,
            csv_files=csv_files,
            assignment_files=assignment_files,
            output_root=output_root,
            epsilon=epsilon,
            sigma=sigma,
            use_knn=use_knn,
            knn_k=knn_k,
            seq_length_x=seq_length_x,
            seq_length_y=seq_length_y,
            y_start=y_start,
            split_ratio=split_ratio,
        )


if __name__ == "__main__":
    process_explicit_files(
        network_csv="data/networks/ingolstadt_770_assignments.csv",
        assignment_pt="data/assignments/assignment_M10_T1032.pt",
        output_root="data/processed_networks_with_assignments",
        epsilon=500.0,
        sigma=None,
        use_knn=False,       # set True if you also want k-NN adjacency
        knn_k=10,
        seq_length_x=12,
        seq_length_y=1,     # set to 1 if you want only q_{t+1} as target
        y_start=1,
        split_ratio=(0.7, 0.1, 0.2),
    )
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
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
    has_steps = any(col.startswith("Step ") for col in header.columns)
    return has_coords and has_steps


def infer_dataset_name_from_file(file_path: Path) -> str:
    """
    Fallback naming if CSV files are placed directly in networks/ instead of subfolders.
    Example:
    traffic_heatmap_data_ep1(in).csv -> ingolstadt_770
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
        step_columns = [col for col in df.columns if col.startswith("Step ")]
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

        X_ep = df[step_columns].to_numpy()          # (N, T)
        X_ep_tensor = np.expand_dims(X_ep.T, axis=-1)  # (T, N, 1)

        tensor_list.append(X_ep_tensor)
        print(f"[tensor] processed: {path.name}, shape={X_ep_tensor.shape}")

    X_full = np.concatenate(tensor_list, axis=0)

    safe_mkdir(Path(output_filename).parent)
    np.savez_compressed(output_filename, X=X_full, nodes=canonical_nodes.to_numpy())

    print(f"[tensor] saved: {output_filename}, shape={X_full.shape}")

    return X_full, canonical_nodes


def generate_wavenet_tensors(
    X: np.ndarray,
    output_dir: str | Path,
    intermediate_dir: str | Path,
    seq_length_x: int = 12,
    seq_length_y: int = 12,
    y_start: int = 1,
    split_ratio: tuple[float, float, float] = (0.7, 0.1, 0.2),
) -> None:
    """
    Generate Graph WaveNet windows and save:
    - all windows to intermediate_tensors/
    - train/val/test splits to dataset output dir
    """
    T, N, C = X.shape
    print(f"[windows] input tensor shape: T={T}, N={N}, C={C}")

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

    for t in range(min_t, max_t):
        x_windows.append(X[t + x_offsets, ...])
        y_windows.append(X[t + y_offsets, ...])

    X_out = np.stack(x_windows, axis=0)
    Y_out = np.stack(y_windows, axis=0)

    print(f"[windows] all_x={X_out.shape}, all_y={Y_out.shape}")

    train_ratio, val_ratio, test_ratio = split_ratio
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("split_ratio must sum to 1.0")

    S = X_out.shape[0]
    num_train = int(np.round(S * train_ratio))
    num_test = int(np.round(S * test_ratio))
    num_val = S - num_train - num_test

    x_train = X_out[:num_train]
    y_train = Y_out[:num_train]

    x_val = X_out[num_train:num_train + num_val]
    y_val = Y_out[num_train:num_train + num_val]

    x_test = X_out[num_train + num_val:]
    y_test = Y_out[num_train + num_val:]

    safe_mkdir(Path(output_dir))
    safe_mkdir(Path(intermediate_dir))

    x_offsets_reshaped = x_offsets.reshape(-1, 1)
    y_offsets_reshaped = y_offsets.reshape(-1, 1)

    # Save all windows as intermediate tensors
    np.savez_compressed(
        Path(intermediate_dir) / "windows_all.npz",
        x=X_out,
        y=Y_out,
        x_offsets=x_offsets_reshaped,
        y_offsets=y_offsets_reshaped,
    )
    print(f"[windows] saved: {Path(intermediate_dir) / 'windows_all.npz'}")

    splits = {
        "train": (x_train, y_train),
        "val": (x_val, y_val),
        "test": (x_test, y_test),
    }

    for split_name, (x_split, y_split) in splits.items():
        filepath = Path(output_dir) / f"{split_name}.npz"
        np.savez_compressed(
            filepath,
            x=x_split,
            y=y_split,
            x_offsets=x_offsets_reshaped,
            y_offsets=y_offsets_reshaped,
        )
        print(f"[{split_name}] saved: {filepath} -> x={x_split.shape}, y={y_split.shape}")


def process_single_dataset(
    dataset_name: str,
    csv_files: list[Path],
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

    print("\n" + "=" * 100)
    print(f"Processing dataset: {dataset_name}")
    print(f"Input files: {len(csv_files)}")
    print(f"Reference CSV for adjacency: {reference_csv}")
    print(f"Output directory: {dataset_output_dir}")
    print("=" * 100)

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

    # Train / val / test tensors
    generate_wavenet_tensors(
        X=X_full,
        output_dir=dataset_output_dir,
        intermediate_dir=intermediate_dir,
        seq_length_x=seq_length_x,
        seq_length_y=seq_length_y,
        y_start=y_start,
        split_ratio=split_ratio,
    )


def process_all_networks(
    networks_root: str | Path = "data/networks",
    output_root: str | Path = "data/processed_networks",
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
    Process every dataset found in networks_root.
    """
    datasets = discover_dataset_inputs(networks_root)
    print(f"Found {len(datasets)} dataset(s) in {networks_root}")

    for dataset_name, csv_files in datasets.items():
        process_single_dataset(
            dataset_name=dataset_name,
            csv_files=csv_files,
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
    process_all_networks(
        networks_root="data/networks",
        output_root="data/processed_networks",
        epsilon=500.0,
        sigma=None,
        use_knn=False,       # set True if you also want k-NN adjacency
        knn_k=10,
        seq_length_x=12,
        seq_length_y=12,
        y_start=1,
        split_ratio=(0.7, 0.1, 0.2),
    )


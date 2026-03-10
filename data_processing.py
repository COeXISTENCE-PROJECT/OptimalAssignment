import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import kneighbors_graph
import glob
import os


def create_spatial_adjacency_matrix(data_path, output_path, sigma=None, epsilon=500.0):
    """
    Creates a weighted adjacency matrix based on spatial coordinates.

    Args:
    - data_path: Path to the input CSV file.
    - output_path: Path to save the resulting matrix.
    - sigma: RBF kernel scaling parameter (defaults to sample std dev).
    - epsilon: Distance threshold to enforce graph sparsity.
    """
    df = pd.read_csv(data_path, index_col=0)

    # Extract coordinates
    coords = df[['coord_x', 'coord_y']].values

    # Calculate symmetric Euclidean distance matrix
    dist_matrix = squareform(pdist(coords, metric='euclidean'))

    # Estimate variance if not provided
    if sigma is None:
        sigma = np.std(dist_matrix)
        if sigma == 0:
            sigma = 1e-6

    # Apply Gaussian kernel
    A = np.exp(- (dist_matrix ** 2) / (sigma ** 2))

    # Apply distance threshold
    A[dist_matrix > epsilon] = 0.0

    # Remove self-loops
    np.fill_diagonal(A, 0.0)

    # Save to CSV
    adj_df = pd.DataFrame(A, index=df.index, columns=df.index)
    adj_df.to_csv(output_path)

    print(f"Number of nodes |V|: {A.shape[0]}")
    print(f"Number of edges |E| (non-zero elements): {np.count_nonzero(A)}")

    return adj_df


def create_knn_adjacency_matrix(data_path, output_path, k=10):
    """
    Creates a symmetric, binary k-NN adjacency matrix for given spatial coordinates.
    """
    df = pd.read_csv(data_path, index_col=0)
    coords = df[['coord_x', 'coord_y']].values
    n_nodes = coords.shape[0]

    # Compute directed k-NN graph
    A_knn_directed = kneighbors_graph(
        X=coords,
        n_neighbors=k,
        mode='connectivity',
        include_self=False,
        n_jobs=-1
    )

    # Symmetrize matrix using logical OR
    A_knn_sym = A_knn_directed.maximum(A_knn_directed.T)

    # Export to dense format
    adj_knn_df = pd.DataFrame(A_knn_sym.toarray(), index=df.index, columns=df.index)
    adj_knn_df.to_csv(output_path)

    # Calculate basic metrics
    degrees = np.array(A_knn_sym.sum(axis=1)).flatten()
    density = A_knn_sym.nnz / (n_nodes * (n_nodes - 1))

    print("k-NN Matrix Analysis (Symmetrized OR)")
    print(f"|V| (Nodes)        = {n_nodes}")
    print(f"|E| (Edges)        = {A_knn_sym.nnz}")
    print(f"Sparsity S         = {(1.0 - density):.4f}")
    print(f"Mean degree        = {np.mean(degrees):.2f}")
    print(f"Max degree         = {np.max(degrees)}")
    print(f"Min degree         = {np.min(degrees)}")

    return A_knn_sym


def compile_sumo_temporal_tensor(file_pattern: str, output_filename: str = "traffic_dataset_clean.npz"):
    """
    Compiles multiple simulation CSV files into a single spatiotemporal tensor.
    Returns the multidimensional time series of traffic dynamics.
    """
    filepaths = sorted(glob.glob(file_pattern))
    if not filepaths:
        raise ValueError(f"No files found for pattern: {file_pattern}")

    print(f"Started processing {len(filepaths)} files...")

    tensor_list = []
    canonical_nodes = None

    for path in filepaths:
        df = pd.read_csv(path, index_col=0)

        # Ensure consistent node ordering
        if canonical_nodes is None:
            canonical_nodes = df.index
        else:
            df = df.reindex(canonical_nodes)

        # Extract time steps
        step_columns = [col for col in df.columns if col.startswith('Step ')]
        X_ep = df[step_columns].values

        # Reshape to (T, N, C) structure
        X_ep_transposed = X_ep.T
        X_ep_tensor = np.expand_dims(X_ep_transposed, axis=-1)

        tensor_list.append(X_ep_tensor)
        print(f"  -> Processed {path}, shape: {X_ep_tensor.shape}")

    # Concatenate along the time dimension
    X_full = np.concatenate(tensor_list, axis=0)

    print("\n--- Finished ---")
    print("Assembled multidimensional input signal tensor X")

    # Export tensor and node labels
    np.savez_compressed(
        output_filename,
        X=X_full,
        nodes=canonical_nodes.values
    )
    print(f"Saved clean dataset to: {output_filename}")

    return X_full, canonical_nodes


def split_time_series_data(X_all, Y_all, train_ratio=0.8):
    """
    Sequentially splits the dataset into training and testing sets, preserving chronology.
    """
    S_total = len(X_all)

    # Determine split index
    split_idx = int(S_total * train_ratio)

    X_train, X_test = X_all[:split_idx], X_all[split_idx:]
    Y_train, Y_test = Y_all[:split_idx], Y_all[split_idx:]

    print("=== DIMENSION VERIFICATION AFTER SPLIT ===")
    print(f"Total training windows (S_total): {S_total}")
    print(f"Split index: {split_idx}")
    print("-" * 40)
    print(f"X_train : {X_train.shape}")
    print(f"Y_train : {Y_train.shape}")
    print("-" * 40)
    print(f"X_test  : {X_test.shape}")
    print(f"Y_test  : {Y_test.shape}")
    print("==========================================")

    return X_train, Y_train, X_test, Y_test


def generate_wavenet_tensors(
        X: np.ndarray,
        output_dir: str,
        seq_length_x: int = 12,
        seq_length_y: int = 12,
        y_start: int = 1,
        split_ratio: tuple = (0.7, 0.1, 0.2)
):
    """
    Transforms a multivariate time series into sliding windows for Graph WaveNet.
    Generates input windows X_out (S x T_in x N x C) and target windows Y_out (S x T_out x N x C).
    """
    T, N, C = X.shape
    print(f"Input tensor X dimensions: T={T}, N={N}, C={C}")

    # Define relative time offsets
    x_offsets = np.sort(np.arange(-(seq_length_x - 1), 1, 1))
    y_offsets = np.sort(np.arange(y_start, (seq_length_y + 1), 1))

    # Define valid range for index t to avoid Out of Bounds errors
    min_t = abs(min(x_offsets))
    max_t = T - abs(max(y_offsets))

    if max_t <= min_t:
        raise ValueError(f"Time series length T={T} is too short to generate windows "
                         f"of sizes T_in={seq_length_x} and T_out={seq_length_y}.")

    print(f"Generating windows for t in [{min_t}, {max_t - 1}]...")

    x_windows, y_windows = [], []

    # Extract time windows
    for t in range(min_t, max_t):
        x_windows.append(X[t + x_offsets, ...])
        y_windows.append(X[t + y_offsets, ...])

    X_out = np.stack(x_windows, axis=0)
    Y_out = np.stack(y_windows, axis=0)

    S = X_out.shape[0]
    print(f"Generated S={S} (x, y) pairs.")
    print(f"Input tensor dimension: {X_out.shape}")
    print(f"Output tensor dimension: {Y_out.shape}")

    # Sequential train/val/test split
    train_ratio, val_ratio, test_ratio = split_ratio
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), "Sum of split ratios must be 1.0"

    num_test = int(np.round(S * test_ratio))
    num_train = int(np.round(S * train_ratio))
    num_val = S - num_train - num_test

    x_train, y_train = X_out[:num_train], Y_out[:num_train]
    x_val, y_val = X_out[num_train:num_train + num_val], Y_out[num_train:num_train + num_val]
    x_test, y_test = X_out[-num_test:], Y_out[-num_test:]

    # Export splits to .npz files
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    splits = {
        "train": (x_train, y_train),
        "val": (x_val, y_val),
        "test": (x_test, y_test)
    }

    # Reshape offsets to include a feature dimension: (seq_len, 1)
    x_offsets_reshaped = x_offsets.reshape(-1, 1)
    y_offsets_reshaped = y_offsets.reshape(-1, 1)

    for split_name, (x_split, y_split) in splits.items():
        filepath = os.path.join(output_dir, f"{split_name}.npz")
        np.savez_compressed(
            filepath,
            x=x_split,
            y=y_split,
            x_offsets=x_offsets_reshaped,
            y_offsets=y_offsets_reshaped
        )
        print(f"Saved {split_name}.npz -> x: {x_split.shape}, y: {y_split.shape}")
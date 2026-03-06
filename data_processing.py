import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import kneighbors_graph
import glob
import numpy as np
import os



def create_spatial_adjacency_matrix(data_path, output_path, sigma=None, epsilon=500.0):
    """
    Tworzy ważoną macierz sąsiedztwa na podstawie współrzędnych przestrzennych.

    Argumenty:
    - data_path: ścieżka do pliku CSV z danymi.
    - output_path: ścieżka do zapisu wynikowej macierzy.
    - sigma: parametr skalujący jądra RBF. Domyślnie odchylenie standardowe z próby.
    - epsilon: próg odległości - wymusza rzadkość (sparsity) grafu.
    """
    # 1. Wczytanie danych z założeniem, że pierwsza kolumna to zbiór wierzchołków V
    df = pd.read_csv(data_path, index_col=0)

    # Wyciągnięcie macierzy współrzędnych X \in R^{n \times 2}
    coords = df[['coord_x', 'coord_y']].values

    # 2. Obliczenie symetrycznej macierzy odległości euklidesowych D (D_{ij} = ||x_i - x_j||_2)
    dist_matrix = squareform(pdist(coords, metric='euclidean'))

    # 3. Estymacja wariancji, jeśli nie została podana z góry
    if sigma is None:
        sigma = np.std(dist_matrix)
        if sigma == 0:  # Zabezpieczenie na przypadek zdegenerowany
            sigma = 1e-6

    # 4. Aplikacja funkcji wagi krawędzi (Jądro Gaussa)
    A = np.exp(- (dist_matrix ** 2) / (sigma ** 2))

    # 5. Progowanie (usunięcie krawędzi słabych / dalekich)
    A[dist_matrix > epsilon] = 0.0

    # 6. Wyzerowanie przekątnej (brak pętli własnych)
    np.fill_diagonal(A, 0.0)

    # 7. Zapis do struktury z zachowaniem etykiet z oryginalnego pliku
    adj_df = pd.DataFrame(A, index=df.index, columns=df.index)
    adj_df.to_csv(output_path)

    print(f"Liczba wierzchołków |V|: {A.shape[0]}")
    print(f"Liczba krawędzi |E| (niezerowych elementów): {np.count_nonzero(A)}")

    return adj_df



def create_knn_adjacency_matrix(data_path, output_path, k=10):
    """
    Tworzy symetryczną, binarną macierz sąsiedztwa opartą na k-NN 
    dla podanych współrzędnych przestrzennych.
    """
    # 1. Wczytanie współrzędnych
    df = pd.read_csv(data_path, index_col=0)
    coords = df[['coord_x', 'coord_y']].values
    n_nodes = coords.shape[0]

    # 2. Wyznaczenie skierowanego grafu k-NN (zwraca csr_matrix)
    # mode='connectivity' oznacza macierz o wartościach w {0, 1}. 
    # Możliwe jest też mode='distance'.
    A_knn_directed = kneighbors_graph(
        X=coords,
        n_neighbors=k,
        mode='connectivity',
        include_self=False,
        n_jobs=-1  # Równoległe przetwarzanie drzew KD-Tree
    )

    # 3. Symetryzacja macierzy: A_sym = max(A, A^T) w przestrzeni operacji rzadkich
    # Użycie maksimum logicznego (OR) dla dwóch binarnych macierzy CSR.
    A_knn_sym = A_knn_directed.maximum(A_knn_directed.T)

    # 4. Eksport na zewnątrz (konwersja na formę gęstą - toarray())
    adj_knn_df = pd.DataFrame(A_knn_sym.toarray(), index=df.index, columns=df.index)
    adj_knn_df.to_csv(output_path)

    # 5. Podstawowa metryzacja
    degrees = np.array(A_knn_sym.sum(axis=1)).flatten()
    density = A_knn_sym.nnz / (n_nodes * (n_nodes - 1))

    print("=== Analiza Macierzy k-NN (Symmetrized OR) ===")
    print(f"|V| (Liczba wierzchołków)  = {n_nodes}")
    print(f"|E| (Niezerowe krawędzie) = {A_knn_sym.nnz}")
    print(f"Sparsity S                  = {(1.0 - density):.4f}")
    print(f"Średni stopień              = {np.mean(degrees):.2f}")
    print(f"Maksymalny stopień          = {np.max(degrees)}")
    print(f"Minimalny stopień           = {np.min(degrees)}")
    print("==============================================")

    return A_knn_sym


def compile_sumo_temporal_tensor(file_pattern: str, output_filename: str = "traffic_dataset_clean.npz"):
    """
    Kompiluje wiele plików CSV z symulacji (epizodów) do jednego tensora czasoprzestrzennego.
    Zwraca tylko wielowymiarowy szereg czasowy dynamiki ruchu.

    Argumenty:
    - file_pattern: maska plików do wczytania, np. 'traffic_heatmap_data_ep*.csv'
    - output_filename: nazwa pliku wynikowego npz.
    """
    filepaths = sorted(glob.glob(file_pattern))
    if not filepaths:
        raise ValueError(f"Nie znaleziono plików dla wzorca: {file_pattern}")

    print(f"Rozpoczęto przetwarzanie {len(filepaths)} plików...")

    tensor_list = []
    canonical_nodes = None

    for path in filepaths:
        df = pd.read_csv(path, index_col=0)

        # 1. Gwarancja jednoznaczności wektora bazy \mathcal{V}
        if canonical_nodes is None:
            canonical_nodes = df.index
        else:
            # Rzutujemy strukturę na układ wierzchołków z pliku pierwotnego
            df = df.reindex(canonical_nodes)

        # 2. Rzutowanie na podprzestrzeń kroków czasowych
        step_columns = [col for col in df.columns if col.startswith('Step ')]

        # X_ep \in \mathbb{R}^{N \times T_{ep}}
        X_ep = df[step_columns].values

        # 3. Transformacja afiniczna do struktury modelu (T, N, C)
        X_ep_transposed = X_ep.T  # \mathbb{R}^{T_{ep} \times N}
        X_ep_tensor = np.expand_dims(X_ep_transposed, axis=-1)  # \mathbb{R}^{T_{ep} \times N \times 1}

        tensor_list.append(X_ep_tensor)
        print(f"  -> Przetworzono {path}, shape: {X_ep_tensor.shape}")

    # 4. Fuzja (konkatenacja) wzdłuż wymiaru T
    X_full = np.concatenate(tensor_list, axis=0)

    print("\n--- Zakończono ---")
    print(f"Złożono wielowymiarowy tensor sygnału wejściowego X \in \mathbb")

    # 5. Eksport - wyłącznie tensor stanów ruchu i etykiety topologiczne
    np.savez_compressed(
        output_filename,
        X=X_full,
        nodes=canonical_nodes.values
    )
    print(f"Zapisano czysty zbiór danych w pliku: {output_filename}")

    return X_full, canonical_nodes



def split_time_series_data(X_all, Y_all, train_ratio=0.8):
    """
    Dokonuje sekwencyjnego cięcia zbioru na część treningową i testową,
    zachowując chronologię (przyczynowość przyczynowo-skutkową w czasie).
    """
    S_total = len(X_all)

    # 1. Wyznaczenie punktu cięcia na trajektorii
    split_idx = int(S_total * train_ratio)

    # 2. Cięcie wzdłuż wymiaru próbek (axis=0)
    X_train, X_test = X_all[:split_idx], X_all[split_idx:]
    Y_train, Y_test = Y_all[:split_idx], Y_all[split_idx:]

    # Podsumowanie wymiarów (weryfikacja algebry tensorów)
    print("=== WERYFIKACJA WYMIARÓW PO PODZIALE ===")
    print(f"Całkowita liczba okien uczących (S_total): {S_total}")
    print(f"Cięcie na indeksie próbki: {split_idx}")
    print("-" * 40)
    print(f"X_train : {X_train.shape}")
    print(f"Y_train : {Y_train.shape}")
    print("-" * 40)
    print(f"X_test  : {X_test.shape}")
    print(f"Y_test  : {Y_test.shape}")
    print("========================================")

    return X_train, Y_train, X_test, Y_test


#generowanie danych treningowych z jednej symulacji

def generate_wavenet_tensors(
        X: np.ndarray,
        output_dir: str,
        seq_length_x: int = 12,
        seq_length_y: int = 12,
        y_start: int = 1,
        split_ratio: tuple = (0.7, 0.1, 0.2)
):
    """
    Transformuje wielowymiarowy szereg czasowy do postaci okien przesuwnych
    wymaganych przez architekturę Graph WaveNet.

    Niech X \in R^{T x N x C} będzie tensorem stanów wejściowych.
    Funkcja generuje tensory X_out \in R^{S x T_in x N x C} oraz Y_out \in R^{S x T_out x N x C},
    gdzie S to liczba wygenerowanych próbek, T_in = seq_length_x, T_out = seq_length_y.
    """

    T, N, C = X.shape
    print(f"Wymiary tensora wejściowego X: T={T}, N={N}, C={C}")

    # 1. Definicja wektorów przesunięć (offsetów) względem punktu t
    # x_offsets \in R^{seq_length_x}: [-11, -10, ..., 0] dla seq_length_x=12
    x_offsets = np.sort(np.arange(-(seq_length_x - 1), 1, 1))

    # y_offsets \in R^{seq_length_y}: [1, 2, ..., 12] dla seq_length_y=12 i y_start=1
    y_offsets = np.sort(np.arange(y_start, (seq_length_y + 1), 1))

    # 2. Wyznaczenie dziedziny dla indeksu t, aby uniknąć wyjścia poza zakres tablicy (Out of Bounds)
    min_t = abs(min(x_offsets))
    max_t = T - abs(max(y_offsets))

    if max_t <= min_t:
        raise ValueError(f"Długość szeregu T={T} jest zbyt mała, aby wygenerować okna "
                         f"o rozmiarach T_in={seq_length_x} i T_out={seq_length_y}.")

    print(f"Generowanie okien dla t \in [{min_t}, {max_t - 1}]...")

    x_windows, y_windows = [], []

    # 3. Ekstrakcja okien czasowych z tensora X
    for t in range(min_t, max_t):
        x_windows.append(X[t + x_offsets, ...])
        y_windows.append(X[t + y_offsets, ...])

    X_out = np.stack(x_windows, axis=0)
    Y_out = np.stack(y_windows, axis=0)

    S = X_out.shape[0]
    print(f"Wygenerowano S={S} par (x, y).")
    print(f"Wymiar tensora wejść: {X_out.shape}")
    print(f"Wymiar tensora wyjść: {Y_out.shape}")

    # 4. Sekwencyjny podział zbioru na podzbiory uczący, walidacyjny i testowy
    train_ratio, val_ratio, test_ratio = split_ratio
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), "Sum of split ratios must be 1.0"

    num_test = int(np.round(S * test_ratio))
    num_train = int(np.round(S * train_ratio))
    num_val = S - num_train - num_test

    x_train, y_train = X_out[:num_train], Y_out[:num_train]
    x_val, y_val = X_out[num_train:num_train + num_val], Y_out[num_train:num_train + num_val]
    x_test, y_test = X_out[-num_test:], Y_out[-num_test:]

    # 5. Eksport tensorów do plików .npz z zachowaniem struktury kluczy
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    splits = {
        "train": (x_train, y_train),
        "val": (x_val, y_val),
        "test": (x_test, y_test)
    }

    # Wymagane kształty offsetów to dodanie wymiaru: (seq_len, 1)
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
        print(f"Zapisano {split_name}.npz -> x: {x_split.shape}, y: {y_split.shape}")





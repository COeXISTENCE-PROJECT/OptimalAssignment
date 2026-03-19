#!/usr/bin/env python3
"""
test_repr_diagnostics.py

Skrypt diagnostyczny do sprawdzania poprawności działania klas:
- fuse
- PositionalEncoding
- PathEncoder
- AssignmentEncoder
- GRU_Representation
- LSTM_Representation
- AttentionRepresentation

Na prostym przykładzie danych sekwencyjnych / tensorów:
- sprawdza inicjalizację klas,
- sprawdza kształty wyjść,
- sprawdza NaN/Inf,
- sprawdza backward / gradient flow,
- sprawdza odporność na permutację agentów,
- sprawdza zachowanie dla pustych agentów,
- wyłapuje częste błędy reprezentacji tensorów.

Uruchomienie:
    python test_repr_diagnostics.py /sciezka/do/pliku_z_modelami.py

Przykład:
    python test_repr_diagnostics.py my_models.py
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import traceback
from pathlib import Path
from typing import Any, List, Tuple

import torch
import torch.nn as nn


# =========================
# Pomocnicze funkcje raportu
# =========================

def line(char: str = "-", n: int = 88) -> str:
    return char * n


def tensor_stats(x: torch.Tensor) -> str:
    if not isinstance(x, torch.Tensor):
        return f"not a tensor: {type(x)}"
    return (
        f"shape={tuple(x.shape)}, dtype={x.dtype}, device={x.device}, "
        f"contiguous={x.is_contiguous()}, "
        f"nan={torch.isnan(x).any().item()}, inf={torch.isinf(x).any().item()}"
    )


def has_bad_values(x: torch.Tensor) -> bool:
    return torch.isnan(x).any().item() or torch.isinf(x).any().item()


def print_header(title: str) -> None:
    print("\n" + line("="))
    print(title)
    print(line("="))


def print_subtitle(title: str) -> None:
    print("\n" + line("-"))
    print(title)
    print(line("-"))


def ok(msg: str) -> None:
    print(f"[OK]    {msg}")


def warn(msg: str) -> None:
    print(f"[WARN]  {msg}")


def fail(msg: str) -> None:
    print(f"[FAIL]  {msg}")


def exc_to_str(e: BaseException) -> str:
    return "".join(traceback.format_exception_only(type(e), e)).strip()


# =========================
# Ładowanie modułu użytkownika
# =========================

def load_module_from_path(path: str):
    path = str(Path(path).resolve())
    spec = importlib.util.spec_from_file_location("model.py", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Nie udało się załadować modułu z: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["user_models_module"] = module
    spec.loader.exec_module(module)
    return module


# =========================
# Dane testowe
# =========================

def build_simple_sequence_data(
    B: int = 2,
    T: int = 4,
    N: int = 3,
    M: int = 3,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Tworzy prosty tensor A_seq o kształcie (B, T, N, N, M).

    Interpretacja:
    - B: batch
    - T: kroki czasowe
    - N x N: macierz ścieżki agenta
    - M: liczba agentów
    """
    A = torch.zeros(B, T, N, N, M, dtype=dtype, device=device)

    for b in range(B):
        for t in range(T):
            for m in range(M):
                i = (b + t + m) % N
                j = (2 * b + t + m) % N
                A[b, t, i, j, m] = 1.0

                if (b + t + m) % 2 == 0:
                    i2 = (i + 1) % N
                    j2 = (j + 1) % N
                    A[b, t, i2, j2, m] = 1.0

    return A


def build_sequence_with_empty_agents(
    B: int = 2,
    T: int = 4,
    N: int = 3,
    M: int = 3,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Tworzy dane z nieaktywnymi agentami i jednym całkowicie pustym krokiem czasowym.
    """
    A = build_simple_sequence_data(B=B, T=T, N=N, M=M, dtype=dtype, device=device)

    # część agentów nieaktywna
    A[0, 1, :, :, 1] = 0.0
    A[1, 2, :, :, 2] = 0.0

    # cały krok czasowy pusty
    A[1, 3, :, :, :] = 0.0

    return A


def permute_agents(A_seq: torch.Tensor, perm: List[int]) -> torch.Tensor:
    return A_seq[..., perm]


# =========================
# Testy ogólne
# =========================

def test_class_exists(module, class_name: str) -> Tuple[bool, Any]:
    obj = getattr(module, class_name, None)
    if obj is None:
        fail(f"Brak klasy `{class_name}` w module.")
        return False, None
    ok(f"Znaleziono klasę `{class_name}`.")
    return True, obj


def zero_grads(model: nn.Module) -> None:
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()


def test_forward_and_backward(model: nn.Module, x, expected_shape=None, model_name="model"):
    model.train()
    zero_grads(model)

    try:
        out = model(x)
    except Exception as e:
        fail(f"{model_name}: błąd w forward(): {exc_to_str(e)}")
        return None

    if isinstance(out, tuple):
        main_out = out[0]
    else:
        main_out = out

    if not isinstance(main_out, torch.Tensor):
        fail(f"{model_name}: forward() nie zwrócił tensora ani krotki z tensorem.")
        return None

    ok(f"{model_name}: forward() działa. {tensor_stats(main_out)}")

    if expected_shape is not None:
        if tuple(main_out.shape) == tuple(expected_shape):
            ok(f"{model_name}: oczekiwany kształt {expected_shape} zgadza się.")
        else:
            fail(f"{model_name}: oczekiwany kształt {expected_shape}, otrzymano {tuple(main_out.shape)}.")

    if has_bad_values(main_out):
        fail(f"{model_name}: wyjście zawiera NaN/Inf.")
    else:
        ok(f"{model_name}: brak NaN/Inf w wyjściu.")

    loss = main_out.sum()
    try:
        loss.backward()
        grad_params = 0
        total_params = 0
        for p in model.parameters():
            total_params += 1
            if p.grad is not None:
                grad_params += 1
        ok(f"{model_name}: backward() działa. Gradienty dla {grad_params}/{total_params} parametrów.")
    except Exception as e:
        fail(f"{model_name}: błąd w backward(): {exc_to_str(e)}")

    return main_out


# =========================
# Testy poszczególnych klas
# =========================

def run_fuse_tests(module):
    print_subtitle("Test klasy fuse")

    exists, Fuse = test_class_exists(module, "fuse")
    if not exists:
        return

    B, T = 2, 4
    dim_Q, dim_A, out_dim = 8, 6, 10
    Q = torch.randn(B, T, dim_Q, requires_grad=True)
    A = torch.randn(B, T, dim_A, requires_grad=True)

    for method in ["concatenate", "Hadamard", "Attention"]:
        print(f"\nMetoda fuse = {method}")
        try:
            model = Fuse(dim_Q=dim_Q, dim_A=dim_A, output_dim=out_dim, method=method)
            ok(f"fuse({method}): inicjalizacja działa.")
        except Exception as e:
            fail(f"fuse({method}): błąd inicjalizacji: {exc_to_str(e)}")
            continue

        try:
            out = model(Q, A)
            ok(f"fuse({method}): forward działa. {tensor_stats(out)}")
            if tuple(out.shape) == (B, T, out_dim):
                ok(f"fuse({method}): poprawny kształt wyjścia {(B, T, out_dim)}.")
            else:
                fail(f"fuse({method}): zły kształt wyjścia {tuple(out.shape)}.")

            if has_bad_values(out):
                fail(f"fuse({method}): wyjście zawiera NaN/Inf.")
            else:
                ok(f"fuse({method}): brak NaN/Inf.")

            out.sum().backward()
            ok(f"fuse({method}): backward działa.")
        except Exception as e:
            fail(f"fuse({method}): błąd forward/backward: {exc_to_str(e)}")

    print("\nUwaga diagnostyczna:")
    print(
        "- Dla method='Attention' poprawne wywołanie MultiheadAttention powinno mieć argumenty:\n"
        "  self.attention(query=queries, key=keys, value=values)\n"
        "  a nie `kay=...` i `values=...`."
    )


def run_positional_encoding_tests(module):
    print_subtitle("Test klasy PositionalEncoding")

    exists, PositionalEncoding = test_class_exists(module, "PositionalEncoding")
    if not exists:
        return

    for d_model in [8, 7]:
        print(f"\nd_model = {d_model}")
        try:
            pe = PositionalEncoding(d_model=d_model, max_len=16)
            ok(f"PositionalEncoding(d_model={d_model}): inicjalizacja działa.")
        except Exception as e:
            fail(f"PositionalEncoding(d_model={d_model}): błąd inicjalizacji: {exc_to_str(e)}")
            continue

        x = torch.randn(2, 5, d_model)
        try:
            y = pe(x)
            ok(f"PositionalEncoding(d_model={d_model}): forward działa. {tensor_stats(y)}")
            if tuple(y.shape) == tuple(x.shape):
                ok("Kształt wyjścia zgadza się z wejściem.")
            else:
                fail(f"Zły kształt wyjścia: {tuple(y.shape)} vs {tuple(x.shape)}")
        except Exception as e:
            fail(f"PositionalEncoding(d_model={d_model}): błąd forward: {exc_to_str(e)}")

    print("\nUwaga diagnostyczna:")
    print(
        "- Implementacja sinus/cosinus zwykle wymaga ostrożności dla nieparzystego d_model.\n"
        "- Jeśli d_model jest nieparzyste, przypisanie do pe[:, 1::2] może mieć zły rozmiar."
    )


def run_path_encoder_tests(module):
    print_subtitle("Test klasy PathEncoder")

    exists, PathEncoder = test_class_exists(module, "PathEncoder")
    if not exists:
        return

    n_nodes = 3
    D = 12
    batch_agents = 5
    x = torch.randn(batch_agents, n_nodes * n_nodes, requires_grad=True)

    for hidden_size in [None, 16]:
        print(f"\nhidden_size = {hidden_size}")
        try:
            model = PathEncoder(
                n_nodes=n_nodes,
                path_embedding_dim=D,
                hidden_size=hidden_size,
                dropout=0.1,
            )
            ok("PathEncoder: inicjalizacja działa.")
        except Exception as e:
            fail(f"PathEncoder: błąd inicjalizacji: {exc_to_str(e)}")
            continue

        test_forward_and_backward(
            model=model,
            x=x,
            expected_shape=(batch_agents, D),
            model_name=f"PathEncoder(hidden_size={hidden_size})",
        )


def run_assignment_encoder_tests(module):
    print_subtitle("Test klasy AssignmentEncoder")

    exists, AssignmentEncoder = test_class_exists(module, "AssignmentEncoder")
    if not exists:
        return

    B, T, N, M = 2, 4, 3, 3
    embedding_size = 16
    path_embedding_dim = 8

    A_seq = build_simple_sequence_data(B=B, T=T, N=N, M=M)
    A_seq_empty = build_sequence_with_empty_agents(B=B, T=T, N=N, M=M)
    perm = [2, 0, 1]
    A_seq_perm = permute_agents(A_seq, perm)

    for method in ["sum", "attention_pool", "k_latent"]:
        print(f"\nmethod = {method}")
        try:
            model = AssignmentEncoder(
                n_nodes=N,
                embedding_size=embedding_size,
                method=method,
                path_embedding_dim=path_embedding_dim,
                agent_hidden_size=12,
                dropout=0.1,
                num_latents=4,
                num_heads=4,
            )
            ok(f"AssignmentEncoder(method={method}): inicjalizacja działa.")
        except Exception as e:
            fail(f"AssignmentEncoder(method={method}): błąd inicjalizacji: {exc_to_str(e)}")
            continue

        out = test_forward_and_backward(
            model=model,
            x=A_seq,
            expected_shape=(B, T, embedding_size),
            model_name=f"AssignmentEncoder(method={method})",
        )

        # Test pustych agentów / pustych kroków
        try:
            with torch.no_grad():
                out_empty = model(A_seq_empty)
            ok(
                f"AssignmentEncoder(method={method}) na danych z pustymi agentami działa. "
                f"{tensor_stats(out_empty)}"
            )
            if has_bad_values(out_empty):
                fail(f"AssignmentEncoder(method={method}): NaN/Inf dla pustych agentów.")
            else:
                ok(f"AssignmentEncoder(method={method}): brak NaN/Inf dla pustych agentów.")
        except Exception as e:
            fail(f"AssignmentEncoder(method={method}): błąd dla pustych agentów: {exc_to_str(e)}")

        # Test permutacji agentów
        if out is not None:
            try:
                model.eval()
                with torch.no_grad():
                    y1 = model(A_seq)
                    y2 = model(A_seq_perm)
                if torch.allclose(y1, y2, atol=1e-5, rtol=1e-5):
                    ok(f"AssignmentEncoder(method={method}): niezmienniczy na permutację agentów.")
                else:
                    warn(
                        f"AssignmentEncoder(method={method}): wynik zmienia się po permutacji agentów. "
                        "To może być błąd, jeśli oczekiwana jest permutacyjna niezmienniczość."
                    )
            except Exception as e:
                fail(f"AssignmentEncoder(method={method}): błąd testu permutacji: {exc_to_str(e)}")
            finally:
                model.train()

        # Test błędnego rozmiaru N
        try:
            bad_A = build_simple_sequence_data(B=B, T=T, N=N + 1, M=M)
            _ = model(bad_A)
            warn(
                f"AssignmentEncoder(method={method}): nie zgłosił błędu dla złego N "
                f"({N+1} zamiast {N})."
            )
        except Exception as e:
            ok(
                f"AssignmentEncoder(method={method}): poprawnie wykrywa zły wymiar przestrzenny. "
                f"({exc_to_str(e)})"
            )

    print("\nUwaga diagnostyczna:")
    print(
        "- Ta wersja testu zakłada poprawną implementację opartą o:\n"
        "  self.n_nodes\n"
        "  self.path_embedding_dim\n"
        "- Nie zakłada istnienia add_count_feature."
    )


def run_gru_representation_tests(module):
    print_subtitle("Test klasy GRU_Representation")

    exists, GRU_Representation = test_class_exists(module, "GRU_Representation")
    if not exists:
        return

    B, T, N, M = 2, 5, 3, 3
    A_seq = build_simple_sequence_data(B=B, T=T, N=N, M=M)
    lengths = torch.tensor([5, 3], dtype=torch.long)

    for method in ["sum", "attention_pool", "k_latent"]:
        print(f"\nassignment_method = {method}")
        try:
            model = GRU_Representation(
                n_nodes=N,
                embedding_size=16,
                hidden_size=12,
                dropout=0.1,
                num_layers=2,
                assignment_method=method,
                path_embedding_dim=8,
                agent_hidden_size=12,
                num_latents=4,
                num_heads=4,
            )
            ok(f"GRU_Representation(method={method}): inicjalizacja działa.")
        except Exception as e:
            fail(f"GRU_Representation(method={method}): błąd inicjalizacji: {exc_to_str(e)}")
            continue

        # Bez lengths
        try:
            out = model(A_seq)
            ok(f"GRU_Representation(method={method}): forward bez lengths działa. {tensor_stats(out)}")
            if tuple(out.shape) == (B, T, 12):
                ok("Poprawny kształt wyjścia.")
            else:
                fail(f"Zły kształt wyjścia: {tuple(out.shape)}")
        except Exception as e:
            fail(f"GRU_Representation(method={method}) bez lengths: {exc_to_str(e)}")

        # Z lengths
        try:
            out2 = model(A_seq, lengths=lengths)
            ok(f"GRU_Representation(method={method}): forward z lengths działa. {tensor_stats(out2)}")
            if tuple(out2.shape) == (B, T, 12):
                ok("Poprawny kształt wyjścia dla packed sequence.")
            else:
                fail(f"Zły kształt wyjścia z lengths: {tuple(out2.shape)}")
        except Exception as e:
            fail(f"GRU_Representation(method={method}) z lengths: {exc_to_str(e)}")

        # Z dodatkowymi zwrotami
        try:
            out3, hidden, step_embeddings = model(
                A_seq,
                return_hidden=True,
                return_step_embeddings=True,
            )
            ok("GRU_Representation: zwracanie hidden i step_embeddings działa.")
            ok(f"output: {tensor_stats(out3)}")
            ok(f"hidden: {tensor_stats(hidden)}")
            ok(f"step_embeddings: {tensor_stats(step_embeddings)}")
        except Exception as e:
            fail(f"GRU_Representation: błąd przy return_hidden/return_step_embeddings: {exc_to_str(e)}")


def run_lstm_representation_tests(module):
    print_subtitle("Test klasy LSTM_Representation")

    exists, LSTM_Representation = test_class_exists(module, "LSTM_Representation")
    if not exists:
        return

    B, T, N, M = 2, 4, 3, 3
    input_size = N * N * M
    A_seq = build_simple_sequence_data(B=B, T=T, N=N, M=M)

    try:
        model = LSTM_Representation(
            input_size=input_size,
            embedding_size=16,
            hidden_size=10,
            dropout=0.1,
            num_layers=2,
        )
        ok("LSTM_Representation: inicjalizacja działa.")
    except Exception as e:
        fail(f"LSTM_Representation: błąd inicjalizacji: {exc_to_str(e)}")
        return

    test_forward_and_backward(
        model=model,
        x=A_seq,
        expected_shape=(B, T, 10),
        model_name="LSTM_Representation",
    )


def run_attention_representation_tests(module):
    print_subtitle("Test klasy AttentionRepresentation")

    exists, AttentionRepresentation = test_class_exists(module, "AttentionRepresentation")
    if not exists:
        return

    B, T, N, M = 2, 4, 3, 3
    input_size = N * N * M
    A_seq = build_simple_sequence_data(B=B, T=T, N=N, M=M)

    try:
        model = AttentionRepresentation(
            input_size=input_size,
            embedding_size=16,
            num_heads=4,
            dim_feedforward=32,
            dropout=0.1,
            num_layers=2,
        )
        ok("AttentionRepresentation: inicjalizacja działa.")
    except Exception as e:
        fail(f"AttentionRepresentation: błąd inicjalizacji: {exc_to_str(e)}")
        return

    test_forward_and_backward(
        model=model,
        x=A_seq,
        expected_shape=(B, T, 16),
        model_name="AttentionRepresentation",
    )

    print("\nTest tensora nieciągłego w pamięci")
    A_seq_noncontig = A_seq.permute(0, 2, 1, 3, 4).permute(0, 2, 1, 3, 4)
    print(f"contiguous={A_seq_noncontig.is_contiguous()}")

    try:
        y = model(A_seq_noncontig)
        ok(f"AttentionRepresentation działa dla non-contiguous input. {tensor_stats(y)}")
    except Exception as e:
        fail(
            "AttentionRepresentation nie działa dla non-contiguous input. "
            f"Prawdopodobnie problem z `.view(...)`: {exc_to_str(e)}"
        )

    print("\nUwaga diagnostyczna:")
    print(
        "- W forward() lepiej użyć:\n"
        "  A_seq_flat = A_seq.reshape(batch_size, T, -1)\n"
        "  zamiast `.view(...)`, bo `.view(...)` wymaga zgodności pamięci."
    )


# =========================
# Podsumowanie błędów konstrukcyjnych
# =========================

def print_known_static_findings():
    print_subtitle("Najbardziej prawdopodobne błędy konstrukcyjne w pokazanym kodzie")

    print("1. fuse(method='Attention')")
    print("   Jeśli nadal masz:")
    print("       output, _ = self.attention(query=queries, kay=keys, values=values)")
    print("   to powinno być:")
    print("       output, _ = self.attention(query=queries, key=keys, value=values)")
    print()

    print("2. PositionalEncoding")
    print("   Dla nieparzystego d_model przypisanie do kanałów cos może dać błąd rozmiaru.")
    print()

    print("3. AttentionRepresentation")
    print("   `.view(...)` może wyłożyć się dla non-contiguous tensorów.")
    print("   Bezpieczniej użyć `.reshape(...)`.")
    print()

    print("4. AssignmentEncoder")
    print("   Ten test zakłada wersję spójną z:")
    print("       self.n_nodes")
    print("       self.path_embedding_dim")
    print("   oraz bez add_count_feature.")
    print()

    print("5. GRU_Representation")
    print("   Jeśli w module brakuje importów:")
    print("       from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence")
    print("   to test z lengths zakończy się błędem.")
    print()


# =========================
# Main
# =========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "module_path",
        type=str,
        help="Ścieżka do pliku .py z klasami modelu.",
    )
    args = parser.parse_args()

    torch.manual_seed(7)
    torch.set_printoptions(precision=4, sci_mode=False)

    print_header("Ładowanie modułu użytkownika")
    try:
        module = load_module_from_path(args.module_path)
        ok(f"Załadowano moduł z: {args.module_path}")
    except Exception as e:
        fail(f"Błąd ładowania modułu: {exc_to_str(e)}")
        print(traceback.format_exc())
        sys.exit(1)

    print_known_static_findings()

    run_fuse_tests(module)
    run_positional_encoding_tests(module)
    run_path_encoder_tests(module)
    run_assignment_encoder_tests(module)
    run_gru_representation_tests(module)
    run_lstm_representation_tests(module)
    run_attention_representation_tests(module)

    print_header("Koniec diagnostyki")


if __name__ == "__main__":
    main()
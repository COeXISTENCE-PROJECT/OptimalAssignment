import pandas as pd

# # Wczytaj swoją nową, uproszczoną macierz
# df = pd.read_csv("simplified_traffic_matrix.csv")
# active_edges = df['final_edge'].unique()

# # Zapisz do formatu selekcji SUMO
# with open("active_edges.txt", "w") as f:
#     for edge in active_edges:
#         f.write(f"edge:{edge}\n")

# print(f"Utworzono plik selekcji dla {len(active_edges)} krawędzi.")


def policz_unikalne_krawedzie(nazwa_pliku, plik_wyjsciowy):
    unique_edges = set()

    try:
        with open(nazwa_pliku, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if line.startswith("edge:"):
                    # Pobieramy część po 'edge:'
                    raw_id = line[len("edge:") :]

                    # Odcinamy wszystko po '#' (jeśli występuje)
                    base_id = raw_id.split("#")[0]

                    # Usuwamy '-' z początku, jeśli istnieje
                    if base_id.startswith("-"):
                        base_id = base_id[1:]

                    if base_id:  # Dodajemy tylko jeśli id nie jest puste
                        unique_edges.add(base_id)
            with open(plik_wyjsciowy, "w", encoding="utf-8") as f_out:
                for edge in sorted(unique_edges):
                    f_out.write(f"edge:{edge}\n")

        print(
            f"Liczba unikalnych krawędzi w pliku '{nazwa_pliku}': {len(unique_edges)}"
        )
        return unique_edges

    except FileNotFoundError:
        print(f"Błąd: Plik '{nazwa_pliku}' nie został znaleziony.")
    except Exception as e:
        print(f"Wystąpił nieoczekiwany błąd: {e}")

# Wywołanie funkcji
policz_unikalne_krawedzie("super_selection.txt", "unique_edges.txt")

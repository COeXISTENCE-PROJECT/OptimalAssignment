import pandas as pd

# Wczytaj swoją nową, uproszczoną macierz
df = pd.read_csv("simplified_traffic_matrix.csv")
active_edges = df['final_edge'].unique()

# Zapisz do formatu selekcji SUMO
with open("active_edges.txt", "w") as f:
    for edge in active_edges:
        f.write(f"edge:{edge}\n")

print(f"Utworzono plik selekcji dla {len(active_edges)} krawędzi.")

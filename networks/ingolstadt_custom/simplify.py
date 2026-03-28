import pandas as pd
import json
import xml.etree.ElementTree as ET
from collections import defaultdict

# 1. Wczytanie danych
with open("edge_to_idx.json", "r") as f:
    edge_to_idx = json.load(f)

# Odwracamy mapowanie: indeks -> nazwa krawędzi
idx_to_edge = {v: k for k, v in edge_to_idx.items()}

agents_df = pd.read_csv("agents.csv")

# 2. Parsowanie sieci SUMO w celu znalezienia powiązań krawędź -> węzeł
# Używamy Twojego pliku .xml, który zawiera definicje krawędzi (edges)
tree = ET.parse("ing_sim.net.xml")
root = tree.getroot()

edge_to_junction = {}
for edge in root.findall("edge"):
    edge_id = edge.get("id")
    # W uproszczeniu: przypisujemy krawędź do jej węzła końcowego (to)
    # lub początkowego (from), aby zgrupować je w obrębie skrzyżowania.
    junction_id = edge.get("to")
    edge_to_junction[edge_id] = junction_id

# 3. Agregacja macierzy przepływów
# Tworzymy listę nowych połączeń (Junction_A -> Junction_B)
aggregated_flows = defaultdict(int)

for _, row in agents_df.iterrows():
    try:
        # Pobierz nazwy krawędzi na podstawie indeksów
        edge_origin_id = idx_to_edge[row["origin"]]
        edge_dest_id = idx_to_edge[row["destination"]]

        # Znajdź odpowiadające im skrzyżowania
        junc_origin = edge_to_junction.get(edge_origin_id)
        junc_dest = edge_to_junction.get(edge_dest_id)

        if junc_origin and junc_dest:
            aggregated_flows[(junc_origin, junc_dest)] += 1
    except KeyError:
        continue

# 4. Tworzenie wynikowej macierzy/listy
output_data = []
for (start, end), count in aggregated_flows.items():
    output_data.append(
        {"origin_junction": start, "dest_junction": end, "flow_volume": count}
    )

new_matrix_df = pd.DataFrame(output_data)
new_matrix_df.to_csv("aggregated_junction_matrix.csv", index=False)

print("Uproszczona macierz została zapisana do: aggregated_junction_matrix.csv")
print(f"Liczba unikalnych połączeń między skrzyżowaniami: {len(new_matrix_df)}")

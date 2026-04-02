import sumolib
import pandas as pd
import re
import xml.etree.ElementTree as ET
from collections import defaultdict

# --- KONFIGURACJA ---
OLD_NET_PATH = "ingolstadt_custom.net.xml"
NEW_NET_PATH = "ing_sim.net.xml"
CSV_INPUT = "data/traffic_heatmap_data_ep1.csv"
OUTPUT_FILE = "data/aggregated_junction_time_matrix.csv"


def strip_lane(edge_id):
    """Usuwa ID pasa, zostawiając ID krawędzi (np. edge_0 -> edge)"""
    edge_id = str(edge_id)
    if edge_id.startswith(":"):
        return re.sub(r"_\d+$", "", edge_id)
    return edge_id.split("_")[0]


def get_network_mappings():
    print("Analizowanie sieci i mapowanie topologii...")
    old_net = sumolib.net.readNet(OLD_NET_PATH)
    new_net = sumolib.net.readNet(NEW_NET_PATH)

    # 1. Mapowanie: Stara krawędź -> Nowa krawędź (na podstawie pozycji)
    old_to_new_edge = {}
    for old_edge in old_net.getEdges():
        old_id = old_edge.getID()
        if new_net.hasEdge(old_id):
            old_to_new_edge[old_id] = old_id
        else:
            shape = old_edge.getShape()
            mid = shape[len(shape) // 2]
            nearby = new_net.getNeighboringEdges(mid[0], mid[1], 5)  # 5m tolerancji
            if nearby:
                old_to_new_edge[old_id] = sorted(nearby, key=lambda x: x[1])[0][
                    0
                ].getID()

    # 2. Mapowanie: Nowa krawędź -> Węzeł docelowy (Junction TO)
    edge_to_junction = {}
    tree = ET.parse(NEW_NET_PATH)
    root = tree.getroot()
    for edge in root.findall("edge"):
        eid = edge.get("id")
        junc = edge.get("to")  # Przypisujemy do węzła "dokąd zmierza"
        if eid and junc:
            edge_to_junction[eid] = junc

    print(len(old_to_new_edge))
    print("-------------")
    print(len(edge_to_junction))
    return old_to_new_edge, edge_to_junction


def process_and_aggregate():
    # Pobierz mapowania
    old_to_new, edge_to_junc = get_network_mappings()

    # Wczytaj dane heatmapy
    print("Wczytywanie danych szeregów czasowych...")
    df = pd.read_csv(CSV_INPUT)
    id_col = df.columns[0]
    step_cols = [c for c in df.columns if "Step" in c]

    # KROK 1: Wyczyść ID pasów
    df["clean_old_id"] = df[id_col].apply(strip_lane)

    # KROK 2: Mapuj na nową sieć
    df["new_edge_id"] = df["clean_old_id"].map(old_to_new)

    # KROK 3: Mapuj na węzły (Junctions)
    df["target_junction"] = df["new_edge_id"].map(edge_to_junc)

    # Usuń wiersze, których nie udało się przypisać do nowej sieci
    initial_count = len(df)
    df = df.dropna(subset=["target_junction"])
    print(f"Zmapowano {len(df)} z {initial_count} wpisów.")

    # KROK 4: Agregacja - sumujemy przepływy dla każdego kroku czasowego w obrębie węzła
    print("Agregowanie danych do węzłów...")
    # Grupujemy po węźle docelowym i sumujemy wszystkie kolumny 'Step X'
    junction_matrix = df.groupby("target_junction")[step_cols].sum().reset_index()

    # Dodanie współrzędnych węzłów z nowej sieci dla wizualizacji
    new_net = sumolib.net.readNet(NEW_NET_PATH)

    def get_junc_pos(jid):
        node = new_net.getNode(jid)
        pos = node.getCoord()
        return pd.Series([pos[0], pos[1]])

    print("Pobieranie współrzędnych węzłów...")
    junction_matrix[["x", "y"]] = junction_matrix["target_junction"].apply(get_junc_pos)

    # Zapis wynikowy
    junction_matrix.to_csv(OUTPUT_FILE, index=False)
    print(f"Gotowe! Wynik zapisany w {OUTPUT_FILE}")
    print(f"Liczba węzłów w macierzy: {len(junction_matrix)}")


if __name__ == "__main__":
    process_and_aggregate()

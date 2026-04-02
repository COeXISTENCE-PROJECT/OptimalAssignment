import pandas as pd
import xml.etree.ElementTree as ET
import re

NET_FILE = "ing_sim.net.xml"
JUNCTION_MATRIX = "aggregated_junction_matrix.csv"
FLOW_DATA = "aggregated_ingolstadt_1033.csv"  # Ta, która ma już zsumowane kroki
OUTPUT_FILE = "corridor_flow_matrix_FIXED.csv"


def get_root(eid):
    return re.split(r"[#_]", str(eid))[0]


def run_fix():
    print("Wczytywanie sieci i danych...")
    tree = ET.parse(NET_FILE)
    root = tree.getroot()

    # 1. Mapujemy krawędzie w sieci do ich 'korzeni' (root_id)
    # network_edges[(from, to)] = [lista root_id]
    network_map = {}
    for edge in root.findall("edge"):
        u, v = edge.get("from"), edge.get("to")
        eid = edge.get("id")
        if u and v and not eid.startswith(":"):
            if (u, v) not in network_map:
                network_map[(u, v)] = set()
            network_map[(u, v)].add(get_root(eid))

    # 2. Wczytujemy Twoje dane (579 krawędzi)
    flow_df = pd.read_csv(FLOW_DATA)
    # Upewniamy się, że target_id to string i czyścimy go na wszelki wypadek
    flow_df["root_id"] = flow_df["target_id"].apply(get_root)
    step_cols = [c for c in flow_df.columns if "Step" in c]

    # Indeksujemy po root_id dla błyskawicznego dostępu
    flow_indexed = flow_df.set_index("root_id")[step_cols]

    # 3. Agregujemy korytarze z macierzy 173 relacji
    junc_df = pd.read_csv(JUNCTION_MATRIX)
    corridor_results = []

    print("Agregowanie korytarzy...")
    for _, row in junc_df.iterrows():
        u, v = str(row["origin_junction"]), str(row["dest_junction"])
        corridor_id = f"{u}_{v}"

        # Pobieramy wszystkie root_id, które łączą te dwa skrzyżowania
        roots_in_net = network_map.get((u, v), [])

        # Filtrujemy tylko te, które faktycznie mamy w danych flowów
        valid_roots = [r for r in roots_in_net if r in flow_indexed.index]

        if valid_roots:
            # Sumujemy wszystkie pasujące drogi dla tego korytarza
            combined_flow = flow_indexed.loc[valid_roots].sum()
            res = {"corridor_id": corridor_id, "matched_edges": len(valid_roots)}
            res.update(combined_flow.to_dict())
            corridor_results.append(res)
        else:
            # Pusty korytarz
            res = {"corridor_id": corridor_id, "matched_edges": 0}
            res.update({s: 0 for s in step_cols})
            corridor_results.append(res)

    # 4. Zapis
    final_df = pd.DataFrame(corridor_results)
    final_df.to_csv(OUTPUT_FILE, index=False)

    print(
        f"Gotowe! Średnia liczba dopasowanych krawędzi na korytarz: {final_df['matched_edges'].mean():.2f}"
    )
    print(f"Zapisano do: {OUTPUT_FILE}")


if __name__ == "__main__":
    run_fix()

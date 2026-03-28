import sumolib
import pandas as pd
import re
import json

# --- KONFIGURACJA PLIKÓW ---
OLD_NET_PATH = 'ingolstadt_custom.net.xml'
NEW_NET_PATH = 'ing_sim.net.xml'
CSV_INPUT = 'traffic_heatmap_data_ep1.csv'
EDGE_JSON = 'edge_to_idx.json'
OUTPUT_FILE = 'simplified_traffic_matrix.csv'

def get_base_id(edge_id):
    """Usuwa informację o pasie ruchu (np. edge_0 -> edge)"""
    edge_id = str(edge_id)
    # Obsługa krawędzi wewnętrznych (np. :123_0) i zwykłych (edge_0)
    if edge_id.startswith(':'):
        return re.sub(r'_\d+$', '', edge_id)
    return edge_id.split('_')[0]

def create_mapping():
    print("Wczytywanie sieci i generowanie mapowania...")
    old_net = sumolib.net.readNet(OLD_NET_PATH)
    new_net = sumolib.net.readNet(NEW_NET_PATH)
    
    mapping = {}
    all_old_edges = old_net.getEdges()
    
    for old_edge in all_old_edges:
        old_id = old_edge.getID()
        # Jeśli ID istnieje w nowej sieci - mapujemy bezpośrednio
        if new_net.hasEdge(old_id):
            mapping[old_id] = old_id
        else:
            # Jeśli krawędź zniknęła (zredukowana geometria), szukamy nowej krawędzi 
            # na podstawie środka geometrycznego starej krawędzi
            shape = old_edge.getShape()
            mid_p = shape[len(shape)//2]
            # Szukamy krawędzi w nowej sieci w promieniu 3 metrów
            nearby = new_net.getNeighboringEdges(mid_p[0], mid_p[1], 3)
            if nearby:
                # Wybieramy najbliższą pasującą krawędź
                mapping[old_id] = sorted(nearby, key=lambda x: x[1])[0][0].getID()
            else:
                mapping[old_id] = None # Krawędź całkowicie usunięta
    return mapping, new_net

def simplify_matrix():
    # 1. Pobierz mapowanie
    mapping, new_net = create_mapping()
    
    # 2. Wczytaj dane
    print("Przetwarzanie danych CSV...")
    df = pd.read_csv(CSV_INPUT)
    # Pierwsza kolumna w Twoim pliku nie ma nazwy (lub jest pusta), nazwijmy ją 'edge_id'
    id_col = df.columns[0]
    
    # 3. Krok 1: Agregacja pasów (np. 123_0 -> 123)
    df['edge_base'] = df[id_col].apply(get_base_id)
    
    # 4. Krok 2: Mapowanie do uproszczonej sieci (np. 123 -> 123_merged_id)
    df['final_edge'] = df['edge_base'].map(mapping)
    
    # Usuwamy dane, które nie pasują do nowej sieci (opcjonalne)
    df = df.dropna(subset=['final_edge'])
    
    # 5. Krok 3: Sumowanie przepływów (wszystkie kolumny Step X)
    step_cols = [c for c in df.columns if 'Step' in c]
    # Grupujemy po nowym ID krawędzi i sumujemy wartości
    simplified_df = df.groupby('final_edge')[step_cols].sum().reset_index()
    
    # 6. Krok 4: Dodanie nowych współrzędnych dla nowej sieci
    def get_coords(eid):
        edge = new_net.getEdge(eid)
        shape = edge.getShape()
        mid = shape[len(shape)//2]
        return pd.Series([mid[0], mid[1]])

    print("Aktualizacja współrzędnych...")
    simplified_df[['coord_x', 'coord_y']] = simplified_df['final_edge'].apply(get_coords)
    
    # Reorganizacja kolumn: ID, X, Y, Kroki...
    cols = ['final_edge', 'coord_x', 'coord_y'] + step_cols
    simplified_df = simplified_df[cols]
    
    # Zapis
    simplified_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Sukces! Skonsolidowano {len(df)} wpisów do {len(simplified_df)} krawędzi.")
    print(f"Wynik zapisano w: {OUTPUT_FILE}")

if __name__ == "__main__":
    simplify_matrix()

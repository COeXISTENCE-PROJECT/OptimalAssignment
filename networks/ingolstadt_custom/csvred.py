import pandas as pd
import re

# 1. Wczytanie danych
print("Wczytywanie pliku...")
df = pd.read_csv('ingolstadt_1033.csv')

# 2. Identyfikacja kolumn
id_col = df.columns[0]
step_cols = [c for c in df.columns if 'Step' in c]

def get_root_id(edge_id):
    """Usuwa wszystko po # i _ (np. '12345#1_0' -> '12345')"""
    if pd.isna(edge_id): return "unknown"
    return re.split(r'[#_]', str(edge_id))[0]

print("Agregacja danych...")
# Tworzymy czyste ID
root_id_series = df[id_col].apply(get_root_id)

# 3. Agregacja: bierzemy tylko kolumny Step i grupujemy po nowych ID
# To automatycznie porzuca kolumny coord_x i coord_y
final_df = df[step_cols].groupby(root_id_series).sum().reset_index()

# Zmiana nazwy kolumny indeksowej na czytelną
final_df.rename(columns={id_col: 'edge_id'}, inplace=True)

# 4. Zapis do pliku
output_name = 'final_aggregated_no_coords.csv'
final_df.to_csv(output_name, index=False)

print(f"Sukces! Plik zapisany jako: {output_name}")
print(f"Liczba unikalnych krawędzi: {len(final_df)}")

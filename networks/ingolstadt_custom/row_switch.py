import pandas as pd
import sys


def move_coords_to_end(input_file, output_file):
    try:
        # Wczytanie pliku CSV
        df = pd.read_csv(input_file)

        # Lista wszystkich kolumn
        cols = list(df.columns)

        # Kolumny, które chcemy przenieść
        to_move = ["coord_x", "coord_y"]

        # Sprawdzenie czy kolumny istnieją w pliku
        existing_to_move = [c for c in to_move if c in cols]

        if not existing_to_move:
            print("Nie znaleziono kolumn coord_x ani coord_y.")
            return

        # Utworzenie nowej kolejności:
        # Wszystkie kolumny OPRÓCZ tych z to_move + kolumny z to_move na końcu
        new_columns = [c for c in cols if c not in existing_to_move] + existing_to_move

        # Reorganizacja DataFrame
        df = df[new_columns]

        # Zapis do nowego pliku
        df.to_csv(output_file, index=False)
        print(f"Sukces! Kolumny przeniesione. Wynik zapisano w: {output_file}")

    except Exception as e:
        print(f"Wystąpił błąd: {e}")


if __name__ == "__main__":
    # Możesz podać nazwy plików tutaj lub jako argumenty konsoli
    input_csv = "aggregated_ingolstadt_1033.csv"  # zmień na swoją nazwę
    output_csv = "ai0.csv"

    move_coords_to_end(input_csv, output_csv)

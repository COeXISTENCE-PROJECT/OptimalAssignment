import csv
from collections import defaultdict


def agreguj_dane_krawedzi(plik_wejsciowy, plik_wyjsciowy):
    # Struktura: base_id -> { 'steps': [suma_step0, suma_step1, ...], 'coords': [suma_x, suma_y], 'count': liczba_wystapien }
    agregacja = defaultdict(lambda: {"steps": None, "coords": [0.0, 0.0], "count": 0})

    try:
        with open(plik_wejsciowy, "r", encoding="utf-8") as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)  # Pobranie nagłówka

            # Indeksy kolumn
            # target_edge to 0, coord_x i coord_y to dwie ostatnie
            idx_coords = [header.index("coord_x"), header.index("coord_y")]

            for row in reader:
                if not row:
                    continue

                # 1. Wyciągnięcie czystego ID (usuwamy '-' i to co po '#')
                raw_id = row[0]
                base_id = raw_id.split("#")[0]
                if base_id.startswith("-"):
                    base_id = base_id[1:]

                # 2. Wyciągnięcie wartości kroków (Step 0 do Step 49)
                # Zakładamy, że kroki są pomiędzy ID a współrzędnymi
                steps_data = [float(x) for x in row[1 : idx_coords[0]]]
                coords_data = [float(row[idx_coords[0]]), float(row[idx_coords[1]])]

                # 3. Agregacja (sumowanie kroków i przygotowanie do średniej współrzędnych)
                entry = agregacja[base_id]
                if entry["steps"] is None:
                    entry["steps"] = [0.0] * len(steps_data)

                for i in range(len(steps_data)):
                    entry["steps"][i] += steps_data[i]

                entry["coords"][0] += coords_data[0]
                entry["coords"][1] += coords_data[1]
                entry["count"] += 1

        # 4. Zapis do nowego pliku
        with open(plik_wyjsciowy, "w", encoding="utf-8", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)  # Zapisujemy ten sam nagłówek

            for base_id in sorted(agregacja.keys()):
                data = agregacja[base_id]

                # Obliczamy średnią współrzędnych
                avg_x = data["coords"][0] / data["count"]
                avg_y = data["coords"][1] / data["count"]

                # Składamy wiersz: ID, kroki..., x, y
                new_row = [base_id] + data["steps"] + [avg_x, avg_y]
                writer.writerow(new_row)

        print(f"Zakończono sukcesem. Zagregowano krawędzie do pliku: {plik_wyjsciowy}")

    except Exception as e:
        print(f"Wystąpił błąd: {e}")


# Uruchomienie skryptu
# Upewnij się, że nazwa pliku wejściowego zgadza się z Twoją rzeczywistą nazwą
agreguj_dane_krawedzi("ai0.csv", "zagregowane_dane2.csv")

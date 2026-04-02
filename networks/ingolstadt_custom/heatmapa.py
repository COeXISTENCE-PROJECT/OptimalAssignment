import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def generuj_heatmape(plik_csv, nazwa_obrazu="heatmapa_krawedzi.png"):
    try:
        # 1. Wczytanie danych
        df = pd.read_csv(plik_csv)

        # 2. Przygotowanie danych do heatmapy
        # Ustawiamy ID krawędzi jako indeks
        df.set_index(df.columns[0], inplace=True)

        # Wybieramy tylko kolumny "Step X" (pomijamy współrzędne)
        kolumny_step = [col for col in df.columns if col.startswith("Step")]
        df_steps = df[kolumny_step]

        # 3. Tworzenie wykresu
        plt.figure(figsize=(15, 10))  # Rozmiar dopasowany do dużej liczby kroków

        sns.heatmap(
            df_steps,
            cmap="YlOrRd",  # Kolory: żółty -> pomarańczowy -> czerwony
            cbar_kws={"label": "Natężenie / Wartość"},
            xticklabels=5,  # Pokaż co piąty krok na osi X, żeby było czytelnie
            yticklabels=True,  # Pokaż ID krawędzi na osi Y
        )

        plt.title("Heatmapa natężenia na krawędziach w czasie (Steps)", fontsize=15)
        plt.xlabel("Kroki czasowe (Steps)")
        plt.ylabel("ID Krawędzi")

        # 4. Zapis i wyświetlenie
        plt.tight_layout()
        plt.savefig(nazwa_obrazu)
        plt.show()

        print(f"Heatmapa została zapisana jako: {nazwa_obrazu}")

    except Exception as e:
        print(f"Błąd podczas tworzenia heatmapy: {e}")


generuj_heatmape("zagregowane_dane.csv")

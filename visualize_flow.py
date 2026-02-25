import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx
import plotly.express as px

class TrafficAnimator:
    def __init__(self, exp_id, edge_coords, G=None):
        """
        exp_id: ID eksperymentu
        edge_coords: słownik {edge_id: (x, y)}
        G: obiekt NetworkX (opcjonalnie)
        """
        self.exp_id = exp_id
        self.snapshots_dir = os.path.join("../results", exp_id, "snapshots")
        self.edge_coords = edge_coords
        self.G = G if G else self._build_dummy_graph()

    def _build_dummy_graph(self):
        """Tworzy szkielet grafu na podstawie dostępnych krawędzi."""
        G = nx.Graph()
        for edge_id, coords in self.edge_coords.items():
            G.add_node(edge_id, pos=coords)
        return G

    def create_mp4_animation(self, episode_num, output_name="traffic_map.mp4", fps=10):
        """Generuje film MP4 z obciążeniem sieci na mapie."""
        file_pattern = f"es_ep{episode_num}_"
        snaps = sorted([f for f in os.listdir(self.snapshots_dir) 
                       if f.startswith(file_pattern) and f.endswith('.csv')])

        if not snaps:
            print("Brak snapshotów do animacji.")
            return

        fig, ax = plt.subplots(figsize=(12, 10))
        pos = self.edge_coords

        # Przygotowanie skali kolorów
        cmap = plt.cm.YlOrRd
        
        print(f"Generowanie filmu z {len(snaps)} klatek...")

        def update(frame):
            ax.clear()
            file_path = os.path.join(self.snapshots_dir, snaps[frame])
            df = pd.read_csv(file_path)
            
            # Liczymy auta na każdej krawędzi
            counts = df['edge_id'].value_counts().to_dict()
            
            # Mapujemy kolory na krawędzie (węzły w naszym grafie uproszczonym)
            node_colors = [counts.get(str(node), 0) for node in self.G.nodes()]
            
            nx.draw(self.G, pos, 
                    node_color=node_colors, 
                    cmap=cmap, 
                    node_size=50, 
                    with_labels=False, 
                    ax=ax,
                    vmin=0, vmax=max(counts.values()) if counts else 10)
            
            ax.set_title(f"Exp: {self.exp_id} | Ep: {episode_num} | Snapshot: {frame}")
            ax.set_facecolor('#f0f0f0')

        ani = animation.FuncAnimation(fig, update, frames=len(snaps), interval=1000/fps)
        
        # Wymaga zainstalowanego ffmpeg: 'brew install ffmpeg' lub 'apt install ffmpeg'
        ani.save(os.path.join(self.snapshots_dir, output_name), writer='ffmpeg')
        plt.close()
        print(f"Film zapisany: {output_name}")

    def create_interactive_html(self, episode_num, max_edges=30):
        """Tworzy interaktywny wykres słupkowy z suwakiem czasu (HTML)."""
        file_pattern = f"es_ep{episode_num}_"
        snaps = sorted([f for f in os.listdir(self.snapshots_dir) 
                       if f.startswith(file_pattern) and f.endswith('.csv')])

        all_data = []
        print("Łączenie danych do HTML...")
        
        for i, snap_file in enumerate(snaps):
            df = pd.read_csv(os.path.join(self.snapshots_dir, snap_file))
            counts = df['edge_id'].value_counts().reset_index()
            counts.columns = ['edge_id', 'vehicle_count']
            counts['step'] = i  # Dodajemy numer kroku jako oś czasu
            all_data.append(counts)

        full_df = pd.concat(all_data)
        
        # Filtrujemy tylko top krawędzie, żeby wykres był czytelny
        top_edges = full_df.groupby('edge_id')['vehicle_count'].sum().nlargest(max_edges).index
        filtered_df = full_df[full_df['edge_id'].isin(top_edges)]

        fig = px.bar(filtered_df, 
                     x="edge_id", y="vehicle_count", 
                     animation_frame="step",
                     color="vehicle_count",
                     range_y=[0, full_df['vehicle_count'].max() + 2],
                     color_continuous_scale="Reds",
                     title=f"Obciążenie Top {max_edges} krawędzi w czasie")

        output_path = os.path.join(self.snapshots_dir, f"flow_slider_ep{episode_num}.html")
        fig.write_html(output_path)
        print(f"Interaktywny HTML zapisany: {output_path}")

# --- PRZYKŁAD UŻYCIA ---
if __name__ == "__main__":
    # Te dane musisz przekazać ze swojego głównego skryptu
    MY_EXP_ID = "wave_test6"
    
    # Przykładowe współrzędne (pobierz je ze swojego edge_coords)
    # edge_coords = { 'edge_1': (0,0), 'edge_2': (10,5), ... }
    # Tutaj wstawiasz swój słownik edge_coords
    
    # animator = TrafficAnimator(MY_EXP_ID, edge_coords)
    # animator.create_mp4_animation(episode_num=0)
    # animator.create_interactive_html(episode_num=0)
    pass
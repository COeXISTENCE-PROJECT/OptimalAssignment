import json

import numpy as np

DIR = "network/ingolstadt_custom"
NUM_CELLS = 195


class Decoder:
    """Turns a genotype into the assignment matrix A."""

    def __init__(self, scenario, num_steps=270):
        self.od_keys = scenario.od_keys
        self.departure_steps = scenario.departure_steps
        self.num_steps = num_steps
        with open(f"{DIR}/route_cells.json") as f:
            self.route_cells = json.load(f)

    def decode(self, genotype):
        """
        genotype: list of cluster ids (0-3), one per agent
        """
        A = np.zeros((self.num_steps, NUM_CELLS), dtype=np.float32)
        for agent, cluster in enumerate(genotype):
            cells = self.route_cells[f"{self.od_keys[agent]}|{cluster}"]
            np.add.at(A[self.departure_steps[agent]], cells, 0.1)
        return A
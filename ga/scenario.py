import ast
from dataclasses import dataclass

import numpy as np
import pandas as pd

DIR = "network/ingolstadt_custom"
STEP_SECONDS = 10
MASK_COLUMNS = ["mask_0", "mask_1", "mask_2", "mask_3"]


@dataclass
class Scenario:
    """Fixed demand of the scenario. Contains the origin-destination pairs, departure times, and allowed route clusters of each agent."""

    od_keys: list                # "origin_edge|destination_edge" of each agent
    departure_steps: np.ndarray  # 10-second step in which each agent departs
    allowed_routes: list                # allowed route clusters of each agent

    @property
    def num_agents(self):
        return len(self.od_keys)


def load_scenario():
    agents = pd.read_csv(f"{DIR}/agents.csv")
    with open(f"{DIR}/od_ingolstadt_custom.txt") as f:
        od = ast.literal_eval(f.read())
    masks = pd.read_csv(f"{DIR}/ingolstadt_custom_action_masks.csv")

    od_keys = [
        f"{od['origins'][o]}|{od['destinations'][d]}"
        for o, d in zip(agents["origin"], agents["destination"])
    ]
    departure_steps = (agents["start_time"] // STEP_SECONDS).to_numpy()

    allowed_by_od = {
        f"{row['origins']}|{row['destinations']}": [k for k, col in enumerate(MASK_COLUMNS) if row[col] == 1]
        for _, row in masks.iterrows()
    }
    allowed_routes = [allowed_by_od[key] for key in od_keys]

    return Scenario(od_keys=od_keys, departure_steps=departure_steps, allowed_routes=allowed_routes)
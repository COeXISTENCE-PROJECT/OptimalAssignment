import pandas as pd
import numpy as np

df = pd.read_csv("agents_old.csv")

scale_factor = 2
time_jitter = 100

dfs = []

for i in range(scale_factor):
    temp = df.copy()

    # Ensure unique IDs
    temp["id"] = temp["id"] + i * len(df)

    if i > 0:
        temp["start_time"] += np.random.randint(
            -time_jitter, time_jitter + 1, size=len(temp)
        )

    dfs.append(temp)

df_scaled = pd.concat(dfs, ignore_index=True)

df_scaled["start_time"] = df_scaled["start_time"].clip(lower=0)

df_scaled.to_csv("agents.csv", index=False)

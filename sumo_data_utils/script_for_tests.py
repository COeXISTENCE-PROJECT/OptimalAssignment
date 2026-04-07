# import os
# from pathlib import Path

# # Path to your assignments folder
# assign_dir = Path("data/assignments")

# # Counter for feedback
# renamed_count = 0

# print("Starting renaming process...")

# for file_path in assign_dir.glob("exp*.npy"):
#     old_name = file_path.name

#     # Check if it's already in the correct format (exp_XYZ.npy)
#     if "_" in old_name:
#         continue

#     # Extract the number: 'exp123.npy' -> '123'
#     # We replace 'exp' and '.npy' with nothing
#     try:
#         exp_num = old_name.replace("exp", "").replace(".npy", "")
#         new_name = f"exp_{exp_num}.npy"

#         # Define the new path
#         new_file_path = file_path.parent / new_name

#         # Rename the file
#         file_path.rename(new_file_path)
#         renamed_count += 1
#     except Exception as e:
#         print(f"Skipping {old_name}: {e}")

# print(f"Renaming complete. Total files updated: {renamed_count}")

import pandas as pd
import numpy as np

asa = np.load("data/assignments/exp_0.npy")
asa = pd.DataFrame(asa[0, :, :])
# mapping = pd.read_csv("hex_mapping.csv")
# TOTAL_NODES = len(mapping)
# print(f"Total expected nodes: {TOTAL_NODES}")
print(asa.max().max())

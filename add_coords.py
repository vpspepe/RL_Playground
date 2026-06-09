import numpy as np
import os
import glob

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../oelwanne/normalized_data"))
NPZ_PATH = os.path.join(SCRIPT_DIR, "offline_dataset.npz")

print("Post-processing dataset: adding valid_points coordinates...")

# 1. Load coordinates X, Y, Z from one of base files
base_files = glob.glob(os.path.join(DATA_DIR, "base_sim_freq_*Hz.npz"))
if not base_files:
    print("Error: base_sim_freq files not found in", DATA_DIR)
    exit(1)
    
base_data = np.load(base_files[0])
X = base_data["X"]
Y = base_data["Y"]
Z = base_data["Z"]
valid_points = np.stack([X, Y, Z], axis=1)

# 2. Load generated offline_dataset.npz
if not os.path.exists(NPZ_PATH):
    print("Error: offline_dataset.npz not found at", NPZ_PATH)
    exit(1)
    
dataset = dict(np.load(NPZ_PATH))

# 3. Add valid_points and save back
dataset["valid_points"] = valid_points
np.savez(NPZ_PATH, **dataset)

print("Successfully injected valid_points into dataset at", NPZ_PATH)

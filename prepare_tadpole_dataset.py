import os
import sys
import glob
import numpy as np
from collections import defaultdict

# ==========================================
# PATH SETUP (Relative to Script Location)
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../oelwanne/normalized_data"))
SAVE_PATH = os.path.join(SCRIPT_DIR, "offline_dataset.npz")
TADPOLE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../PaperModels/TadPole"))

# Add Tadpole path to sys.path
sys.path.append(TADPOLE_DIR)

from huggingface_hub import hf_hub_download
import torch
from tadpole.model.encoder import TadpoleEncoder

# ==========================================
# CONFIGURATION
# ==========================================
PAIRS_PER_SIM = 10
GRID_SIZE = 64
SEED = 42
np.random.seed(SEED)

def prepare_dataset():
    print(f"Script Directory: {SCRIPT_DIR}")
    print(f"Tadpole Directory: {TADPOLE_DIR}")
    print(f"Loading base simulations from {DATA_DIR}...")
    base_files = glob.glob(os.path.join(DATA_DIR, "base_sim_freq_*Hz.npz"))
    
    base_sims = {}
    X, Y, Z = None, None, None
    for bf in base_files:
        filename = os.path.basename(bf)
        try:
            freq = int(filename.split("_")[-1].replace("Hz.npz", ""))
            base_data = np.load(bf)
            base_sims[freq] = {
                "erp": float(base_data["erp"]),
                "frequency": float(base_data["frequency"]),
                "vel_abs": base_data["vel_abs"],
                "sti_abs": base_data["sti_abs"],
            }
            if X is None:
                X = base_data["X"]
                Y = base_data["Y"]
                Z = base_data["Z"]
        except Exception as e:
            print(f"Error parsing base sim {filename}: {e}")
            
    print(f"Found {len(base_sims)} base simulations.")

    # Get coordinate bounds from the static 10,000 Fibonacci points
    X_min, X_max = X.min(), X.max()
    Y_min, Y_max = Y.min(), Y.max()
    Z_min, Z_max = Z.min(), Z.max()
    print(f"Coordinates Bounds:")
    print(f"X: [{X_min:.2f}, {X_max:.2f}]")
    print(f"Y: [{Y_min:.2f}, {Y_max:.2f}]")
    print(f"Z: [{Z_min:.2f}, {Z_max:.2f}]")

    # Define voxel index mapper
    def map_to_indices(x, y, z):
        xn = (x - X_min) / (X_max - X_min + 1e-8)
        yn = (y - Y_min) / (Y_max - Y_min + 1e-8)
        zn = (z - Z_min) / (Z_max - Z_min + 1e-8)
        
        ix = np.clip((xn * (GRID_SIZE - 1)).astype(np.int32), 0, GRID_SIZE - 1)
        iy = np.clip((yn * (GRID_SIZE - 1)).astype(np.int32), 0, GRID_SIZE - 1)
        iz = np.clip((zn * (GRID_SIZE - 1)).astype(np.int32), 0, GRID_SIZE - 1)
        return ix, iy, iz

    # Pre-compute voxel index mapping for the 10,000 Fibonacci points
    ix, iy, iz = map_to_indices(X, Y, Z)

    # Voxelization function: maps flat 10k points to 64x64x64 structured 3D grid
    def voxelize(vel_diff, sti_diff):
        grid_val = np.zeros((2, GRID_SIZE, GRID_SIZE, GRID_SIZE), dtype=np.float32)
        grid_cnt = np.zeros((GRID_SIZE, GRID_SIZE, GRID_SIZE), dtype=np.float32)
        
        # Accumulate values using numpy.add.at
        np.add.at(grid_val[0], (ix, iy, iz), vel_diff)
        np.add.at(grid_val[1], (ix, iy, iz), sti_diff)
        np.add.at(grid_cnt, (ix, iy, iz), 1.0)
        
        # Average values where multiple points map to the same voxel cell
        mask = grid_cnt > 0
        grid_val[0][mask] /= grid_cnt[mask]
        grid_val[1][mask] /= grid_cnt[mask]
        
        return grid_val

    print(f"Loading differential simulations from {DATA_DIR}...")
    diff_files = glob.glob(os.path.join(DATA_DIR, "diff_*.npz"))
    
    diff_sims = []
    for df in diff_files:
        try:
            data = np.load(df)
            diff_sims.append({
                "frequency": float(data["frequency"]),
                "erp": float(data["erp"]),
                "position": data["position"],  # [x, y, z] in mm
                "m_d": float(data["m_d"]),     # mass in g
                "vel_diff": data["vel_diff"],  # 10000 points
                "sti_diff": data["sti_diff"],  # 10000 points
            })
        except Exception as e:
            print(f"Error loading diff sim {os.path.basename(df)}: {e}")
            
    print(f"Loaded {len(diff_sims)} differential simulations.")

    if not base_sims or not diff_sims:
        print("Error: No data found. Exiting.")
        return

    # ==========================================
    # INITIALIZE TADPOLE ENCODER
    # ==========================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    print("Downloading Tadpole weights from Hugging Face Hub...")
    try:
        weights_path = hf_hub_download(repo_id="thuerey-group/Tadpole", filename="tadpole_b_encoder.safetensors")
        print(f"Weights downloaded to: {weights_path}")
    except Exception as e:
        print(f"Error downloading weights: {e}")
        return

    print("Initializing Tadpole Encoder (Size: B)...")
    encoder = TadpoleEncoder(
        size="B",
        weight_encoder=weights_path,
        latent_type="mode", # Use deterministic latents (mean of the distribution)
    )
    encoder.to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    # ==========================================
    # PRE-COMPUTE LATENTS (COMPUTATIONAL OPTIMIZATION)
    # ==========================================
    print("\nPre-computing latents for differential simulations...")
    batch_grids = []
    for d in diff_sims:
        grid = voxelize(d["vel_diff"], d["sti_diff"])
        batch_grids.append(grid)
        
    batch_grids = np.array(batch_grids, dtype=np.float32)
    
    # Process in batches to prevent CUDA Out-Of-Memory while remaining fast
    batch_size = 32
    latents = []
    for start_idx in range(0, len(batch_grids), batch_size):
        end_idx = min(start_idx + batch_size, len(batch_grids))
        chunk_t = torch.tensor(batch_grids[start_idx:end_idx], dtype=torch.float32).to(device)
        with torch.no_grad():
            latent_maps = encoder(chunk_t)
            # Global Average Pooling over spatial dimensions (4x4x4)
            latent_zs = latent_maps.mean(dim=(2, 3, 4)).cpu().numpy()
        latents.append(latent_zs)
        if (start_idx + batch_size) % 256 == 0 or end_idx == len(batch_grids):
            print(f"Encoded {end_idx}/{len(batch_grids)} simulations...")

    latents = np.concatenate(latents, axis=0)
    for idx, d in enumerate(diff_sims):
        d["latent_z"] = latents[idx]

    # ==========================================
    # CONSTRUCT RL TRANSITIONS
    # ==========================================
    # Group differential simulations by frequency
    sims_by_freq = defaultdict(list)
    for sim in diff_sims:
        freq_key = int(round(sim["frequency"]))
        sims_by_freq[freq_key].append(sim)

    observations = []
    actions = []
    rewards = []
    next_observations = []
    terminals = []

    print("\nConstructing transitions (Active-to-Active pairs only)...")
    for freq, b_sim in base_sims.items():
        d_sims = sims_by_freq.get(freq, [])
        if not d_sims:
            print(f"No differential simulations found for frequency {freq}Hz. Skipping.")
            continue
            
        erp_base = b_sim["erp"]
        
        # Pair each active simulation with randomly chosen target active simulations
        for i, d_src in enumerate(d_sims):
            num_targets = min(PAIRS_PER_SIM, len(d_sims) - 1)
            if num_targets <= 0:
                continue
                
            indices = np.random.choice([idx for idx in range(len(d_sims)) if idx != i], 
                                       size=num_targets, 
                                       replace=False)
            
            # State vector: [z (1024d) || freq (1d) || position (3d) || mass (1d)] = 1029d
            state_src = np.concatenate([
                d_src["latent_z"],
                [freq],
                d_src["position"],
                [d_src["m_d"]]
            ], dtype=np.float32)
            
            for idx in indices:
                d_dst = d_sims[idx]
                state_dst = np.concatenate([
                    d_dst["latent_z"],
                    [freq],
                    d_dst["position"],
                    [d_dst["m_d"]]
                ], dtype=np.float32)
                
                action = np.array([
                    d_dst["position"][0] - d_src["position"][0],
                    d_dst["position"][1] - d_src["position"][1],
                    d_dst["position"][2] - d_src["position"][2],
                    d_dst["m_d"] - d_src["m_d"]
                ], dtype=np.float32)
                
                # Reward: Relative ERP decrease between source and target
                reward = float((d_src["erp"] - d_dst["erp"]) / erp_base)
                
                observations.append(state_src.copy())
                actions.append(action)
                rewards.append(reward)
                next_observations.append(state_dst)
                terminals.append(0)

    # Convert to numpy arrays
    observations = np.array(observations, dtype=np.float32)
    actions = np.array(actions, dtype=np.float32)
    rewards = np.array(rewards, dtype=np.float32)
    next_observations = np.array(next_observations, dtype=np.float32)
    terminals = np.array(terminals, dtype=np.int32)
    
    print(f"\nGenerated {len(observations)} transitions.")

    # ==========================================
    # NORMALIZATION
    # ==========================================
    print("\nNormalizing dataset...")
    obs_mean = observations.mean(axis=0)
    obs_std = observations.std(axis=0) + 1e-8
    
    observations_norm = (observations - obs_mean) / obs_std
    next_observations_norm = (next_observations - obs_mean) / obs_std
    
    action_maxs = np.max(np.abs(actions), axis=0) + 1e-8
    actions_norm = actions / action_maxs
    
    print("Dataset stats:")
    print(f"Observations shape: {observations_norm.shape}")
    print(f"Actions shape: {actions_norm.shape}")
    print(f"Obs Mean shape: {obs_mean.shape}")
    print(f"Obs Std shape: {obs_std.shape}")
    print(f"Action Maxs (scaling factors): {action_maxs}")
    
    valid_points = np.stack([X, Y, Z], axis=1)
    
    # Save the processed dataset
    np.savez(SAVE_PATH,
             observations=observations_norm,
             actions=actions_norm,
             rewards=rewards,
             next_observations=next_observations_norm,
             terminals=terminals,
             obs_mean=obs_mean,
             obs_std=obs_std,
             action_maxs=action_maxs,
             valid_points=valid_points,
             raw_observations=observations,
             raw_actions=actions)
             
    print(f"\nSuccessfully generated and saved dataset to {SAVE_PATH}!")
    print("\nTo train your offline RL agent on this dataset, run:")
    print("python train_custom_pytorch.py")

if __name__ == "__main__":
    prepare_dataset()

import os
import glob
import numpy as np
from collections import defaultdict

# ==========================================
# PATH SETUP (Relative to Script Location)
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../MarcoProject/oelwanne/normalized_data"))
SAVE_PATH = os.path.join(SCRIPT_DIR, "offline_dataset.npz")

# Number of random destination simulations to pair with each source simulation
# to keep the dataset size reasonable (around 40k-80k transitions)
PAIRS_PER_SIM = 10

# Seed for reproducibility
SEED = 42
np.random.seed(SEED)

def prepare_dataset():
    print(f"Script Directory: {SCRIPT_DIR}")
    print(f"Loading base simulations from {DATA_DIR}...")
    base_files = glob.glob(os.path.join(DATA_DIR, "base_sim_freq_*Hz.npz"))
    
    base_sims = {}
    for bf in base_files:
        filename = os.path.basename(bf)
        try:
            # Extract frequency from name, e.g. base_sim_freq_1253Hz.npz -> 1253
            freq = int(filename.split("_")[-1].replace("Hz.npz", ""))
            base_data = np.load(bf)
            base_sims[freq] = {
                "erp": float(base_data["erp"]),
                "frequency": float(base_data["frequency"]),
                "vel_abs": base_data["vel_abs"],
                "sti_abs": base_data["sti_abs"],
            }
        except Exception as e:
            print(f"Error parsing base sim {filename}: {e}")
            
    print(f"Found {len(base_sims)} base simulations.")

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
        print("Error: No data found. Please check if the path exists and contains .npz files:")
        print(f"Path: {DATA_DIR}")
        return

    # Group differential simulations by frequency
    sims_by_freq = defaultdict(list)
    for sim in diff_sims:
        freq_key = int(round(sim["frequency"]))
        sims_by_freq[freq_key].append(sim)

    # Lists to store transitions
    observations = []
    actions = []
    rewards = []
    next_observations = []
    terminals = []

    print("\nConstructing transitions...")
    for freq, b_sim in base_sims.items():
        d_sims = sims_by_freq.get(freq, [])
        if not d_sims:
            print(f"No differential simulations found for frequency {freq}Hz. Skipping.")
            continue
            
        erp_base = b_sim["erp"]
        
        # 1. Base-to-Active Transitions (placing the TMD from no-TMD state)
        # Base state: TMD parameters at [0, 0, 0] with mass 0.0
        pos_base = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        m_base = 0.0
        state_base = np.array([pos_base[0], pos_base[1], pos_base[2], m_base, freq], dtype=np.float32)
        
        for d in d_sims:
            # Transition: Base -> Active TMD configuration
            state_active = np.array([d["position"][0], d["position"][1], d["position"][2], d["m_d"], freq], dtype=np.float32)
            
            action = np.array([
                d["position"][0] - pos_base[0],
                d["position"][1] - pos_base[1],
                d["position"][2] - pos_base[2],
                d["m_d"] - m_base
            ], dtype=np.float32)
            
            # Reward: Percentage reduction in ERP
            reward = float((erp_base - d["erp"]) / erp_base)
            
            observations.append(state_base.copy())
            actions.append(action)
            rewards.append(reward)
            next_observations.append(state_active)
            terminals.append(0)

        # 2. Active-to-Active Transitions (moving the TMD)
        # Pair each simulation with random other simulations at the same frequency
        for i, d_src in enumerate(d_sims):
            # Select target simulations to pair with
            num_targets = min(PAIRS_PER_SIM, len(d_sims) - 1)
            if num_targets <= 0:
                continue
                
            indices = np.random.choice([idx for idx in range(len(d_sims)) if idx != i], 
                                       size=num_targets, 
                                       replace=False)
            
            state_src = np.array([d_src["position"][0], d_src["position"][1], d_src["position"][2], d_src["m_d"], freq], dtype=np.float32)
            
            for idx in indices:
                d_dst = d_sims[idx]
                state_dst = np.array([d_dst["position"][0], d_dst["position"][1], d_dst["position"][2], d_dst["m_d"], freq], dtype=np.float32)
                
                action = np.array([
                    d_dst["position"][0] - d_src["position"][0],
                    d_dst["position"][1] - d_src["position"][1],
                    d_dst["position"][2] - d_src["position"][2],
                    d_dst["m_d"] - d_src["m_d"]
                ], dtype=np.float32)
                
                # Reward: Relative ERP improvement (how much ERP decreased between source and target)
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
    # NORMALIZATION (Standard Best Practice)
    # ==========================================
    print("\nNormalizing dataset...")
    # Normalize observations (mean=0, std=1)
    obs_mean = observations.mean(axis=0)
    obs_std = observations.std(axis=0) + 1e-8
    
    observations_norm = (observations - obs_mean) / obs_std
    next_observations_norm = (next_observations - obs_mean) / obs_std
    
    # Normalize actions to range [-1.0, 1.0] by dividing by max absolute value of each dimension
    action_maxs = np.max(np.abs(actions), axis=0) + 1e-8
    actions_norm = actions / action_maxs
    
    print("Dataset stats:")
    print(f"Observations shape: {observations_norm.shape}")
    print(f"Actions shape: {actions_norm.shape}")
    print(f"Obs Mean: {obs_mean}")
    print(f"Obs Std: {obs_std}")
    print(f"Action Maxs (scaling factors): {action_maxs}")
    
    # Save the processed dataset
    # We save both raw and normalized values, plus the scaling factors so the evaluation script
    # can easily scale the actor outputs back to physical coordinates.
    np.savez(SAVE_PATH,
             observations=observations_norm,
             actions=actions_norm,
             rewards=rewards,
             next_observations=next_observations_norm,
             terminals=terminals,
             obs_mean=obs_mean,
             obs_std=obs_std,
             action_maxs=action_maxs,
             raw_observations=observations,
             raw_actions=actions)
             
    print(f"\nSuccessfully generated and saved dataset to {SAVE_PATH}!")
    print("\nTo train your offline RL agent on this dataset, run:")
    print("python train_custom_pytorch.py")

if __name__ == "__main__":
    prepare_dataset()

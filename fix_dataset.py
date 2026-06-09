import numpy as np
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NPZ_PATH = os.path.join(SCRIPT_DIR, "offline_dataset.npz")

print("Loading dataset...")
data = dict(np.load(NPZ_PATH))
raw_obs = data["raw_observations"]
raw_acts = data["raw_actions"]
rewards = data["rewards"]
terminals = data["terminals"]
valid_points = data["valid_points"]

print("Original shape:", raw_obs.shape)

# Step 1: Filter out any raw_observations that contain NaNs
# We want to identify the clean simulations.
# Each row in raw_obs corresponds to a source simulation.
clean_rows_mask = ~np.isnan(raw_obs).any(axis=1)
clean_raw_obs = raw_obs[clean_rows_mask]

lookup = {}
for row in clean_raw_obs:
    freq = row[1024]
    pos = tuple(row[1025:1028])
    mass = row[1028]
    key = (freq, pos, mass)
    lookup[key] = row[0:1024] # latent_z

print(f"Built lookup dictionary with {len(lookup)} clean simulations.")

# Step 2: Reconstruct raw_next_observations for each transition
raw_next_obs = []
valid_indices = []

for idx, (row, act) in enumerate(zip(raw_obs, raw_acts)):
    if np.isnan(row).any():
        continue # skip if source is NaN
        
    freq = row[1024]
    src_pos = row[1025:1028]
    src_mass = row[1028]
    
    dst_pos = src_pos + act[0:3]
    dst_mass = src_mass + act[3]
    
    # Find matching key in lookup
    dst_pos_key = None
    for k in lookup.keys():
        if k[0] == freq and abs(k[2] - dst_mass) < 1e-4:
            if np.allclose(k[1], dst_pos, atol=1e-2):
                dst_pos_key = k
                break
                
    if dst_pos_key is None:
        # Target simulation was a NaN simulation, so skip this transition!
        continue
        
    latent_dst = lookup[dst_pos_key]
    state_dst = np.concatenate([
        latent_dst,
        [freq],
        dst_pos_key[1],
        [dst_pos_key[2]]
    ])
    
    raw_next_obs.append(state_dst)
    valid_indices.append(idx)

raw_next_obs = np.array(raw_next_obs, dtype=np.float32)
clean_raw_obs = raw_obs[valid_indices]
clean_raw_acts = raw_acts[valid_indices]
clean_rewards = rewards[valid_indices]
clean_terminals = terminals[valid_indices]

print(f"Reconstructed {len(clean_raw_obs)} clean transitions (removed {len(raw_obs) - len(clean_raw_obs)} transitions).")

# Step 3: Re-normalize the dataset
obs_mean = clean_raw_obs.mean(axis=0)
obs_std = clean_raw_obs.std(axis=0) + 1e-8

observations_norm = (clean_raw_obs - obs_mean) / obs_std
next_observations_norm = (raw_next_obs - obs_mean) / obs_std

action_maxs = np.max(np.abs(clean_raw_acts), axis=0) + 1e-8
actions_norm = clean_raw_acts / action_maxs

# Step 4: Save the fixed dataset back
np.savez(NPZ_PATH,
         observations=observations_norm,
         actions=actions_norm,
         rewards=clean_rewards,
         next_observations=next_observations_norm,
         terminals=clean_terminals,
         obs_mean=obs_mean,
         obs_std=obs_std,
         action_maxs=action_maxs,
         valid_points=valid_points,
         raw_observations=clean_raw_obs,
         raw_actions=clean_raw_acts)

print("Successfully saved clean dataset to", NPZ_PATH)

import numpy as np
import d3rlpy
import torch

def train_with_d3rlpy():
    print("Loading offline dataset for d3rlpy...")
    data = np.load("offline_dataset.npz")
    
    # In d3rlpy, we create an MDPDataset
    # The dataset requires episodes. For simplicity, if we don't have episode terminals perfectly aligned
    # we can pass it as a single chunk, but d3rlpy handles terminals well.
    dataset = d3rlpy.dataset.MDPDataset(
        observations=data["observations"],
        actions=data["actions"],
        rewards=data["rewards"],
        terminals=data["terminals"]
    )
    
    print(f"Loaded {len(dataset)} transitions.")
    
    # Define the TD3+BC algorithm (state of the art for offline RL with continuous actions)
    # alpha is the conservative penalty (similar to our custom script)
    td3_bc = d3rlpy.algos.TD3PlusBCConfig(
        actor_learning_rate=3e-4,
        critic_learning_rate=3e-4,
        alpha=2.5,
        use_gpu=torch.cuda.is_available()
    ).create()
    
    print("Starting d3rlpy training...")
    # Train the agent purely offline
    td3_bc.fit(
        dataset,
        n_steps=10000,
        n_steps_per_epoch=1000,
        save_interval=10,
        experiment_name="td3_bc_arm"
    )
    
    # Save the model
    td3_bc.save("d3rlpy_offline_agent.pt")
    print("Saved d3rlpy agent to d3rlpy_offline_agent.pt")

if __name__ == "__main__":
    # Note: Requires 'pip install d3rlpy'
    train_with_d3rlpy()

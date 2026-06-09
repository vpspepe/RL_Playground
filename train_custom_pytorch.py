import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import os
import hydra
from omegaconf import DictConfig
from torch.utils.data import Dataset, DataLoader

# ==========================================
# STATE VECTOR STRUCTURE INDEX CONSTANTS
# ==========================================
LATENT_SLICE = slice(0, 1024)
FREQ_INDEX = 1024
POS_SLICE = slice(1025, 1028)
MASS_INDEX = 1028

# --- 1. Define Networks ---
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hidden_dims):
        super(Actor, self).__init__()
        layers = []
        in_dim = state_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, action_dim))
        layers.append(nn.Tanh()) # Outputs between -1 and 1
        self.net = nn.Sequential(*layers)
        self.max_action = max_action

    def forward(self, state):
        return self.max_action * self.net(state)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims):
        super(Critic, self).__init__()
        layers = []
        in_dim = state_dim + action_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        return self.net(sa)

# --- 2. Offline RL Algorithm (TD3+BC style) ---
class CustomOfflineRL:
    def __init__(self, state_dim, action_dim, max_action, cfg: DictConfig, valid_points, obs_mean, obs_std, action_maxs):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.actor = Actor(state_dim, action_dim, max_action, cfg.model.actor_hidden_dims).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=cfg.hyperparameters.lr)

        self.critic = Critic(state_dim, action_dim, cfg.model.critic_hidden_dims).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=cfg.hyperparameters.lr)

        self.max_action = max_action
        self.discount = cfg.hyperparameters.discount
        self.tau = cfg.hyperparameters.tau
        self.alpha = cfg.hyperparameters.alpha # BC weight (how strictly to follow the dataset)
        self.boundary_penalty_coeff = cfg.hyperparameters.boundary_penalty_coeff

        # Downsample valid_points to 1000 points for speed
        if len(valid_points) > 1000:
            indices = np.random.choice(len(valid_points), size=1000, replace=False)
            valid_points = valid_points[indices]
            
        self.valid_points = torch.as_tensor(valid_points, dtype=torch.float32, device=self.device)
        self.obs_mean = torch.as_tensor(obs_mean, dtype=torch.float32, device=self.device)
        self.obs_std = torch.as_tensor(obs_std, dtype=torch.float32, device=self.device)
        self.action_maxs = torch.as_tensor(action_maxs, dtype=torch.float32, device=self.device)

    def train_step(self, state, action, next_state, reward, done):
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        action = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        next_state = torch.as_tensor(next_state, dtype=torch.float32, device=self.device)
        reward = torch.as_tensor(reward, dtype=torch.float32, device=self.device).unsqueeze(1)
        done = torch.as_tensor(done, dtype=torch.float32, device=self.device).unsqueeze(1)

        # --- Train Critic ---
        with torch.no_grad():
            # Get next action from target actor
            next_action = self.actor_target(next_state)
            # Add some noise for smoothing (TD3 trick)
            noise = (torch.randn_like(action) * 0.2).clamp(-0.5, 0.5)
            next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

            # Compute target Q-value
            target_Q = self.critic_target(next_state, next_action)
            target_Q = reward + (1 - done) * self.discount * target_Q

        # Get current Q-value
        current_Q = self.critic(state, action)
        
        # Critic Loss: Minimize Bellman Error
        critic_loss = nn.MSELoss()(current_Q, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- Train Actor ---
        # Get actor's proposed action
        pi = self.actor(state)
        Q_pi = self.critic(state, pi)

        # Behavioral Cloning Loss (stay close to dataset actions)
        bc_loss = nn.MSELoss()(pi, action)
        
        # Q-learning Loss (maximize Q value). We use a trick from TD3+BC to balance them.
        lmbda = self.alpha / Q_pi.abs().mean().detach()
        
        # Boundary Penalty: penalize suggested positions outside the structure
        # state is normalized: s = [z || freq || pos (3d) || mass]
        # We un-normalize pos (indices 1025:1028)
        pos_norm = state[:, POS_SLICE]
        pos_mean = self.obs_mean[POS_SLICE]
        pos_std = self.obs_std[POS_SLICE]
        pos_raw = pos_norm * pos_std + pos_mean
        
        # pi is normalized: [disp_x, disp_y, disp_z, delta_mass] in [-1, 1]
        # We un-normalize displacement (indices 0:3)
        disp_raw = pi[:, 0:3] * self.action_maxs[0:3]
        
        # Predicted raw target position
        pred_pos_raw = pos_raw + disp_raw
        
        # Compute pairwise distance squared to valid surface points
        diff = pred_pos_raw.unsqueeze(1) - self.valid_points.unsqueeze(0)
        dists = torch.sum(diff ** 2, dim=2) # (B, V)
        min_dists, _ = torch.min(dists, dim=1) # (B,)
        boundary_penalty = torch.mean(min_dists)
        
        actor_loss = -lmbda * Q_pi.mean() + bc_loss + self.boundary_penalty_coeff * boundary_penalty

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # --- Update Target Networks ---
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return critic_loss.item(), actor_loss.item()

# --- 3. PyTorch Dataset ---
class OfflineRLDataset(Dataset):
    def __init__(self, observations, actions, rewards, next_observations, terminals):
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.next_observations = next_observations
        self.terminals = terminals

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        return {
            "observation": self.observations[idx],
            "action": self.actions[idx],
            "reward": self.rewards[idx],
            "next_observation": self.next_observations[idx],
            "terminal": self.terminals[idx]
        }

# --- 4. Training Loop ---
@hydra.main(config_path=".", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print("Loading offline dataset...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, cfg.training.dataset_path)
    
    try:
        data = np.load(dataset_path)
    except FileNotFoundError:
        print(f"Error: Dataset not found at {dataset_path}. Please build it first.")
        return
        
    states = data["observations"]
    actions = data["actions"]
    rewards = data["rewards"]
    next_states = data["next_observations"]
    terminals = data["terminals"]

    # Normalize rewards (helps training stability)
    rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

    state_dim = states.shape[1]
    action_dim = actions.shape[1]
    max_action = float(np.max(np.abs(actions)))
    
    # Load metadata required for un-normalization and boundary checking
    obs_mean = data["obs_mean"]
    obs_std = data["obs_std"]
    action_maxs = data["action_maxs"]
    valid_points = data["valid_points"]
    
    print(f"Initialized CustomOfflineRL with State Dim: {state_dim}, Action Dim: {action_dim}, Max Action: {max_action}")
    agent = CustomOfflineRL(
        state_dim, action_dim, max_action, cfg, 
        valid_points=valid_points, 
        obs_mean=obs_mean, 
        obs_std=obs_std, 
        action_maxs=action_maxs
    )
    
    # Instantiate PyTorch Dataset and DataLoader
    dataset = OfflineRLDataset(states, actions, rewards, next_states, terminals)
    dataloader = DataLoader(
        dataset, 
        batch_size=cfg.training.batch_size, 
        shuffle=True, 
        drop_last=True
    )
    
    epochs = cfg.training.epochs
    print(f"Starting Training for {epochs} epochs...")
    
    for epoch in range(epochs):
        epoch_critic_loss = 0
        epoch_actor_loss = 0
        steps = 0
        
        for batch in dataloader:
            batch_states = batch["observation"]
            batch_actions = batch["action"]
            batch_rewards = batch["reward"]
            batch_next_states = batch["next_observation"]
            batch_terminals = batch["terminal"]
            
            c_loss, a_loss = agent.train_step(
                batch_states, batch_actions, batch_next_states, batch_rewards, batch_terminals
            )
            
            epoch_critic_loss += c_loss
            epoch_actor_loss += a_loss
            steps += 1
            
        print(f"Epoch {epoch+1}/{epochs} | Critic Loss: {epoch_critic_loss/steps:.4f} | Actor Loss: {epoch_actor_loss/steps:.4f}")
    
    save_path = os.path.join(script_dir, cfg.training.save_model_path)
    torch.save(agent.actor.state_dict(), save_path)
    print(f"Saved custom offline agent to {save_path}")

if __name__ == "__main__":
    main()

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy

# --- 1. Define Networks ---
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh() # Outputs between -1 and 1
        )
        self.max_action = max_action

    def forward(self, state):
        return self.max_action * self.net(state)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        return self.net(sa)

# --- 2. Offline RL Algorithm (TD3+BC style) ---
class CustomOfflineRL:
    def __init__(self, state_dim, action_dim, max_action):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=3e-4)

        self.max_action = max_action
        self.discount = 0.99
        self.tau = 0.005 # Soft update parameter
        self.alpha = 2.5 # BC weight (how strictly to follow the dataset)

    def train_step(self, state, action, next_state, reward, done):
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        done = torch.FloatTensor(done).unsqueeze(1).to(self.device)

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
        actor_loss = -lmbda * Q_pi.mean() + bc_loss

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # --- Update Target Networks ---
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return critic_loss.item(), actor_loss.item()

# --- 3. Training Loop ---
if __name__ == "__main__":
    print("Loading offline dataset...")
    data = np.load("offline_dataset.npz")
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
    
    agent = CustomOfflineRL(state_dim, action_dim, max_action)
    
    batch_size = 256
    epochs = 10
    total_steps = len(states) // batch_size
    
    print(f"Starting Training for {epochs} epochs...")
    
    for epoch in range(epochs):
        # Shuffle dataset
        indices = np.random.permutation(len(states))
        
        epoch_critic_loss = 0
        epoch_actor_loss = 0
        
        for step in range(total_steps):
            batch_idx = indices[step * batch_size : (step + 1) * batch_size]
            
            batch_states = states[batch_idx]
            batch_actions = actions[batch_idx]
            batch_rewards = rewards[batch_idx]
            batch_next_states = next_states[batch_idx]
            batch_terminals = terminals[batch_idx]
            
            c_loss, a_loss = agent.train_step(
                batch_states, batch_actions, batch_next_states, batch_rewards, batch_terminals
            )
            
            epoch_critic_loss += c_loss
            epoch_actor_loss += a_loss
            
        print(f"Epoch {epoch+1}/{epochs} | Critic Loss: {epoch_critic_loss/total_steps:.4f} | Actor Loss: {epoch_actor_loss/total_steps:.4f}")
    
    torch.save(agent.actor.state_dict(), "custom_offline_actor.pth")
    print("Saved custom offline agent to custom_offline_actor.pth")

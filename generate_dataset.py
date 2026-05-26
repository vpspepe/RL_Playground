import numpy as np
from arm_env import RoboticArmEnv
import math
import os

def generate_dataset(num_episodes=5000, max_steps_per_episode=50, save_path="offline_dataset.npz"):
    env = RoboticArmEnv()
    
    observations = []
    actions = []
    rewards = []
    next_observations = []
    terminals = []
    
    print("Generating offline dataset...")
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        
        for step in range(max_steps_per_episode):
            observations.append(obs.copy())
            
            # Mix of random actions and "greedy" actions to ensure some success
            if np.random.rand() < 0.3:
                # Random action
                action = env.action_space.sample()
            else:
                # Simple greedy heuristic: try a few random actions and pick the one that gets us closer
                best_action = None
                best_reward = -float('inf')
                for _ in range(5):
                    test_action = env.action_space.sample()
                    # Simulate step
                    t1, t2, tx, ty = obs
                    dt1, dt2 = test_action
                    nt1 = (t1 + dt1 + np.pi) % (2 * np.pi) - np.pi
                    nt2 = (t2 + dt2 + np.pi) % (2 * np.pi) - np.pi
                    
                    cx = env.l1 * math.cos(nt1) + env.l2 * math.cos(nt1 + nt2)
                    cy = env.l1 * math.sin(nt1) + env.l2 * math.sin(nt1 + nt2)
                    dist = math.sqrt((cx - tx)**2 + (cy - ty)**2)
                    
                    rew = env.current_distance - dist
                    if rew > best_reward:
                        best_reward = rew
                        best_action = test_action
                
                action = best_action
            
            # Add some noise
            action = action + np.random.normal(0, 0.05, size=action.shape)
            action = np.clip(action, env.action_space.low, env.action_space.high)
            
            # Step environment
            next_obs, reward, terminated, _, _ = env.step(action)
            
            actions.append(action.copy())
            rewards.append(reward)
            next_observations.append(next_obs.copy())
            terminals.append(1 if terminated else 0)
            
            obs = next_obs
            
            if terminated:
                break
                
        if (episode + 1) % 1000 == 0:
            print(f"Generated {episode + 1} episodes...")
            
    observations = np.array(observations)
    actions = np.array(actions)
    rewards = np.array(rewards)
    next_observations = np.array(next_observations)
    terminals = np.array(terminals)
    
    print(f"Dataset generated! Total transitions: {len(observations)}")
    
    np.savez(save_path, 
             observations=observations, 
             actions=actions, 
             rewards=rewards, 
             next_observations=next_observations,
             terminals=terminals)
    
    print(f"Saved to {save_path}")

if __name__ == "__main__":
    generate_dataset()

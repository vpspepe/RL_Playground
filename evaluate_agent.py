import torch
import numpy as np
import matplotlib.pyplot as plt
from arm_env import RoboticArmEnv
from train_custom_pytorch import Actor
import math

def evaluate_custom_agent():
    env = RoboticArmEnv()
    
    # Initialize the custom PyTorch Actor
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    actor = Actor(state_dim, action_dim, max_action)
    try:
        actor.load_state_dict(torch.load("custom_offline_actor.pth"))
        actor.eval()
        print("Loaded custom_offline_actor.pth successfully.")
    except FileNotFoundError:
        print("Model file not found. Please run train_custom_pytorch.py first.")
        return
        
    obs, _ = env.reset()
    target_x, target_y = obs[2], obs[3]
    
    # Track positions for plotting
    ee_x = []
    ee_y = []
    
    # Get initial position
    x0, y0 = env._get_ee_pos(obs[0], obs[1])
    ee_x.append(x0)
    ee_y.append(y0)
    
    print(f"Target is at: ({target_x:.2f}, {target_y:.2f})")
    print(f"Starting distance: {env.current_distance:.2f}")
    
    total_reward = 0
    for step in range(50):
        # The trained actor is deterministic in evaluation
        with torch.no_grad():
            state_tensor = torch.FloatTensor(obs).unsqueeze(0)
            action = actor(state_tensor).numpy()[0]
            
        obs, reward, terminated, _, _ = env.step(action)
        total_reward += reward
        
        # Get new position
        x, y = env._get_ee_pos(obs[0], obs[1])
        ee_x.append(x)
        ee_y.append(y)
        
        if terminated:
            print(f"Reached target in {step+1} steps!")
            break
            
    print(f"Final distance: {env.current_distance:.2f}")
    print(f"Total Reward: {total_reward:.2f}")
    
    # Plotting
    plt.figure(figsize=(6, 6))
    
    # Plot target
    plt.scatter([target_x], [target_y], color='red', s=100, marker='*', label='Target')
    
    # Plot path
    plt.plot(ee_x, ee_y, 'b-', alpha=0.5, label='End-Effector Path')
    plt.scatter(ee_x[0], ee_y[0], color='green', label='Start')
    plt.scatter(ee_x[-1], ee_y[-1], color='blue', label='End')
    
    # Plot reachable workspace boundary (circle of radius l1+l2)
    circle = plt.Circle((0, 0), env.l1 + env.l2, color='gray', fill=False, linestyle='--')
    plt.gca().add_patch(circle)
    
    plt.xlim(-2.5, 2.5)
    plt.ylim(-2.5, 2.5)
    plt.grid(True)
    plt.legend()
    plt.title("Iterative Refinement (Offline RL Agent)")
    
    # Save plot instead of showing to avoid hanging the script
    plt.savefig("evaluation_plot.png")
    print("Saved trajectory plot to evaluation_plot.png")

if __name__ == "__main__":
    evaluate_custom_agent()

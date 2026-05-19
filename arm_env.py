import gymnasium as gym
import numpy as np
from gymnasium import spaces
import math

class RoboticArmEnv(gym.Env):
    """
    Custom Environment that follows gym interface.
    This simulates a 2-link robotic arm. The agent's action is to change the joint angles (deltas).
    """
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self):
        super().__init__()
        
        # Arm lengths
        self.l1 = 1.0
        self.l2 = 1.0
        
        # Max angle change per step (Delta actions)
        self.max_action = np.pi / 4.0 # 45 degrees max per step
        
        # Actions: [Delta Theta 1, Delta Theta 2]
        self.action_space = spaces.Box(
            low=np.array([-self.max_action, -self.max_action], dtype=np.float32), 
            high=np.array([self.max_action, self.max_action], dtype=np.float32),
            shape=(2,), 
            dtype=np.float32
        )
        
        # Observations: [Theta1, Theta2, TargetX, TargetY]
        # Angles are from -pi to pi, Target can be anywhere in [-2, 2]
        self.observation_space = spaces.Box(
            low=np.array([-np.pi, -np.pi, -2.0, -2.0], dtype=np.float32),
            high=np.array([np.pi, np.pi, 2.0, 2.0], dtype=np.float32),
            shape=(4,),
            dtype=np.float32
        )
        
        self.state = None
        self.target = None
        self.current_distance = None

    def _get_ee_pos(self, theta1, theta2):
        """Calculate end-effector (x, y) position using forward kinematics"""
        x = self.l1 * math.cos(theta1) + self.l2 * math.cos(theta1 + theta2)
        y = self.l1 * math.sin(theta1) + self.l2 * math.sin(theta1 + theta2)
        return x, y

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize random joint angles
        theta1 = self.np_random.uniform(low=-np.pi, high=np.pi)
        theta2 = self.np_random.uniform(low=-np.pi, high=np.pi)
        
        # Initialize random target that is reachable
        # Generate reachable target by picking random angles
        t_theta1 = self.np_random.uniform(low=-np.pi, high=np.pi)
        t_theta2 = self.np_random.uniform(low=-np.pi, high=np.pi)
        target_x, target_y = self._get_ee_pos(t_theta1, t_theta2)
        
        self.target = np.array([target_x, target_y], dtype=np.float32)
        self.state = np.array([theta1, theta2, target_x, target_y], dtype=np.float32)
        
        # Calculate initial distance
        curr_x, curr_y = self._get_ee_pos(theta1, theta2)
        self.current_distance = math.sqrt((curr_x - target_x)**2 + (curr_y - target_y)**2)
        
        return self.state, {}

    def step(self, action):
        # Apply deltas
        theta1, theta2, target_x, target_y = self.state
        delta_t1, delta_t2 = action
        
        # Update state and wrap angles to [-pi, pi]
        new_theta1 = (theta1 + delta_t1 + np.pi) % (2 * np.pi) - np.pi
        new_theta2 = (theta2 + delta_t2 + np.pi) % (2 * np.pi) - np.pi
        
        self.state = np.array([new_theta1, new_theta2, target_x, target_y], dtype=np.float32)
        
        # Calculate new distance
        curr_x, curr_y = self._get_ee_pos(new_theta1, new_theta2)
        new_distance = math.sqrt((curr_x - target_x)**2 + (curr_y - target_y)**2)
        
        # Reward is the decrease in distance (Iterative Refinement formulation)
        # If new_distance is smaller, reward is positive!
        reward = float(self.current_distance - new_distance)
        
        # Update current distance for next step
        self.current_distance = new_distance
        
        # Check termination (reached target)
        terminated = bool(new_distance < 0.05)
        
        # Give a big bonus for reaching the target
        if terminated:
            reward += 10.0
            
        return self.state, reward, terminated, False, {}

    def render(self):
        pass # We will visualize it separately in evaluate_agent.py


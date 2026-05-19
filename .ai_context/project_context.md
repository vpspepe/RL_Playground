# RL_Playground: Project Context

This folder (`.ai_context/`) serves as the persistent memory for the AI agent across different machines (personal PC vs. lab PC).

## 1. Project Goal
The real-world goal is to optimize the placement of Tuned Mass Dampers (TMDs) to minimize Equivalent Radiated Power (ERP) using COMSOL simulations. 

This specific repository (`RL_Playground`) is an **Educational Sandbox** to learn Offline Reinforcement Learning (Offline RL) using an **Iterative Refinement MDP**. We avoid the heavy physics calculations here and use a toy problem that perfectly mimics the mechanics of the real problem.

## 2. MDP Formulation (Iterative Refinement)
Instead of predicting absolute values, the agent learns to iteratively adjust parameters step-by-step.
- **Domain**: 2D Robotic Arm Inverse Kinematics (IK).
- **State ($S$)**: Current joint angles $[\theta_1, \theta_2]$, current $[x, y]$, and target $[x^*, y^*]$.
- **Action ($A$)**: Deltas $[\Delta \theta_1, \Delta \theta_2]$.
- **Next State ($S'$)**: Updated position based on deltas.
- **Reward ($R$)**: Decrease in distance to the target ($Distance_{old} - Distance_{new}$).

## 3. Implemented Components
1. **`arm_env.py`**: Custom Gymnasium environment for the 2D Robotic Arm.
2. **`generate_dataset.py`**: Generates synthetic transitions using a mix of random and greedy actions, saved to `offline_dataset.npz` (Simulates the COMSOL offline data).
3. **`train_custom_pytorch.py`**: A raw, from-scratch PyTorch implementation of Offline RL using Behavioral Cloning + Q-Learning (Bellman Error). This is the core educational script.
4. **`train_d3rlpy.py`**: An implementation using the industry-standard `d3rlpy` offline RL library (TD3+BC).
5. **`evaluate_agent.py`**: Evaluates the trained PyTorch agent and plots the arm's trajectory to the target.

## 4. How to Resume Work
When resuming on a new computer:
1. Ensure the Python environment has `torch`, `gymnasium`, `numpy`, and `matplotlib` installed.
2. Generate the dataset if `offline_dataset.npz` is not present: `python generate_dataset.py`
3. If asked to modify or add features, read the scripts above to understand the current architecture.

## 5. History Log
- **[2026-05-19]**: Initialized repository. Built the IK environment, dataset generator, and two Offline RL training scripts (custom PyTorch and d3rlpy). Successfully verified the custom agent can walk to the target using only offline data.

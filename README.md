# Hands-On Reinforcement Learning

This repository provides hands-on implementations of classic and advanced reinforcement learning (RL) algorithms, applied to both control and robotic manipulation tasks. The project is designed for learners and practitioners who want to understand, experiment with, and extend RL methods in Python.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Implemented Algorithms](#implemented-algorithms)
  - [REINFORCE for CartPole](#reinforce-for-cartpole)
  - [A2C for CartPole (Stable-Baselines3)](#a2c-for-cartpole-using-stable-baselines3)
  - [DDPG for Panda Robot (Stable-Baselines3)](#ddpg-for-panda-robot-using-stable-baselines3)
- [Reports](#reports)
- [License](#license)
- [Author](#author)

## Project Overview

This project explores three reinforcement learning approaches:

1. **REINFORCE (Vanilla Policy Gradient)** for CartPole
2. **A2C (Advantage Actor-Critic)** for CartPole using Stable-Baselines3
3. **DDPG (Deep Deterministic Policy Gradient)** for the Panda robotic arm on a reaching task

Each implementation demonstrates different aspects of RL, from basic policy gradients to actor-critic methods and experience replay in continuous control.

## Features
- Clean, well-commented code for each RL algorithm
- PyTorch and Stable-Baselines3 implementations
- Integration with Weights & Biases for experiment tracking
- Model packaging and sharing via Hugging Face Model Hub
- Evaluation scripts and reproducible results
- PDF reports for in-depth analysis

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/hands-on-rl.git
   cd hands-on-rl
   ```
2. **Install dependencies:**
   (You may want to use a virtual environment)
   ```bash
   pip install torch gymnasium stable-baselines3 panda-gym wandb huggingface_sb3
   ```
   - For the Panda robot tasks, you may also need `pybullet` and additional dependencies for `panda-gym`.

## Implemented Algorithms

### REINFORCE for CartPole
A custom implementation of the REINFORCE algorithm to solve the CartPole-v1 environment using PyTorch.

- **Source:** [`reinforce_cartpole.py`](reinforce_cartpole.py)
- **Evaluation:** [`evaluate_reinforce_cartpole.py`](evaluate_reinforce_cartpole.py)

**How to train:**
```bash
python reinforce_cartpole.py
```

**How to evaluate:**
```bash
python evaluate_reinforce_cartpole.py
```

**Results:**
- Average Reward: 500.00/500.00
- Success Rate: 100%
- Number of evaluation episodes: 100

### A2C for CartPole (using Stable-Baselines3)
Implementation of the A2C algorithm using Stable-Baselines3, with experiment tracking via Weights & Biases and model sharing on Hugging Face.

- **Source:** [`a2c_sb3_cartpole.py`](a2c_sb3_cartpole.py)
- **Model:** [Hugging Face Model Hub](https://huggingface.co/StevanLS/a2c-cartpole-v1)

**How to train:**
```bash
python a2c_sb3_cartpole.py
```

**How to use the pretrained model:**
```python
from stable_baselines3 import A2C
# Load the model directly from Hugging Face Hub
model = A2C.load("StevanLS/a2c-cartpole-v1")
```

**Results:**
- Trained for 50,000 timesteps
- 100% success rate in evaluation (100 episodes)

### DDPG for Panda Robot (using Stable-Baselines3)
Implements DDPG with Hindsight Experience Replay (HER) for the Panda robotic arm on the `PandaReachJointsDense-v3` environment from `panda-gym`.

- **Source:** [`ddpg_sb3_panda_reach.py`](ddpg_sb3_panda_reach.py)
- **Environment:** [panda-gym: PandaReachJointsDense-v3](https://panda-gym.readthedocs.io/en/latest/usage/environments.html)

**Environment Description:**
> The PandaReachJointsDense-v3 environment tasks the Franka Emika Panda robot with reaching a target position using joint control and a dense reward function. The closer the robot's end-effector is to the target, the higher the reward. This environment is ideal for testing continuous control algorithms with dense feedback.

**How to train:**
```bash
python ddpg_sb3_panda_reach.py
```

**Results:**
- Trained for 500,000 timesteps
- Model packaging and upload to Hugging Face Model Hub

## Reports
- **CartPole A2C (SB3) with Weights & Biases:** `SB3 Cartpole Wandb report.pdf`
- **DDPG (SB3) on PandaReachJointsDense:** `DDPG SB3 PandaReachJointsDense Report.pdf`

These PDF reports provide detailed training curves, hyperparameters, and analysis.

## Author
Stevan Le Stanc
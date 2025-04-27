# Hands-On Reinforcement Learning

This repository contains implementations of various reinforcement learning algorithms applied to classic control and robotic manipulation tasks.

## Project Overview

This project explores three reinforcement learning approaches:

1. **REINFORCE (Vanilla Policy Gradient)** for CartPole
2. **A2C (Advantage Actor-Critic)** for CartPole using Stable-Baselines3
3. **DDPG (Deep Deterministic Policy Gradient)** for Panda robotic arm reaching tasks

Each implementation demonstrates different aspects of reinforcement learning, from basic policy gradients to more complex actor-critic methods with experience replay.

## Implementations

### REINFORCE for CartPole

Custom implementation of the REINFORCE algorithm to solve the CartPole-v1 environment:

- Built a neural network policy with PyTorch
- Achieved perfect 500/500 score in evaluation
- 100% success rate over 100 evaluation episodes

The implementation can be found in [`reinforce_cartpole.py`](reinforce_cartpole.py) with evaluation code in [`evaluate_reinforce_cartpole.py`](evaluate_reinforce_cartpole.py).

#### Evaluation Results
- **Average Reward**: 500.00/500.00
- **Success Rate**: 100.00%
- **Number of evaluation episodes**: 100

These results show that our implementation achieved perfect performance, with the agent consistently reaching the maximum possible score of 500 steps in every evaluation episode. The success criterion for CartPole-v1 (maintaining the pole upright for at least 195 steps) was met in 100% of the evaluation episodes.

### A2C for CartPole (using Stable-Baselines3)

Implementation using the Stable-Baselines3 library:

- Integrated with Weights & Biases for experiment tracking
- Trained for 50,000 timesteps
- Achieved 100% success rate in evaluation
- Model available on [Hugging Face Model Hub](https://huggingface.co/StevanLS/a2c-cartpole-v1)

Source code: [`a2c_sb3_cartpole.py`](a2c_sb3_cartpole.py)

#### Model Availability

The trained A2C model for CartPole-v1 is available on the Hugging Face Model Hub:
- Repository: [StevanLS/a2c-cartpole-v1](https://huggingface.co/StevanLS/a2c-cartpole-v1)
- Architecture: A2C (Advantage Actor-Critic)
- Environment: CartPole-v1

#### How to Use the Model

You can load and use this model with:
```python
from stable_baselines3 import A2C

# Load the model directly from Hugging Face Hub
model = A2C.load("StevanLS/a2c-cartpole-v1")

Stevan Le Stanc
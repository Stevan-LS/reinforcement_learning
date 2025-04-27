import gymnasium as gym
import torch
import numpy as np
from torch.distributions import Categorical

import torch.nn as nn
import torch.optim as optim

# Create the policy network
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.network(x)

def train():
    # Initialize environment and agent
    env = gym.make('CartPole-v1', render_mode=None)
    input_dim = env.observation_space.shape[0]  # 4 for CartPole
    output_dim = env.action_space.n  # 2 for CartPole
    hidden_dim = 128

    policy = PolicyNetwork(input_dim, hidden_dim, output_dim)
    optimizer = optim.Adam(policy.parameters(), lr=5e-3)

    # Training loop
    episodes = 500
    gamma = 0.99
    episode_rewards = []

    for episode in range(episodes):
        state, _ = env.reset()
        rewards = []
        log_probs = []
        done = False
        episode_reward = 0

        while not done:
            state = torch.FloatTensor(state).unsqueeze(0)
            # Get action probabilities
            action_probs = policy(state).squeeze(0)
            # Sample action
            dist = Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            # Take action in environment
            state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            
            # Store reward and log probability
            rewards.append(reward)
            log_probs.append(log_prob)
            episode_reward += reward

        episode_rewards.append(episode_reward)

        # Compute returns
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)

        # Convert to tensor and normalize
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        # Compute policy loss
        policy_loss = []
        for log_prob, R in zip(log_probs, returns):
            policy_loss.append(-log_prob * R)
        policy_loss = torch.stack(policy_loss).sum()

        # Update policy
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        if (episode + 1) % 50 == 0:
            print(f'Episode {episode + 1}, Average Reward: {np.mean(episode_rewards[-50:])}')

    # Save the model
    torch.save(policy.state_dict(), 'reinforce_cartpole.pth')
    env.close()

    # Print final performance
    print(f'Final average reward over last 50 episodes: {np.mean(episode_rewards[-50:])}')

if __name__ == '__main__':
    train()
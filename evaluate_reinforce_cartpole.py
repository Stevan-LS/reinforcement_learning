import gymnasium as gym
import torch
from reinforce_cartpole import PolicyNetwork
import numpy as np

# Create environment (no rendering for multiple evaluations)
env = gym.make('CartPole-v1', render_mode=None)

# Initialize model with same parameters
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n
hidden_dim = 128

# Create and load the trained model
policy = PolicyNetwork(input_dim, hidden_dim, output_dim)
policy.load_state_dict(torch.load('reinforce_cartpole.pth'))
policy.eval()

# Run 100 evaluation episodes
num_episodes = 100
rewards = []

for episode in range(num_episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        state = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            action_probs = policy(state)
            action = torch.argmax(action_probs).item()
        
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
    
    rewards.append(total_reward)
    if (episode + 1) % 10 == 0:
        print(f"Episode {episode + 1}/100, Current average reward: {np.mean(rewards):.2f}")

env.close()

# Calculate success metrics
average_reward = np.mean(rewards)
success_rate = sum(r >= 195.0 for r in rewards) / num_episodes  # CartPole is considered solved if reward >= 195.0

print("\nEvaluation Results:")
print(f"Average Reward: {average_reward:.2f}")
print(f"Success Rate: {success_rate:.2%}")
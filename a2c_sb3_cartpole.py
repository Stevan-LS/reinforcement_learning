import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np
from huggingface_sb3 import package_to_hub
import wandb
from wandb.integration.sb3 import WandbCallback

# Initialize wandb
wandb.init(
    project="sb3-cartpole",
    sync_tensorboard=True,
    monitor_gym=True,
    save_code=True,
)

# Create environment
env = gym.make("CartPole-v1")

# Initialize A2C agent
model = A2C(
    policy="MlpPolicy",
    env=env,
    learning_rate=0.001,
    n_steps=5,
    gamma=0.99,
    verbose=1,
    tensorboard_log=f"runs/{wandb.run.id}"
)

# Train the agent
total_timesteps = 50000
model.learn(
    total_timesteps=total_timesteps,
    callback=WandbCallback(
        gradient_save_freq=100,
        model_save_path=f"models/{wandb.run.id}",
        verbose=2,
    )
)

# Save the model
model.save("a2c_cartpole")

# Evaluate the trained agent
mean_reward, std_reward = evaluate_policy(
    model, 
    env, 
    n_eval_episodes=100,
    deterministic=True
)

print(f"\nMean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

# Calculate success rate (episodes with reward >= 195)
eval_episodes = 100
rewards = []

for _ in range(eval_episodes):
    obs, _ = env.reset()
    done = False
    episode_reward = 0
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        episode_reward += reward
    
    rewards.append(episode_reward)

success_rate = sum(r >= 195.0 for r in rewards) / eval_episodes
print(f"Success rate: {success_rate:.2%}")

# Create eval environment
eval_env = gym.make("CartPole-v1")

# Package the model to the hub
package_to_hub(
    model=model,
    model_name="a2c-CartPole-v1",
    model_architecture="A2C",
    env_id="CartPole-v1",
    eval_env=eval_env,
    repo_id="StevanLS/a2c-cartpole-v1",
    commit_message="Upload A2C CartPole model"
)

# Close eval environment
eval_env.close()

env.close()

# Don't forget to close wandb at the end
wandb.finish()
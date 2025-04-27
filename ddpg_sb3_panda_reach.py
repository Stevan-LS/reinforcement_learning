import gymnasium as gym
from stable_baselines3 import DDPG, HerReplayBuffer
from stable_baselines3.common.evaluation import evaluate_policy
import panda_gym
import wandb
from wandb.integration.sb3 import WandbCallback
from huggingface_sb3 import package_to_hub

# Initialize wandb
wandb.init(
    project="sb3-panda-reach",
    sync_tensorboard=True,
    monitor_gym=True,
    save_code=True,
)

env = None
eval_env = None
try:
    env = gym.make("PandaReachJointsDense-v3")
    
    # Initialize DDPG agent
    model = DDPG(
        policy="MultiInputPolicy",
        env=env,
        replay_buffer_class=HerReplayBuffer,
        verbose=1,
        tensorboard_log=f"runs/{wandb.run.id}"
    )

    # Train the agent
    model.learn(
        total_timesteps=500000,
        callback=WandbCallback(
            gradient_save_freq=100,
            model_save_path=f"models/{wandb.run.id}",
            verbose=2,
        )
    )

    # Save the model
    model.save("DDPG_PandaReachJointsDense_500")

    # Evaluate the model
    eval_env = gym.make("PandaReachJointsDense-v3")
    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=10,
        deterministic=True
    )
    print(f"\nMean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    # Package the model to HuggingFace Hub
    package_to_hub(
        model=model,
        model_name="ddpg-panda-reach-500k",
        model_architecture="DDPG",
        env_id="PandaReachJointsDense-v3",
        eval_env=eval_env,
        repo_id="StevanLS/ddpg-panda-reach-500",
        commit_message="Upload DDPG Panda Reach model for 500 000 timesteps",
    )

finally:
    if env is not None:
        env.close()
    if eval_env is not None:
        eval_env.close()
    wandb.finish()
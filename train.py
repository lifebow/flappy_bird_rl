import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from env import FlappyBirdEnv
import time
import torch
import numpy as np
import os

def make_env(rank, seed=0, render_mode=None, speed_up=False):
    """
    Utility function for multiprocessed env.
    
    :param rank: (int) index of the subprocess
    :param seed: (int) the initial seed for RNG
    :param render_mode: (str) render mode
    :param speed_up: (bool) if True, don't cap FPS
    """
    def _init():
        # Only rank 0 gets to render (if render_mode is set)
        show_window = (render_mode == "human" and rank == 0)
        env = FlappyBirdEnv(show_window=show_window, speed_up=speed_up)
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init

class RewardLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardLoggerCallback, self).__init__(verbose)
        self.episode_count = 0

    def _on_step(self) -> bool:
        # Check if any of the environments finished an episode
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_count += 1
                score = info.get("score", 0)
                reward = info["episode"]["r"]
                print(f"Episode {self.episode_count}: Score = {score}, Reward = {reward:.2f}")
        return True

class BestModelCallback(BaseCallback):
    """Callback to save the best model based on mean reward."""
    def __init__(self, save_path="best_model", verbose=0):
        super(BestModelCallback, self).__init__(verbose)
        self.save_path = save_path
        self.best_mean_reward = -float('inf')

    def _on_step(self) -> bool:
        # Check if we have episode info
        if len(self.model.ep_info_buffer) > 0:
            mean_reward = np.mean([ep['r'] for ep in self.model.ep_info_buffer])
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.model.save(self.save_path)
                print(f"\n*** New best model saved! Mean reward: {mean_reward:.2f} ***\n")
        return True

if __name__ == "__main__":
    num_cpu = 12  # Matched to CPU cores for optimal performance
    
    # Choose if you want to see training (rank 0 only)
    RENDER_MODE = 'human' 
    SPEED_UP = True # Set to True to training as fast as possible
    
    # Create the vectorized environment
    env = SubprocVecEnv([make_env(i, render_mode=RENDER_MODE, speed_up=SPEED_UP) for i in range(num_cpu)])
    
    # Add Monitor for better logging
    env = VecMonitor(env)

    # Device detection
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize or Load the agent
    model_path = "ppo_flappy_bird_new.zip"
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path} for fine-tuning...")
        model = PPO.load("ppo_flappy_bird_new", env=env, device=device)
    else:
        print("No existing model found. Starting training from scratch...")
        model = PPO(
            "MlpPolicy", 
            env, 
            verbose=1, 
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            device=device
        )

    print("Starting training...")
    
    # Create checkpoint directory
    os.makedirs("checkpoints", exist_ok=True)
    
    # Set up callbacks
    reward_callback = RewardLoggerCallback()
    best_model_callback = BestModelCallback(save_path="best_model")
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,  # Save every 50k steps
        save_path="checkpoints",
        name_prefix="ppo_checkpoint"
    )
    callbacks = CallbackList([reward_callback, best_model_callback, checkpoint_callback])
    
    try:
        model.learn(total_timesteps=1000000, callback=callbacks)
    except KeyboardInterrupt:
        print("Training interrupted.")
    
    # Save the agent
    model.save("ppo_flappy_bird_new")
    print("Model saved.")
    
    env.close()

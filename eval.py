import time
from stable_baselines3 import PPO
from env import FlappyBirdEnv

def evaluate():
    # Load the environment with show_window=True
    env = FlappyBirdEnv(show_window=True)
    
    # Load the trained model
    try:
        model = PPO.load("best_model")
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure 'ppo_flappy_bird_new.zip' exists by running 'train.py' first.")
        return

    obs, _ = env.reset()
    done = False
    total_reward = 0
    
    print("Starting evaluation. Press Ctrl+C to stop.")
    try:
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            
            if done or truncated:
                print(f"Episode finished. Reward: {total_reward:.2f}, Score: {info.get('score', 0)}")
                obs, _ = env.reset()
                total_reward = 0
                time.sleep(1) # Pause briefly before next episode
    except KeyboardInterrupt:
        print("Evaluation stopped.")
    finally:
        env.close()

if __name__ == "__main__":
    evaluate()

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from game import FlappyBirdGame

class FlappyBirdEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, show_window=False, speed_up=False):
        super(FlappyBirdEnv, self).__init__()
        self.show_window = show_window
        self.game = FlappyBirdGame(render_mode=show_window, speed_up=speed_up)
        
        # Action space: 0 (nothing), 1 (flap)
        self.action_space = spaces.Discrete(2)
        
        # Observation space:
        # 1. bird_y / height
        # 2. bird_vel / 15
        # 3. next_pipe_dist / width
        # 4. next_pipe_top_gap / height
        # 5. next_pipe_bottom_gap / height
        self.observation_space = spaces.Box(
            low=np.array([0.0, -2.0, 0.0, -1.0, -1.0]),
            high=np.array([1.0, 2.0, 1.0, 1.0, 1.0]),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        state = self.game.reset()
        return np.array(state, dtype=np.float32), {}

    def step(self, action):
        state, reward, done, score = self.game.step(action)
        if self.show_window:
            self.render()
        return np.array(state, dtype=np.float32), reward, done, False, {"score": score}

    def render(self):
        self.game.render()

    def close(self):
        self.game.close()

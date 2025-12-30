import pygame

# Window settings
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600
FPS = 60

# Physics constants
GRAVITY = 0.5
FLAP_STRENGTH = -8
BIRD_X = 50
PIPE_SPEED = 3
PIPE_GAP = 120
PIPE_SPAWN_RATE = 1500  # milliseconds

# Reward constants
REWARD_ALIVE = 0.1
REWARD_PASS = 1.0
REWARD_CRASH_MAX = -1.0

# Colors
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

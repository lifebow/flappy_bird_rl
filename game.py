import pygame
import random
from config import *

class FlappyBirdGame:
    def __init__(self, render_mode=False, speed_up=False):
        self.render_mode = render_mode
        self.speed_up = speed_up
        if self.render_mode:
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            self.clock = pygame.time.Clock()
            pygame.font.init()
            self.font = pygame.font.SysFont('Arial', 32)
        
        self.reset()

    def reset(self):
        self.bird_y = SCREEN_HEIGHT // 2
        self.bird_vel = 0
        self.pipes = []
        self.score = 0
        self.done = False
        self.last_pipe_dist = 0
        self._spawn_pipe()
        return self._get_state()

    def _spawn_pipe(self):
        gap_y = random.randint(100, SCREEN_HEIGHT - 100 - PIPE_GAP)
        self.pipes.append({
            'x': SCREEN_WIDTH,
            'gap_y': gap_y,
            'passed': False
        })
        self.last_pipe_dist = 0

    def step(self, action):
        if action == 1: # Flap
            self.bird_vel = FLAP_STRENGTH
        
        # Apply gravity
        self.bird_vel += GRAVITY
        self.bird_y += self.bird_vel
        
        # Update pipes and track distance for spawning
        self.last_pipe_dist += PIPE_SPEED
        for pipe in self.pipes:
            pipe['x'] -= PIPE_SPEED
        
        # Spawn new pipes based on distance (roughly 250-300 pixels apart)
        if self.last_pipe_dist >= 270:
            self._spawn_pipe()
            
        # Check for score
        reward = REWARD_ALIVE
        for pipe in self.pipes:
            if not pipe['passed'] and pipe['x'] + 50 < BIRD_X:
                pipe['passed'] = True
                self.score += 1
                reward = REWARD_PASS
        
        # Remove old pipes
        self.pipes = [p for p in self.pipes if p['x'] > -50]
        
        # Check collision
        if self._check_collision():
            self.done = True
            reward = self._calculate_crash_reward()
            
        return self._get_state(), reward, self.done, self.score

    def _check_collision(self):
        # Floor or Ceiling
        if self.bird_y < 0 or self.bird_y > SCREEN_HEIGHT:
            return True
        
        # Pipe collision
        bird_rect = pygame.Rect(BIRD_X, self.bird_y, 30, 30)
        for pipe in self.pipes:
            top_pipe = pygame.Rect(pipe['x'], 0, 50, pipe['gap_y'])
            bottom_pipe = pygame.Rect(pipe['x'], pipe['gap_y'] + PIPE_GAP, 50, SCREEN_HEIGHT)
            if bird_rect.colliderect(top_pipe) or bird_rect.colliderect(bottom_pipe):
                return True
        return False

    def _calculate_crash_reward(self):
        # Find the next pipe the bird is approaching
        next_pipe = None
        for pipe in self.pipes:
            if pipe['x'] + 50 > BIRD_X:
                next_pipe = pipe
                break
        
        if next_pipe:
            gap_center = next_pipe['gap_y'] + PIPE_GAP / 2
            distance = abs(self.bird_y - gap_center)
            max_dist = SCREEN_HEIGHT / 2
            # Reward is closer to 0 if closer to gap center, closer to -1 if far
            penalty = -1.0 * (min(distance, max_dist) / max_dist)
            return penalty
        return REWARD_CRASH_MAX

    def _get_state(self):
        next_pipe = None
        for pipe in self.pipes:
            if pipe['x'] + 50 > BIRD_X:
                next_pipe = pipe
                break
        
        if next_pipe:
            return [
                self.bird_y / SCREEN_HEIGHT,
                self.bird_vel / 15.0,
                (next_pipe['x'] - BIRD_X) / SCREEN_WIDTH,
                (next_pipe['gap_y'] - self.bird_y) / SCREEN_HEIGHT,
                (next_pipe['gap_y'] + PIPE_GAP - self.bird_y) / SCREEN_HEIGHT
            ]
        else:
            return [self.bird_y / SCREEN_HEIGHT, self.bird_vel / 15.0, 1.0, 0.0, 0.0]

    def render(self):
        if not self.render_mode:
            return
            
        # Process events to keep the window responsive
        pygame.event.pump()
        
        self.screen.fill(WHITE)
        # Draw Bird
        pygame.draw.rect(self.screen, BLUE, (BIRD_X, self.bird_y, 30, 30))
        
        # Draw Pipes
        for pipe in self.pipes:
            pygame.draw.rect(self.screen, GREEN, (pipe['x'], 0, 50, pipe['gap_y']))
            pygame.draw.rect(self.screen, GREEN, (pipe['x'], pipe['gap_y'] + PIPE_GAP, 50, SCREEN_HEIGHT))
        
        # Draw Score
        score_surface = self.font.render(f'Score: {self.score}', True, (0, 0, 0))
        self.screen.blit(score_surface, (10, 10))
            
        pygame.display.flip()
        if not self.speed_up:
            self.clock.tick(FPS)

    def close(self):
        if self.render_mode:
            pygame.quit()

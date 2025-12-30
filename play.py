import pygame
import sys
from game import FlappyBirdGame
from config import FPS

def main():
    game = FlappyBirdGame(render_mode=True)
    
    running = True
    while running:
        action = 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    action = 1
                if event.key == pygame.K_ESCAPE:
                    running = False

        state, reward, done, score = game.step(action)
        game.render()

        if done:
            print(f"Game Over! Final Score: {score}")
            game.reset()

    game.close()
    pygame.quit()

if __name__ == "__main__":
    main()

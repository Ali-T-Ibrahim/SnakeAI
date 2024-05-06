import gymnasium as gym
import numpy as np
import pygame
from enum import Enum
from gymnasium import spaces
from snake_game import SnakeGame


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


class SnakeEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    def __init__(self):
        super(SnakeEnv, self).__init__()
        self.game = SnakeGame()
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=3,
                                            shape=(768,), dtype=int)

    def step(self, action):
        # mapping action space to controller
        directions = [Direction.RIGHT, Direction.LEFT, Direction.UP, Direction.DOWN]
        self.game.direction = directions[action]

        # one step of game through SnakeGame
        game_over, score = self.game.play_step()

        # update observation
        self.game.update_state_matrix()
        self.render()

        # prepare return values
        observation = self.game.state_matrix
        reward = score
        done = game_over
        timeout = False
        info = {}

        return observation.flatten(), reward, done, timeout, info

    def reset(self, seed=None):
        info = {}
        self.game.reset()
        return self.game.state_matrix.flatten(), info  # reward, done, info can't be included

    def render(self, mode='human'):
        if mode == 'human':
            self.game.update_ui()

    def close(self):
        pygame.quit()

import gym
import numpy as np
import pygame
from enum import Enum
from gym import spaces
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
                                            shape=(self.game.grid_h, self.game.grid_w), dtype=np.int)

    def step(self, action):
        # mapping action space to controller
        directions = [Direction.RIGHT, Direction.LEFT, Direction.UP, Direction.DOWN]
        self.game.direction = directions[action]

        # one step of game through SnakeGame
        game_over, score = self.game.play_step()

        # update observation
        self.game.update_state_matrix()

        # prepare return values
        observation = self.game.state_matrix
        reward = score
        done = game_over
        info = {}

        return observation, reward, done, info

    def reset(self):
        self.game.__init__()
        self.game.update_state_matrix()
        return self.game.state_matrix  # reward, done, info can't be included

    def render(self, mode='human'):
        if mode == 'human':
            self.game.update_ui()

    def close(self):
        pygame.quit()
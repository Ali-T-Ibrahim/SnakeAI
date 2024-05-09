import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import random
import time
from collections import deque

MAX_LEN_GOAL = 5


def collision_with_apple(apple_position, score):
    apple_position = [random.randrange(1, 26) * 10, random.randrange(1, 26) * 10]
    score += 1
    return apple_position, score


def collision_with_boundaries(snake_head):
    if snake_head[0] >= 250 or snake_head[0] < 0 or snake_head[1] >= 250 or snake_head[1] < 0:
        return 1
    else:
        return 0


def collision_with_self(snake_position):
    snake_head = snake_position[0]
    if snake_head in snake_position[1:]:
        return 1
    else:
        return 0


class SnakeEnv3(gym.Env):
    """Custom Environment that follows gym interface"""

    def __init__(self):
        super(SnakeEnv3, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(4)
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(low=-500, high=500,
                                            shape=(MAX_LEN_GOAL + 5,), dtype=np.int32)

    def step(self, action):
        self.frame += 1

        initial_penalty = -50
        scale = 1
        penalty = initial_penalty + scale * np.log1p(self.frame)
        penalty = min(penalty, -5)

        cv2.imshow('a', self.img)
        cv2.waitKey(1)
        self.img = np.zeros((260, 260, 3), dtype='uint8')
        # Display Apple
        cv2.rectangle(self.img, (self.apple_position[0], self.apple_position[1]),
                      (self.apple_position[0] + 10, self.apple_position[1] + 10),
                      (0, 0, 255), 3)
        # Display Snake
        for position in self.snake_position:
            cv2.rectangle(self.img, (position[0], position[1]), (position[0] + 10, position[1] + 10), (0, 255, 0), 3)

        # Takes step after fixed time
        t_end = time.time() + 0.05
        k = -1
        while time.time() < t_end:
            if k == -1:
                k = cv2.waitKey(1)
            else:
                continue

        # 0-Left, 1-Right, 3-Up, 2-Down, q-Break
        # a-Left, d-Right, w-Up, s-Down

        # Change the head position based on the button direction
        if action == 1:
            self.snake_head[0] += 10
        elif action == 0:
            self.snake_head[0] -= 10
        elif action == 2:
            self.snake_head[1] += 10
        elif action == 3:
            self.snake_head[1] -= 10

        apple_reward = 0
        # Increase Snake length on eating apple
        if self.snake_head == self.apple_position:
            self.apple_position, self.score = collision_with_apple(self.apple_position, self.score)
            self.snake_position.insert(0, list(self.snake_head))
            apple_reward = 100

        else:
            self.snake_position.insert(0, list(self.snake_head))
            self.snake_position.pop()

        # On collision kill the snake and print the score
        death_penalty = 0
        if collision_with_boundaries(self.snake_head) == 1 or collision_with_self(self.snake_position) == 1:
            font = cv2.FONT_HERSHEY_SIMPLEX
            self.img = np.zeros((260, 260, 3), dtype='uint8')
            cv2.putText(self.img, 'Your Score is {}'.format(self.score), (140, 250), font, 1, (255, 255, 255), 2,
                        cv2.LINE_AA)
            cv2.imshow('a', self.img)
            death_penalty = penalty

            self.done = True

        distance = np.linalg.norm(np.array(self.snake_head) - np.array(self.apple_position))

        self.reward = ((128 - distance) / 100) + apple_reward + death_penalty

        # observation
        # head_y, head_x, apple_y, apple_x, snake_len, prev_moves

        head_y = self.snake_head[0]
        head_x = self.snake_head[1]
        apple_y = self.apple_position[0]
        apple_x = self.apple_position[1]
        snake_len = self.score + 3

        self.prev_actions = deque(maxlen=MAX_LEN_GOAL)
        for _ in range(MAX_LEN_GOAL):
            self.prev_actions.append(-1)

        self.observation = [head_y, head_x, apple_y, apple_x, snake_len] + list(self.prev_actions)
        self.observation = np.array(self.observation)

        info = {}
        truncated = False

        # print(self.reward)
        return self.observation, self.reward, self.done, truncated, info

    def reset(self, seed=None):
        self.done = False
        self.frame = 0
        self.img = np.zeros((260, 260, 3), dtype='uint8')
        # Initial Snake and Apple position

        # randomize start position
        start_position = {
            0: [[130, 130], [140, 130], [150, 130]],  # facing left
            1: [[130, 130], [120, 130], [110, 130]],  # facing right
            2: [[130, 130], [130, 140], [130, 150]],  # facing up
            3: [[130, 130], [130, 120], [130, 110]]  # facing down
        }

        # choose start position
        pos_selection = random.choice([0, 1, 2, 3])
        self.snake_position = start_position[pos_selection]

        self.apple_position = [random.randrange(1, 26) * 10, random.randrange(1, 26) * 10]
        self.score = 0
        self.reward = 0
        self.prev_button_direction = pos_selection
        self.button_direction = pos_selection
        self.snake_head = [130, 130]

        # observation
        # head_y, head_x, apple_y, apple_x, snake_len, prev_moves

        head_y = self.snake_head[0]
        head_x = self.snake_head[1]
        apple_y = self.apple_position[0]
        apple_x = self.apple_position[1]
        snake_len = self.score + 3

        self.prev_actions = deque(maxlen=MAX_LEN_GOAL)
        for _ in range(MAX_LEN_GOAL):
            self.prev_actions.append(-1)

        self.observation = [head_y, head_x, apple_y, apple_x, snake_len] + list(self.prev_actions)
        self.observation = np.array(self.observation)

        info = {}
        return self.observation, info  # reward, done, info can't be included

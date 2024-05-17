from stable_baselines3 import PPO
import gymnasium as gym
import os
from snake_env3 import SnakeEnv3
import time

model_code = 1715634279
step = 21460000

models_dir = f"models/{model_code}"
model_path = f"{models_dir}/{step}.zip"

env = SnakeEnv3()
env.reset()

model = PPO.load(model_path, env=env, tensorboard_log=models_dir)

episodes = 50

for episode in range(episodes):
    done = False
    obs = env.reset()
    while not done:  #not done:
        action, _ = model.predict(obs)
        # print("action", random_action)
        obs, reward, done, truncated ,info = env.step(action)
        # print('reward', reward)
env.close()

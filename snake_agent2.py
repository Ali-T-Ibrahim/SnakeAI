from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
import os
from snake_env4 import SnakeEnv4
import time

models_dir = f"models/{int(time.time())}/"
logdir = f"logs/{int(time.time())}/"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

env = SnakeEnv4()
env.reset()

model = PPO('MlpPolicy', env, verbose=2, tensorboard_log=logdir)

TIMESTEPS = 10000
iters = 0
while True:
    iters += 1
    model.learn(total_timesteps=TIMESTEPS,
                reset_num_timesteps=False,
                tb_log_name=f"PPO")

    model.save(f"{models_dir}/{TIMESTEPS * iters}")

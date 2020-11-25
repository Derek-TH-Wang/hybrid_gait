import gym

import os
import time
import random
import numpy as np
from mpi4py import MPI

from stable_baselines import PPO1
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.callbacks import CheckpointCallback


def main(output_dir="output"):
    save_path = os.path.join(output_dir, "model.zip")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    num_procs = MPI.COMM_WORLD.Get_size()

    env = gym.make('Pendulum-v0')

    agent = PPO1(MlpPolicy, env, tensorboard_log="output", verbose=1)
    # agent.load_parameters()
    agent.learn(total_timesteps=int(10000/num_procs), callback=[], save_path=save_path)

    obs = env.reset()
    for i in range(1000):
        action, _states = agent.predict(obs)
        # print(action)
        obs, rewards, dones, info = env.step(action)
        if MPI.COMM_WORLD.Get_rank() == 0:
            env.render()

    env.close()


if __name__ == '__main__':
    main()
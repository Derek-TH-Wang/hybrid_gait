import gym

import os
import time
import yaml
import random
import argparse
import numpy as np
from mpi4py import MPI
import tensorflow as tf

from stable_baselines import PPO1
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.callbacks import CheckpointCallback



def build_env():
    env = gym.make('Pendulum-v0')
    return env


def build_agent(env, num_procs, timesteps_per_actorbatch, optim_batchsize, output_dir):
    policy_kwargs = {
        "net_arch": [{"pi": [64, 64],
                      "vf": [64, 64]}],
        "act_fun": tf.nn.relu
    }
    timesteps_per_actorbatch = int(
        np.ceil(float(timesteps_per_actorbatch) / num_procs))
    optim_batchsize = int(np.ceil(float(optim_batchsize) / num_procs))

    agent = PPO1(MlpPolicy, env, 
                 timesteps_per_actorbatch=timesteps_per_actorbatch,
                 optim_batchsize=optim_batchsize,
                 policy_kwargs=policy_kwargs,
                 tensorboard_log=output_dir, 
                 verbose=1)
    return agent


def train(agent, env, num_train_episodes, output_dir="", int_save_freq=0):
    if (output_dir == ""):
        save_path = None
    else:
        save_path = os.path.join(output_dir, "model.zip")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    callbacks = []
    # Save a checkpoint every n steps
    if (output_dir != ""):
        if (int_save_freq > 0):
            int_dir = os.path.join(output_dir, "intermedate")
            callbacks.append(CheckpointCallback(save_freq=int_save_freq, 
                                                save_path=int_dir, 
                                                name_prefix='model'))

    agent.learn(total_timesteps=num_train_episodes, 
                save_path=save_path, 
                callback=callbacks)
    
    return


def test(agent, env, num_procs, num_episodes=None):

    if num_episodes is not None:
        num_local_episodes = int(np.ceil(float(num_episodes) / num_procs))
    else:
        num_local_episodes = np.inf

    obs = env.reset()
    for _ in range(num_local_episodes):
        action, _states = agent.predict(obs)
        # print(action)
        obs, rewards, dones, info = env.step(action)
        if MPI.COMM_WORLD.Get_rank() == 0:
            env.render()
    env.close()
    return


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--task", dest="task", type=str, default="hybrid_gait")
    args = arg_parser.parse_args()

    with open('hybrid_gait/config/trainning_params.yaml') as f:
        training_params_dict = yaml.safe_load(f)
        if args.task in list(training_params_dict.keys()):
            training_params = training_params_dict[args.task]
        else:
            raise ValueError(
                "task not found for pybullet_sim_config.yaml")

    num_procs = MPI.COMM_WORLD.Get_size()

    env = build_env()
    agent = build_agent(env=env,
                        num_procs=num_procs,
                        timesteps_per_actorbatch=training_params["timesteps_per_actorbatch"],
                        optim_batchsize=training_params["optim_batchsize"],
                        output_dir=training_params["output_dir"])

    if training_params["model_file"] != "":
        agent.load_parameters(training_params["model_file"])

    if training_params["mode"] == "train":
        train(agent=agent,
              env=env,
              num_train_episodes=training_params["num_train_episodes"],
              output_dir=training_params["output_dir"],
              int_save_freq=training_params["int_save_freq"])
    elif training_params["mode"] == "test":
        test(agent=agent,
             env=env,
             num_procs=num_procs,
             num_episodes=training_params["num_test_episodes"])
    else:
        assert False, "Unsupported mode: " + training_params["mode"]

    return


if __name__ == '__main__':
    main()
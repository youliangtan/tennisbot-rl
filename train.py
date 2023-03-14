#!/usr/bin/env python3

import gym
import torch
from agent import TRPOAgent
import tennisbot
import time
from stable_baselines3 import PPO
import argparse
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import EvalCallback



tmp_path = "./tmp/ppo/"

# new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])


def main():

    # Use PPO agent
    # https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html

    env = gym.make('Tennisbot-v0')
    model = PPO("MlpPolicy", env, verbose=0,tensorboard_log=tmp_path)
    # model = PPO.load("ppo_agent")
    # model.set_logger(new_logger)
    # eval_callback = EvalCallback(env, best_model_save_path=tmp_path,
    #                             log_path=tmp_path, eval_freq=500,
    #                             deterministic=True, render=False)

    for i in range(1000):
        print("iteration: ", i)
        model.learn(total_timesteps=2200,callback=None)
        env.reset()
        if i%100 == 99:
            print(f"saving {i+1}th file")
            model.save("ppo_agent")





if __name__ == '__main__':
    main()

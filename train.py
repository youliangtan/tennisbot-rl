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
LOAD_MODEL = True

def main(args):

    # Use PPO agent
    # https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
    total_timesteps = 25000
    env = gym.make('Tennisbot-v0')
    model = PPO("MlpPolicy", env, verbose=0,tensorboard_log=tmp_path)
    print(model.policy)

    if args.load:
        model.load(tmp_path+'ppo_agent.zip')

    # model.set_logger(new_logger)
    eval_callback = EvalCallback(env, best_model_save_path=tmp_path,
                                log_path=tmp_path, eval_freq=total_timesteps,
                                deterministic=True, render=False)

    for i in range(1000):
        print("iteration: ", i)
        model.learn(total_timesteps=total_timesteps,callback=eval_callback)
        env.reset()
        if i%100 == 99:
            print(f"saving {i+1}th file")
            model.save("ppo_agent.zip")

##############################################################################

if __name__ == '__main__':
    print("start running")
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', action='store_true', help="load model")
    args = parser.parse_args()
    main(args)

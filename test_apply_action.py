#!/usr/bin/env python3

import gym
import torch
from agent import TRPOAgent
import tennisbot
import time
from stable_baselines3 import PPO
import argparse





def main():

    # Use PPO agent
    # https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
    # env = gym.make('Tennisbot-v0')
    # model = PPO("MlpPolicy", env, verbose=0,tensorboard_log="./ppo_log/")
    env = gym.make('Tennisbot-v0')
    ob = env.reset()
    action = [1,0]

    while True:
        ob, _, done, _ = env.step(action)
        print(ob[:3])
        env.render()
        time.sleep(1/240)
        # if done:
        #     ob = env.reset()
        #     time.sleep(1)



if __name__ == '__main__':
    main()

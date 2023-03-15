#!/usr/bin/env python3

import gym
import torch
from agent import TRPOAgent
import tennisbot
import time
from stable_baselines3 import PPO
import argparse



tmp_path = "./tmp/ppo/"


def main():

    # Use PPO agent
    # https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
    # env = gym.make('Tennisbot-v0')
    # model = PPO("MlpPolicy", env, verbose=0,tensorboard_log="./ppo_log/")
    env = gym.make('Tennisbot-v0')

    model =PPO.load(tmp_path + "best_model")

    print("start running")
    
    ob = env.reset()
    while True:
        action,_states = model.predict(ob)
        ob, _, done, _ = env.step(action)
        env.render()
        time.sleep(0.01)
        if done:
            ob = env.reset()
            time.sleep(1)



if __name__ == '__main__':
    main()

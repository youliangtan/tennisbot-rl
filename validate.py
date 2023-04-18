#!/usr/bin/env python3

import gym
import torch
import tennisbot
import time
from stable_baselines3 import PPO
# from stable_baselines3 import SAC
import argparse

# tmp_path = "./tmp/ppo/"

def main():

    # Use PPO agent
    # https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
    # env = gym.make('Tennisbot-v0')
    # model = PPO("MlpPolicy", env, verbose=0,tensorboard_log="./ppo_log/")
    env = gym.make('Tennisbot-v0')

    model = PPO.load("model/ppo/best_model.zip")
    # model = SAC.load("sac_agent_80.zip")

    print("------------- start running -------------")
    
    ob = env.reset()
    while True:
        action,_states = model.predict(ob)
        ob, _, done, _ = env.step(action)
        # print("reward", reward)
        env.render("human")
        # time.sleep(1/240)
        if done:
            ob = env.reset()
            time.sleep(1)



if __name__ == '__main__':
    main()

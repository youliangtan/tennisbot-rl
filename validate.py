#!/usr/bin/env python3

import gym
import torch
import tennisbot
import time
from stable_baselines3 import PPO, SAC

import argparse

##############################################################################

def main(args):
    env = gym.make('Tennisbot-v0')

    if (args.select == 'ppo'):
        model = PPO.load("model/ppo/best_model.zip")
    elif (args.select == 'sac'):
        model = SAC.load("model/sac/best_model.zip")

    print("------------- start running -------------")
    
    ob = env.reset()
    while True:
        action,_states = model.predict(ob)
        ob, _, done, _ = env.step(action)
        # print("action", action)
        env.render("human")
        # time.sleep(1/240)
        if done:
            ob = env.reset()
            time.sleep(1)

##############################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--select',
                        type=str, default='ppo',
                        help="select model: ppo or sac")
    args = parser.parse_args()
    main(args)

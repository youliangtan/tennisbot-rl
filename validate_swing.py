#!/usr/bin/env python3

import gym
import torch
import tennisbot
import time
from stable_baselines3 import PPO, SAC

import argparse

##############################################################################

def main(args):
    
    if args.headless:
        env = gym.make('SwingRacket-v0', use_gui=False, delay_mode=False)
    else:
        env = gym.make('SwingRacket-v0', use_gui=True, delay_mode=True)

    if args.model_file:
        if args.select == 'ppo':
            model = PPO.load(args.model_file)
        elif args.select == 'sac':
            model = SAC.load(args.model_file)
    elif (args.select == 'ppo'):
        model = PPO.load("model/ppo_swing/best_model.zip")
    elif (args.select == 'sac'):
        model = SAC.load("model/ppo_swing/best_model.zip")

    print("------------- start running -------------")
    ob = env.reset()

    count = 1
    while True:
        action,_states = model.predict(ob)
        ob, _, done, _ = env.step(action)
        # print("action", action)

        if not args.headless:
            time.sleep(1/240)

        if done:
            print("-------------------- Done ", count, "-------------------")
            count += 1               
            ob = env.reset()

##############################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--select',
                        type=str, default='ppo',
                        help="select model: ppo or sac")
    parser.add_argument('-m', '--model_file',
                        type=str, default='',
                        help="model file name")
    parser.add_argument('--headless', action='store_true')

    args = parser.parse_args()
    main(args)

#!/usr/bin/env python3

import gym
import torch
import tennisbot
import time
from stable_baselines3 import PPO
from stable_baselines3 import SAC
import argparse
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import EvalCallback

tmp_path_ppo = "./tmp/ppo/"
tmp_path_sac = "./tmp/sac/"

# new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
# LOAD_MODEL = False

def main(args):

    # Use PPO agent
    # https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
    total_timesteps = 5000
    env = gym.make('Tennisbot-v0')
    # model = PPO("MlpPolicy", env, verbose=0, tensorboard_log=tmp_path_ppo, batch_size=2048)
    model = SAC("MlpPolicy", env, verbose=0, tensorboard_log=tmp_path_sac)
    
    # if args.load:
    #     model.load(tmp_path+'ppo_agent.zip')
    model.load('sac_agent_20.zip')

    # model.set_logger(new_logger)
    eval_callback = EvalCallback(env, best_model_save_path=tmp_path_sac,
                                log_path=tmp_path_sac, eval_freq=total_timesteps,
                                deterministic=False, render=False)

    for i in range(1, 1000):
        print("---------- Epoch: ", i, "----------")
        model.learn(total_timesteps=total_timesteps,callback=eval_callback, progress_bar=False)
        # model.learn(total_timesteps=total_timesteps)
        env.reset()
        if i%20 == 0:
            print(f"saving {i+1}th file")
            # model.save("ppo_agent.zip")
            model_file = "sac_agent_"+str(i)+".zip"
            model.save(model_file)

##############################################################################

if __name__ == '__main__':
    print("start running")
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', action='store_true', help="load model")
    args = parser.parse_args()
    main(args)

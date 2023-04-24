#!/usr/bin/env python3

import gym
import torch
import tennisbot
import time
from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3.common.noise import NormalActionNoise
import argparse
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import EvalCallback

# MODEL = "PPO"
MODEL = "SAC"

tmp_path_ppo = "./tmp/ppo/"
tmp_path_sac = "./tmp/sac/"

model_save_path_ppo = "./model/ppo/"
model_save_path_sac = "./model/sac/"

def main(args):

    env = gym.make('Tennisbot-v0')
    
    if MODEL == "PPO":
        # Use PPO agent
        # https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
        total_timesteps = 2e6
        evaluation_frequency = 1000
        n_epochs = int(total_timesteps / evaluation_frequency)
        batch_size = 128
        rollout_steps = 1024
        model = PPO("MlpPolicy", env, verbose=0, tensorboard_log=tmp_path_ppo, batch_size=batch_size, n_steps=rollout_steps, n_epochs=n_epochs)

        # if args.load:
        #     model.load('model/ppo/best_model.zip')
    
        # model.set_logger(new_logger)
        eval_callback = EvalCallback(env, best_model_save_path=model_save_path_ppo,
                                log_path=tmp_path_ppo, eval_freq=evaluation_frequency,
                                deterministic=False, render=False)

        model.learn(total_timesteps=total_timesteps,callback=eval_callback, progress_bar=True)
        
        
    if MODEL == "SAC":
        total_timesteps = 1e6
        buffer_size = int(1e6)
        batch_size = 128
        train_freq = (5, "step")
        evaluation_frequency = 1000
        action_noise = NormalActionNoise(0, 0.001)
        gradient_steps = 1
        model = SAC("MlpPolicy", env, verbose=0, 
                    tensorboard_log=tmp_path_sac, 
                    action_noise=action_noise, 
                    buffer_size=buffer_size, 
                    batch_size=batch_size,
                    gradient_steps=gradient_steps,
                    train_freq=train_freq)
        
        if args.load:
            model.load('model/ppo/best_model.zip')

        eval_callback = EvalCallback(env, best_model_save_path=model_save_path_sac,
                                log_path=tmp_path_sac, eval_freq=evaluation_frequency,
                                deterministic=False, render=False)

        model.learn(total_timesteps=total_timesteps,callback=eval_callback, progress_bar=True)

##############################################################################

if __name__ == '__main__':
    print("start running")
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', action='store_true', help="load model")
    args = parser.parse_args()
    main(args)

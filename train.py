#!/usr/bin/env python3

import gym
import torch.nn as nn
from agent import TRPOAgent
import tennisbot
import time
from stable_baselines3 import PPO
from stable_baselines3 import SAC
import argparse
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import EvalCallback

tmp_path_ppo = "./tmp/ppo/"
tmp_path_sac = "./tmp/sac/"

model_save_path_ppo = "./model/ppo/"

##############################################################################

class CustomNetwork(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(CustomNetwork, self).__init__()
        self.fc1 = nn.Linear(input_shape, 64)
        self.custom_layer = nn.Linear(64, 32) # add your custom layer here
        self.fc2 = nn.Linear(32, output_shape)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.custom_layer(x))
        x = self.fc2(x)
        return x

##############################################################################

def main(args):

    # Use PPO agent
    # https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
    total_timesteps = 5e5
    evaluation_frequency = 500
    n_epochs = int(total_timesteps / evaluation_frequency)
    batch_size = 1024
    rollout_steps = 1024
    env = gym.make('Tennisbot-v0')

    if (args.select == 'ppo'):
        model = PPO("MlpPolicy", env,
                    verbose=0,
                    tensorboard_log=tmp_path_ppo,
                    batch_size=batch_size, 
                    n_steps=rollout_steps,
                    n_epochs=n_epochs)

    elif (args.select == 'sac'):
        model = SAC("MlpPolicy", env, verbose=0, tensorboard_log=tmp_path_sac)

    else:
        print("Please select a valid agent: ppo or sac")
        return
        # TODO: Work-in-progress
        # model = PPO(CustomNetwork(env.observation_space.shape[0], env.action_space.shape[0]), env,
        #             verbose=0,
        #             tensorboard_log=tmp_path_ppo,
        #             batch_size=batch_size, 
        #             n_steps=rollout_steps,
        #             n_epochs=n_epochs)

    print(model.policy)
   
    # if args.load:
    #     model.load(tmp_path+'ppo_agent.zip')
    # model.load('model/ppo/best_model.zip')

    # model.set_logger(new_logger)
    eval_callback = EvalCallback(env, best_model_save_path=model_save_path_ppo,
                                log_path=tmp_path_ppo, eval_freq=total_timesteps/n_epochs,
                                deterministic=False, render=False)


    model.learn(total_timesteps=total_timesteps,callback=eval_callback, progress_bar=True)

##############################################################################

if __name__ == '__main__':
    print("start running")
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', action='store_true', help="load model")
    parser.add_argument('--render', action='store_true', help="render")
    parser.add_argument('-s', '--select', type=str, default='ppo', help="select model: ppo or sac")
    args = parser.parse_args()
    main(args)

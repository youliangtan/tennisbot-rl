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
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sb3_contrib import TQC, TRPO

tmp_path_ppo = "./tmp/ppo_swing/"
tmp_path_sac = "./tmp/sac_swing/"
tmp_path_tqc = "./tmp/tqc_swing/"
tmp_path_trpo = "./tmp/trpo_swing/"

##############################################################################

class CustomFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.box.Box, output_shape):
        super().__init__(observation_space, output_shape)

        input_shape = observation_space.shape[0]
        self.layers = nn.Sequential(
            nn.Linear(input_shape, 64),
            nn.ReLU(),
            nn.Linear(64, output_shape),
            nn.ReLU(),
       )

    def forward(self, x):
        return self.layers(x)

##############################################################################


def main(args):

    # Use PPO agent
    # https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
    total_timesteps = 20e5
    evaluation_frequency = 500
    n_epochs = int(total_timesteps / evaluation_frequency)
    batch_size = 1100
    rollout_steps = 1100
    env = gym.make('SwingRacket-v0', use_gui=args.gui)

    # model_path
    if ('ppo' in args.select):
        model_save_path = "./model/ppo_swing/"
    elif ('sac' in args.select):
        model_save_path = "./model/sac_swing/"
    elif ('tqc' in args.select):
        model_save_path = "./model/tqc_swing/"
    elif ('trpo' in args.select):
        model_save_path = "./model/trpo_swing/"
    else:
        print("Please select a valid agent: ppo or sac")
        return

    if args.load:
        if ('ppo' in args.select):
            print("loading previously trained PPO model")
            model = PPO.load(model_save_path + "best_model.zip", env=env,
                            tensorboard_log=tmp_path_ppo,
                            batch_size=batch_size,
                            learning_rate=1e-3,)
        elif (args.select == 'sac'):
            # TODO: load sac model
            pass

    elif (args.select == 'ppo'):
        policy_kwargs = dict(
                net_arch=dict(pi=[32, 64, 32], vf=[32, 64, 32])  # actor and critic network arch
                )
        model = PPO("MlpPolicy", env,
                    verbose=0,
                    tensorboard_log=tmp_path_ppo,
                    batch_size=batch_size,
                    ent_coef=0.002,
                    # gamma=0.95,
                    n_steps=rollout_steps,
                    policy_kwargs=policy_kwargs
                )

    elif (args.select == 'sac'):
        model = SAC("MlpPolicy", env,
                    batch_size=batch_size,
                    verbose=0, tensorboard_log=tmp_path_sac)

    elif (args.select == 'tqc'):
        model = TQC("MlpPolicy", env, verbose=0, tensorboard_log=tmp_path_tqc)
    
    elif (args.select == 'trpo'):
        model = TRPO("MlpPolicy", env, verbose=0, tensorboard_log=tmp_path_trpo)

    else:
        print("Please select a valid agent: ppo, tqc, trpo or sac")
        return

    print(model.policy)

    # Create callbacks
    eval_callback = EvalCallback(env, best_model_save_path=model_save_path,
                                 log_path=tmp_path_ppo,
                                 eval_freq=1000,
                                 deterministic=False, render=False)
    checkpoint_cb = CheckpointCallback(save_freq=10000,
                                       save_path=model_save_path,
                                       name_prefix='rl_model')

    callback_list = [eval_callback, checkpoint_cb]

    model.learn(total_timesteps=total_timesteps,
                callback=callback_list, progress_bar=True)


##############################################################################


if __name__ == '__main__':
    print("start running")
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', action='store_true',
                        help="load model")
    parser.add_argument('--gui', action='store_true',
                        help="show pybullet gui")
    parser.add_argument('--curri', action='store_true',
                        help="curriculum learning size change of racket")
    parser.add_argument('-s', '--select', type=str, default='ppo',
                        help="select model: ppo or sac or tuned_ppo")
    args = parser.parse_args()
    main(args)

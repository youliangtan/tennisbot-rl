#!/usr/bin/env python3

"""
Default PPO network
ActorCriticPolicy(
  (features_extractor): FlattenExtractor(
    (flatten): Flatten(start_dim=1, end_dim=-1)
  )
  (pi_features_extractor): FlattenExtractor(
    (flatten): Flatten(start_dim=1, end_dim=-1)
  )
  (vf_features_extractor): FlattenExtractor(
    (flatten): Flatten(start_dim=1, end_dim=-1)
  )
  (mlp_extractor): MlpExtractor(
    (shared_net): Sequential()
    (policy_net): Sequential(
      (0): Linear(in_features=12, out_features=64, bias=True)
      (1): Tanh()
      (2): Linear(in_features=64, out_features=64, bias=True)
      (3): Tanh()
    )
    (value_net): Sequential(
      (0): Linear(in_features=12, out_features=64, bias=True)
      (1): Tanh()
      (2): Linear(in_features=64, out_features=64, bias=True)
      (3): Tanh()
    )
  )
  (action_net): Linear(in_features=64, out_features=3, bias=True)
  (value_net): Linear(in_features=64, out_features=1, bias=True)
)
"""

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
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

tmp_path_ppo = "./tmp/ppo/"
tmp_path_sac = "./tmp/sac/"

model_save_path = "./model/ppo/"

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
    total_timesteps = 10e5
    evaluation_frequency = 500
    n_epochs = int(total_timesteps / evaluation_frequency)
    batch_size = 1024
    rollout_steps = 1024
    env = gym.make('Tennisbot-v0', use_gui=args.gui)

    # model_path
    if ('ppo' in args.select):
        model_save_path = "./model/ppo/"
    elif ('sac' in args.select):
        model_save_path = "./model/sac/"
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
        model = PPO("MlpPolicy", env,
                    verbose=0,
                    tensorboard_log=tmp_path_ppo,
                    batch_size=batch_size,
                    n_steps=rollout_steps,
                    n_epochs=n_epochs)

    elif (args.select == 'tuned_ppo'):
        policy_kwargs = dict(
            activation_fn=nn.ReLU,
            features_extractor_class=CustomFeaturesExtractor,
            features_extractor_kwargs=dict(
                    output_shape=env.action_space.shape[0]
                ),
            net_arch=dict(pi=[32, 64, 32], vf=[32, 64, 32])  # actor and critic network arch
        )
        model = PPO("MlpPolicy", env,
                    verbose=0,
                    tensorboard_log=tmp_path_ppo,
                    batch_size=batch_size,
                    n_steps=rollout_steps,
                    n_epochs=n_epochs,
                    policy_kwargs=policy_kwargs)

    elif (args.select == 'sac'):
        model = SAC("MlpPolicy", env, verbose=0, tensorboard_log=tmp_path_sac)
    else:
        print("Please select a valid agent: ppo or sac")
        return

    print(model.policy)

    eval_callback = EvalCallback(env, best_model_save_path=model_save_path,
                                 log_path=tmp_path_ppo, eval_freq=total_timesteps/n_epochs,
                                 deterministic=False, render=False)

    model.learn(total_timesteps=total_timesteps, callback=eval_callback, progress_bar=True)

##############################################################################


if __name__ == '__main__':
    print("start running")
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', action='store_true', help="load model")
    parser.add_argument('--gui', action='store_true', help="show pybullet gui")
    parser.add_argument('-s', '--select', type=str, default='ppo', help="select model: ppo or sac")
    args = parser.parse_args()
    main(args)

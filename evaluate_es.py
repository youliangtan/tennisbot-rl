#!/usr/bin/env python3

"""
Source repo: https://github.com/enajx/ES
"""

import gym
import time
import argparse
import sys
import torch
from os.path import join, exists
from os import mkdir
import gym
from gym.spaces import Discrete, Box

import numpy as np
import pybullet_envs
import tennisbot

from tennisbot.ES.evolution_strategy_static import EvolutionStrategyStatic, Normalizer
from tennisbot.ES.policies import MLP, CNN, GatedCNN
from tennisbot.ES.fitness_functions import fitness_static

gym.logger.set_level(40)


##############################################################################

ROUNDS = 50

##############################################################################

def evaluate_static(
            environment: str, render: bool, evolved_parameters: [np.array]
        ):
    """
    Evaluate an agent 'evolved_parameters' in an environment 'environment' during a lifetime.
    """
    # TODO: 12 is hardcoded, should be changed
    rew_ep = fitness_static(evolved_parameters, environment, Normalizer(6), render)

    print('\n Episode cumulative rewards ', int(rew_ep))
    return rew_ep

def main(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--environment', type=str, default='CarRacing-v0', metavar='',
                        help='Gym environment: any OpenAI Gym may be used')
    parser.add_argument('--path_weights', type=str, help='path to the evolved weights')
    parser.add_argument('--render', type=bool, default=1, help='path to the evolved weights')

    args = parser.parse_args()
    evolved_parameters = torch.load(args.path_weights)

    # Run the environment
    cum_rewards = 0
    for i in range(ROUNDS):
        cum_rewards += evaluate_static(args.environment, args.render, evolved_parameters)

    print('Average cumulative reward over {} rounds: {}'.format(ROUNDS, cum_rewards / ROUNDS))


##############################################################################

if __name__ == '__main__':
    main(sys.argv)

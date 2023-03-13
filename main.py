#!/usr/bin/env python3

import gym
import torch
from agent import TRPOAgent
import tennisbot
import time
from stable_baselines3 import PPO
import argparse


class Strategy:
    TRPO = 0
    PPO = 1

##############################################################################


def main(args):
    """
    Run the agent
    """

    # Use default custom TRPO agent
    if args.strategy == Strategy.TRPO:
        obs_size = 9
        action_size = 2
        nn = torch.nn.Sequential(
            torch.nn.Linear(obs_size, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, action_size)
        )
        agent = TRPOAgent(policy=nn)

        if args.load:
            print("loading previously trained TRPO model")
            agent.load_model("agent.pth")

        agent.train("Tennisbot-v0", seed=0, batch_size=5000,
                    iterations=args.iteration,
                    max_episode_length=2500, verbose=True)
        agent.save_model("agent.pth")

    # Use PPO agent
    # https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
    elif args.strategy == Strategy.PPO:
        env = gym.make('Tennisbot-v0')
        model = PPO("MlpPolicy", env, verbose=0,tensorboard_log="./ppo_log/")

        if args.load:
            print("loading previously trained PPO model")
            model.load("ppo_agent")

        for i in range(args.iteration):
            print("iteration: ", i)
            model.learn(total_timesteps=2500)
            env.reset()
        model.save("ppo_agent")
    else:
        raise ValueError("Invalid strategy")

    print("start running")
    env = gym.make('Tennisbot-v0')
    ob = env.reset()
    # while True:
    #     action = agent(ob)
    #     ob, _, done, _ = env.step(action)
    #     # env.render()
    #     if done:
    #         ob = env.reset()
    #         time.sleep(1/30)

##############################################################################


if __name__ == '__main__':
    print("start running")
    parser = argparse.ArgumentParser()
    # parser.add_argument('--gui', action='store_true')
    # parser.add_argument('--delay', action='store_true')
    parser.add_argument('-i', '--iteration', type=int, default=10)
    parser.add_argument('--load', action='store_true', help="load model")
    parser.add_argument('-s', '--strategy', type=int, default=0,
                        help="0: for trpo, 1: for ppo")
    args = parser.parse_args()

    main(args)

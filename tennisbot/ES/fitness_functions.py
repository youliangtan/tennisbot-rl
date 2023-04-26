"""
Source repo: https://github.com/enajx/ES
"""

import gym
from gym import wrappers as w
from gym.spaces import Discrete, Box
import pybullet_envs
import numpy as np
import torch
import torch.nn as nn
from typing import List, Any

from tennisbot.ES.policies import MLP, CNN,GatedCNN

HISTORY_LEN = 8

def fitness_static(evolved_parameters: np.array, environment : str, normalizer, render = False) -> float:
    """
    Evaluate an agent 'evolved_parameters' in an environment 'environment' during a lifetime.
    Returns the episodic fitness of the agent.
    """
            
    with torch.no_grad():
                    
        # Load environment
        try:
            env = gym.make(environment)
        except:
            env = gym.make(environment)
        if render:
            env.render()

            # env.render()  # bullet envs
        


        # Check if selected env is pixel or state-vector 
        # if len(env.observation_space.shape) == 3:     # Pixel-based environment
        #     pixel_env = True
        #     env = w.ResizeObservation(env, 84)        # Resize and normilise input
        #     env = ScaledFloatFrame(env)
        #     input_channels = 3
        if len(env.observation_space.shape) == 1:
            pixel_env = False
            input_dim = env.observation_space.shape[0]
        elif len(env.observation_space.shape) == 0:   
            pixel_env = False
            input_dim = env.observation_space.n
        else:
            print("figure out the observation space length")
            
        # Determine action space dimension
        if isinstance(env.action_space, Box):
            action_dim = env.action_space.shape[0]
        elif isinstance(env.action_space, Discrete):
            action_dim = env.action_space.n
        else:
            raise ValueError('Only Box and Discrete action spaces supported')
        
        # Initialise policy network: with CNN layer for pixel envs and simple MLP for state-vector envs

        p = GatedCNN(input_dim , action_dim)

        # Load weights into the policy network
        nn.utils.vector_to_parameters( torch.tensor (evolved_parameters, dtype=torch.float32 ),  p.parameters() )
            
        observation = env.reset()
        # add it to the history of observations and normalize them

        normalizer.observe(observation)
        normed_observation = normalizer.normalize(observation)

        normed_observation = np.reshape(normed_observation, newshape=(1,-1))



        history_normed_obs = np.repeat(normed_observation, repeats=8, axis=0)  # make sure to use a history of 8 time step


        # if pixel_env: observation = np.swapaxes(observation,0,2) #(3, 84, 84)

        # Burnout phase for the bullet quadruped so it starts off from the floor


        #
        # action = np.zeros(8)
        # for _ in range(HISTORY_LEN):
        #     __ = env.step(action)

        # Inner loop
        neg_count = 0
        rew_ep = 0
        t = 0
        while True:
            
            # For obaservation ∈ gym.spaces.Discrete, we one-hot encode the observation
            # if isinstance(env.observation_space, Discrete):
            #     observation = (observation == torch.arange(env.observation_space.n)).float()
            # print("time",t)
            # print("here we are inputting the ob history at", (history_obs[-8:].T).shape, history_obs)


            o3 = p((history_normed_obs[-8:].T))
           # print( 'the returned action is ', o3)

            # Bounding the action space
            # if environment == 'CarRacing-v0':
            #     action = np.array([ torch.tanh(o3[0]), torch.sigmoid(o3[1]), torch.sigmoid(o3[2]) ])
            #     o3 = o3.numpy()
            # elif environment[-12:-6] == 'Bullet':
            #     o3 = np.tanh(o3).numpy()
            #     action = o3
            # else:

            if isinstance(env.action_space, Box):
                action = o3.numpy()
                action = np.clip(action, env.action_space.low, env.action_space.high)
            elif isinstance(env.action_space, Discrete):
                action = np.argmax(o3).numpy()

            
            # Environment simulation step
            # print("the action is now", action)
            observation, reward, done, info = env.step(action)

            normalizer.observe(observation)
            normed_observation = normalizer.normalize(observation)
            normed_observation = np.reshape(normed_observation, newshape=(1, -1))
            history_normed_obs = np.append(history_normed_obs, normed_observation, axis=0)

            # if environment == 'AntBulletEnv-v0': reward = env.unwrapped.rewards[1] # Distance walked
            #
            rew_ep += reward
            
            # env.render('human') # Gym envs
            
            if pixel_env: observation = np.swapaxes(observation,0,2) #(3, 84, 84)
                                       
            # Early stopping conditions
            # if environment == 'CarRacing-v0':
            #     neg_count = neg_count+1 if reward < 0.0 else 0
            #     if (done or neg_count > 20):
            #         break
            # elif environment[-12:-6] == 'Bullet':
            #     if t > 200:
            #         neg_count = neg_count+1 if reward < 0.0 else 0
            #         if (done or neg_count > 30):
            #             break
            # else:
            if done:
                break
            # else:
            #     neg_count = neg_count+1 if reward < 0.0 else 0
            #     if (done or neg_count > 50):
            #         break
            
            t += 1
            
        env.close()

    return rew_ep
    # return max(rew_ep, 0)

import numpy as np
import multiprocessing as mp
import copy
import torch
import sys
import time
from os.path import join, exists
from os import mkdir
import time
import argparse
import sys
import torch
from os.path import join, exists
from os import mkdir
import gym
from gym.spaces import Discrete, Box
from tennisbot.ES.fitness_functions import fitness_static
import itertools

repeat_j =10
class Normalizer():

    def __init__(self, nb_inputs=6):
        self.n = np.zeros(shape=(1,nb_inputs))
        self.mean = np.zeros(shape=(1,nb_inputs))
        self.mean_diff = np.zeros(shape=(1,nb_inputs))
        self.var = np.zeros(shape=(1,nb_inputs))

    def observe(self, x):
        # print("is anything wring", x.shape, self.mean.shape)
        self.n += 1.
        last_mean = self.mean.copy()
        self.mean += (x - self.mean) / self.n
        self.mean_diff += (x - last_mean) * (x - self.mean)
        self.var = (self.mean_diff / self.n).clip(min=1e-2)

    def normalize(self, inputs):
        obs_mean = self.mean
        obs_std = np.sqrt(self.var)
        return (inputs - obs_mean) / obs_std






def compute_ranks(positive_rewards, negative_rewards, nb_best_directions = 100):
    """
    Returns rank as a vector of len(x) with integers from 0 to len(x)
    """
    # assert x.ndim == 1
    scores = {k: (r_pos - r_neg) for k, (r_pos, r_neg) in enumerate(zip(positive_rewards, negative_rewards))}
    order = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:nb_best_directions]


    #
    # diff = np.zeros(shape= int(POPULATION_SIZE))
    # for i in range(int(POPULATION_SIZE) - 1):
    #     diff[i] = x[i]-x[i+1]
    #
    # ranks = np.empty(int(POPULATION_SIZE/2), dtype=int)
    # ranks[diff.argsort()] = np.arange(int(POPULATION_SIZE/2))
    return order

# def compute_centered_ranks(x):
#   """
#   Maps x to [-0.5, 0.5] and returns the rank
#   """
#   y = compute_ranks(x).astype(np.float32)
#   pop_size_pairs = len(x)
#
#   y /= (pop_size_pairs - 1)
#   y -= .5
#   return y

def worker_process(arg):
    get_reward_func, weights, env,normalizer_state = arg
    
    # wp = np.array(weights)
    # decay = -0.01 * np.mean(wp**2)
    # r = get_reward_func(weights, env,normalizer_state) + decay
    r = get_reward_func(weights, env, normalizer_state)
    return r 


class EvolutionStrategyStatic(object):
    def __init__(self, weights, environment, population_size=200, sigma=0.1, learning_rate=0.2, decay=0.995, num_threads=-1, K=70):
        
        self.weights = weights
        self.environment = environment
        self.POPULATION_SIZE = population_size # we sample 2 * population size, since we are doing +,  -
        self.SIGMA = np.float32(sigma)
        self.learning_rate = learning_rate
        self.decay = decay
        self.num_threads = mp.cpu_count() if num_threads == -1 else num_threads
        self.update_factor = self.learning_rate / (self.POPULATION_SIZE * self.SIGMA)
        self.K = K
        
        self.get_reward = fitness_static
        
    def _get_weights_try(self, w, p):
        
        # weights_try = []
        # for index, i in enumerate(p):
        #     jittered = np.float32(self.SIGMA * i)
        #     weights_try.append(w[index] + jittered)
        # weights_try = np.array(weights_try)
        # return weights_try   # weights_try[i] = w[i] + sigma * p[i]
        
        return w + p*self.SIGMA
 
    def get_weights(self):
        return self.weights

    def _get_population(self):
        print("getting pop")
        population = []
        for i in range(self.POPULATION_SIZE):
            x = []
            x2 = []
            for w in self.weights:
                j = np.random.randn(*w.shape)
                x.append(j)
                x2.append(-j) 

            population.append(x)
            population.append(x2)
            
        population = np.array(population).astype(np.float32)

        return population    # [[w_i... w_92000], [w_j... w_92000], [...], ...]


    def _get_rewards(self, pool, population,normalizer_state):

        # population_negative =  -1 * population_positive
        # population = np.append(population_positive,population_negative)

        print("we are using these populations", population.shape)
        
        # Multi-core
        if pool is not None:
            worker_args = []
            for p in population:

                weights_try1 = []

                for index, i in enumerate(p):
                    jittered = self.SIGMA * i
                    weights_try1.append(self.weights[index] + jittered)
                weights_try = np.array(weights_try1).astype(np.float32)

                for i in range(repeat_j):
                    worker_args.append((self.get_reward, weights_try, self.environment, normalizer_state) )


            rewards_repeated  = pool.map(worker_process, worker_args)
            print("here we have reward in total", len(rewards_repeated))
            rewards = np.array(rewards_repeated).reshape(newshape=(repeat_j,-1))
            rewards = np.average(rewards,axis=1)

            
            # worker_args = []
            # jittered = self.SIGMA * population
            # for i in range(len(population)):
            #     worker_args.append( (self.get_reward, self.weights + jittered[i], self.environment) )
            # rewards  = pool.map(worker_process, worker_args)
            
        # Single-core
        else:
            rewards = []
            for p in population:
                weights_try = np.array(self._get_weights_try(self.weights, p))   # weights_try[i] = self.weights[i] + sigma * p[i]
                temp = []
                for i in range(repeat_j):
                    temp.append(self.get_reward(weights_try, self.environment, normalizer=normalizer_state)) # modified already, to reduce variance

                rewards.append(np.average(temp))

        rewards = np.array(rewards).astype(np.float32)

        return rewards

    def _update_weights(self, rewards, population):
        positive_rewards =  rewards[::2]
        negative_rewards = rewards[1::2]

        # positive_pertubation = population[::2]
        # negative_pertubation = population[::2]

        order = compute_ranks(positive_rewards, negative_rewards, self.K)   # in population, only postiive
        # all_rewards = np.array(positive_rewards + negative_rewards)
        # std = all_rewards.std()

        # std = rewards.std()

        # if std == 0:
        #     raise ValueError('Variance should not be zero')

        rollouts = [(positive_rewards[k], negative_rewards[k], population[2*k]) for k in order]

        elite_rewards = np.array(rollouts[0] + rollouts[1])

        std = elite_rewards.std()



        reward_diff = np.array(rollouts[0]) - np.array( rollouts[1])


        # step = np.dot( np.array(rollouts[0] - rollouts[1]) , np.array(rollouts[-1]) )

        elite_population =  [ population[2*k] for k in order]


        # rewards = (rewards - rewards.mean()) / std  # Normalize rewards

        for index, w in enumerate(self.weights):
            layer_population = np.array([p[index] for p in elite_population])   # Array of all weights[i] for all the networks in the population
            
            self.update_factor = self.learning_rate / (std * self.K)
            # K * layer_para_size , K *1
            self.weights[index] = w + self.update_factor * np.matmul( (layer_population.T), np.reshape(reward_diff,(-1,1)))

        if self.learning_rate > 0.001:
            self.learning_rate *= self.decay

        #Decay sigma
        if self.SIGMA>0.01:
            self.SIGMA *= 0.999


    def run(self, iterations, print_step=10, path='weights'):
        
        id_ = str(int(time.time()))
        if not exists(path + '/' + id_):
            mkdir(path + '/' + id_)
            
        print('\n********************\n \nRUN: ' + id_ + '\n\n********************\n')
        
        pool = mp.Pool(self.num_threads) if self.num_threads > 1 else None
        
        generations_rewards = []






        normalizer_state = Normalizer()


        
        for iteration in range(iterations):                     # Algorithm 2. Salimans, 2017: https://arxiv.org/abs/1703.03864
            
            population = self._get_population()                 # List of list of random nets [[w1, w2, .., w122888],[...],[...]] : Step 5
            rewards = self._get_rewards(pool, population,normalizer_state)       # List of corresponding rewards for self.weights + jittered populations : Step 6
            self._update_weights(rewards, population)           # Updates self.weights : Steps 8->12 

            if (iteration + 1) % print_step == 0:
                rew_ = rewards.mean()
                print('iter %4i | reward: %3i |  update_factor: %f  lr: %f | sum_w: %i sum_abs_w: %i' % ( iteration + 1, rew_ , self.update_factor, self.learning_rate, int(np.sum(self.weights)) ,int(np.sum(abs(self.weights))) ), flush=True)
                
                # if rew_ > 100:
                torch.save(self.get_weights(), path + "/"+ id_ + "/" + self.environment + "__rew_" + str(int(rew_)) + "__pop_" + str(self.POPULATION_SIZE) + "__{}.dat".format(iteration))

                generations_rewards.append(rew_)
                np.save(path + "/"+ id_ + '/Fitness_values_' + id_ + '_' + self.environment + '.npy', np.array(generations_rewards))

        if pool is not None:
            pool.close()
            pool.join()
            

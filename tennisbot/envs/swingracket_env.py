
#!/usr/bin/env python3

import gym
import numpy as np
import math
from math import pi as PI
import pybullet as p
import matplotlib.pyplot as plt
import time
import random

from tennisbot.resources.racket import Racket
from tennisbot.resources.objects import Court, Ball, Goal


##############################################################################
# Configurations

DELAY_MODE = False

##############################################################################

class SwingRacketEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, use_gui=True, delay_mode=DELAY_MODE):
        # Define action and observation space in 6DOF force and torque
        self.action_space = gym.spaces.box.Box(
            low=np.array([-2, -2.0, -2.0, -5, -5, -5], dtype=np.float32),
            high=np.array([2, 2.0, 2.0, 5, 5, 5], dtype=np.float32))

        # the observation space is the racket's state + target goal
        self.observation_space = gym.spaces.box.Box(
                low=np.array([-20, -10, -20, -10, -15, -5],
                            dtype=np.float32),
                high=np.array([20, 10, 20, 10, 0, 5],
                            dtype=np.float32)
            )
        self.np_random, _ = gym.utils.seeding.np_random()
        self.step_count = 0

        if use_gui:
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

        # model in the world
        self.court = None
        self.racket = None
        self.ball = None
        self.done = False
        self.previous_ball_goal_dist = None
        self.initial_dist_to_goal = None
        self.court_ball_contact_count = 0
        self.prev_action = None
        self.step_count = 0
        self.delay_mode = delay_mode
        print("init done")
        self.reset()

    def moved_dist_to_goal(self):
        # calc reward based on ball's distance to goal and traveled distance
        dist_ball_to_goal = np.linalg.norm(
            np.array(self.ball.get_observation()[:2]) - np.array(self.goal))
        return self.initial_dist_to_goal - dist_ball_to_goal

    def step(self, action):
        self.racket.apply_target_action(
                [action[0]*200, action[1]*200, action[2]*200+4*9.81],
                [action[3], action[4], action[5]]
            )

        # print(action)
        p.stepSimulation()
        self.step_count += 1

        if self.delay_mode:
            time.sleep(1/240.)

        reward = 0

        ## This is to get the reward based on the diff in travel to goal between steps
        # if self.previous_ball_goal_dist is None:
        #     self.previous_ball_goal_dist = dist_ball_to_goal
        # else:
        #     reward += max(self.previous_ball_goal_dist - dist_ball_to_goal, 0)*10
        #     self.previous_ball_goal_dist = dist_ball_to_goal

        # check if ball is in contact with the racket at first few steps
        if self.step_count < 25:
            contact_ball_racket = p.getContactPoints(self.racket.id, self.ball.id)
            if len(contact_ball_racket) > 0:
                reward += 2
                print("Contact ball racket at step: ", self.step_count)

        ## Fast foward and run the simulation for 800 steps
        if self.step_count > 25:
            while not self.done:
                p.stepSimulation()
                self.step_count += 1

                # check if ball is in contact with court
                contact_ball_court = p.getContactPoints(self.court.id, self.ball.id)
                if len(contact_ball_court) > 0:
                    self.done = True
                    reward += self.moved_dist_to_goal()
                    print("Contact ball ground at step [", self.step_count,
                        "] with moved distance: ", self.moved_dist_to_goal())

                # check if ball is in contact with goal_obj
                contact_ball_goal = p.getContactPoints(self.goal_obj.id, self.ball.id)
                if len(contact_ball_goal) > 0:
                    reward += self.moved_dist_to_goal()
                    reward += 50
                    self.done = True
                    print(f"!!! Contact GOAL at step: {self.step_count}")

                # exceed 800 steps
                if self.step_count > 800:
                    self.done = True
                    print("Timeout, exceed 800 steps")

                if self.delay_mode:
                    time.sleep(1/240.)
                    
                ### This is a hack to move racket to the original position
                curr_x, curr_y, curr_z = self.racket.get_observation()
                self.racket.apply_target_action(
                    [
                      -50*(curr_x - self.spawn_pos[0]),
                      -2*(curr_y - self.spawn_pos[1]),
                      -2*(curr_z - self.spawn_pos[2] - 4)
                    ])

        obs = self.racket.get_observation()[:2] + \
            self.ball.get_observation()[:2] +  self.goal
        return obs, reward, self.done, dict()

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self):
        # print("Reset environment!")
        p.resetSimulation(self.client)
        p.setGravity(0, 0, -9.81)
        self.step_count = 0

        # Reload the tennis court and racket
        self.court = Court(self.client)

        # Randomly set the location of the racket and ball
        _rand_x = random.uniform(5.5, 11)
        _rand_y = random.uniform(-4, 4)
        _rand_z = 0.6
        self.racket = Racket(self.client,
                             #  [9.5, 0, 0.2],
                             [_rand_x, _rand_y, _rand_z],
                             rpy=[0, 0.5, 0])
        self.spawn_pos = [_rand_x, _rand_y, _rand_z]
        self.ball = Ball(self.client,
                         pos=[_rand_x-0.1, _rand_y, _rand_z+0.8])

        # Set the goal to a random target
        self.goal = (np.random.uniform(-3, -12), np.random.uniform(-5, 5))
        self.initial_dist_to_goal = np.linalg.norm(
            np.array(self.ball.get_observation()[:2]) - np.array(self.goal))
        self.done = False

        # Visual element of the goal
        self.goal_obj = Goal(self.client, self.goal)

        self.done = False
        self.step_count = 0

        obs = self.racket.get_observation()[:2] + \
            self.ball.get_observation()[:2] +  self.goal
        return obs

    def close(self):
        p.disconnect(self.client)

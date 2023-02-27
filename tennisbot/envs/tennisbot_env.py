#!/usr/bin/env python3

import gymnasium as gym
import numpy as np
import math
import pybullet as p
import matplotlib.pyplot as plt

from tennisbot.resources.racket import Racket
from tennisbot.resources.objects import Court, Ball


##############################################################################
# Configurations

GUI_MODE = True

##############################################################################

class TennisbotEnv(gym.Env):
    """
    Setup Gym environment for tennisbot
    refer to api: 
        https://gymnasium.farama.org/api/env/
    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.action_space = gym.spaces.box.Box(
            low=np.array([-10.0, -10.0, -10.0, -1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([10.0, 10.0, 10.0, 1.0, 1.0, 1.0], dtype=np.float32))
        self.observation_space = gym.spaces.box.Box(
            low=np.array([-20, -20, -5, -4, -4, -4], dtype=np.float32),
            high=np.array([20, 20, 5, -4, -4, -4], dtype=np.float32))
        self.np_random, _ = gym.utils.seeding.np_random()

        if GUI_MODE:
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)

        # Reduce length of episodes for RL algorithms
        p.setTimeStep(1/30, self.client)

        # model in the world
        self.racket = None
        self.done = False
        self.prev_dist_to_goal = None
        self.rendered_img = None
        self.render_rot_matrix = None
        self.reset()

    def step(self, action):
        # Feed action to the racket and get observation of racket's state
        self.racket.apply_action(action)
        p.stepSimulation()
        racket_ob = self.racket.get_observation()
        """
        # Compute reward as L2 change in distance to goal
        dist_to_goal = math.sqrt(((car_ob[0] - self.goal[0]) ** 2 +
                                  (car_ob[1] - self.goal[1]) ** 2))
        reward = max(self.prev_dist_to_goal - dist_to_goal, 0)
        self.prev_dist_to_goal = dist_to_goal

        # Done by running off boundaries
        if (car_ob[0] >= 10 or car_ob[0] <= -10 or
                car_ob[1] >= 10 or car_ob[1] <= -10):
            self.done = True
        # Done by reaching goal
        elif dist_to_goal < 1:
            self.done = True
            reward = 50
        """
        reward = 10

        # TODO: Get observation of the racket and ball state
        # TODO: define reward when the ball is hit by the racket
        ob = np.array(racket_ob, dtype=np.float32)
        return ob, reward, self.done, dict()

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self):
        p.resetSimulation(self.client)
        p.setGravity(0, 0, -10)

        # Reload the tennis court and racket
        Court(self.client)
        self.racket = Racket(self.client)

        # Set the goal to a random target
        x = (np.random.uniform(5, 9) if np.random.randint(2) else
             np.random.uniform(-5, -9))
        y = (np.random.uniform(5, 9) if np.random.randint(2) else
             np.random.uniform(-5, -9))

        self.done = False

        # Get observation to return
        racket_ob = self.racket.get_observation()
        return np.array(racket_ob, dtype=np.float32)

    def render(self, mode='human'):
        if self.rendered_img is None:
            self.rendered_img = plt.imshow(np.zeros((100, 100, 4)))

        # Base information
        racket_id, client_id = self.racket.get_ids()
        proj_matrix = p.computeProjectionMatrixFOV(fov=80, aspect=1,
                                                   nearVal=0.01, farVal=100)
        pos, ori = [list(l) for l in
                    p.getBasePositionAndOrientation(racket_id, client_id)]
        pos[2] = 0.2

        # Rotate camera direction
        rot_mat = np.array(p.getMatrixFromQuaternion(ori)).reshape(3, 3)
        camera_vec = np.matmul(rot_mat, [1, 0, 0])
        up_vec = np.matmul(rot_mat, np.array([0, 0, 1]))
        view_matrix = p.computeViewMatrix(pos, pos + camera_vec, up_vec)

        # This will get the camera view img, not avail now
        """
        frame = p.getCameraImage(100, 100, view_matrix, proj_matrix)[2]
        frame = np.reshape(frame, (100, 100, 4))
        self.rendered_img.set_data(frame)
        plt.draw()
        plt.pause(.00001)
        """

    def close(self):
        p.disconnect(self.client)

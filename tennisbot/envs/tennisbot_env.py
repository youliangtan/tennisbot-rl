#!/usr/bin/env python3

import gym
import numpy as np
import math
from math import pi as PI
import pybullet as p
import matplotlib.pyplot as plt
import time

from tennisbot.resources.racket import Racket
from tennisbot.resources.objects import Court, Ball


##############################################################################
# Configurations

GUI_MODE = True
DELAY_MODE = True
BALL_SHOOT_FRAMES = 450
BALL_FORCE = 5
ENABLE_ORIENTATION = False

##############################################################################

class TennisbotEnv(gym.Env):
    """
    Setup Gym environment for tennisbot
    refer to api: 
        https://gymnasium.farama.org/api/env/
    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        # Define action and observation space
        self.action_space = gym.spaces.box.Box(
            ## NOTE: This is the simplified action space in 2DOF x, y axis
            low=np.array([5.0, -5.0], dtype=np.float32),
            high=np.array([20.0, 5.0], dtype=np.float32))
        
            ## NOTE: This is the original action space in 6DOF
            # low=np.array([5.0, -5.0, 0.0, -PI, -PI, -PI], dtype=np.float32),
            # high=np.array([20.0, 5.0, 5.0, PI, PI, PI], dtype=np.float32))
        self.observation_space = gym.spaces.box.Box(
            low=np.array([-20, -20, -5, -PI, -PI, -PI, -4, -4, -4],
                         dtype=np.float32),
            high=np.array([20, 20, 5, PI, PI, PI, -4, -4, -4],
                          dtype=np.float32))
        self.np_random, _ = gym.utils.seeding.np_random()
        self.step_count = 0

        if GUI_MODE:
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)

        # Reduce length of episodes for RL algorithms
        # p.setTimeStep(1/30, self.client) # 30 fps TODO

        # model in the world
        self.court = None
        self.racket = None
        self.ball = None
        self.done = False
        self.prev_ball_racket_yz_dist = None
        self.rendered_img = None
        self.render_rot_matrix = None
        self.reset()
        self.court_ball_contact_count = 0
        self.prev_action = None

    def step(self, action):
        # Feed action to the racket and get observation of racket's state
        action = np.append(action, [1.5, 0, 0, 0])
        self.racket.apply_action(action)
        # print("action: ", action)

        # Shoot the ball
        if self.step_count < BALL_SHOOT_FRAMES:
            self.ball.apply_force([BALL_FORCE, 0, BALL_FORCE*2.2])

        p.stepSimulation()
        self.step_count += 1
        if DELAY_MODE:
            time.sleep(1./24000.)

        # set reward depends on y-z distance of racket and ball
        # Compute reward as L2 change in distance
        ball_pose = self.ball.get_observation()
        racket_pose = self.racket.get_observation()
        yz_dist = math.sqrt(((ball_pose[2] - racket_pose[2]) ** 2 +
                            (ball_pose[1] - racket_pose[1]) ** 2))
        reward = max(self.prev_ball_racket_yz_dist - yz_dist, 0)
        self.prev_ball_racket_yz_dist = yz_dist
        # print("Reward: ", yz_dist)

        # get collision info
        contacts = p.getContactPoints(self.racket.id, self.ball.id)
        if len(contacts) > 0:
            print(" BINGO!!!! Ball hits the racket!")
            reward = 200
            self.done = True

        # end the episode if the ball hits the court
        contacts = p.getContactPoints(self.court.id, self.ball.id)
        if len(contacts) > 0:
            # print("Ball hits the court!")
            self.court_ball_contact_count += 1
            if self.court_ball_contact_count >= 2: # arbitrary number
                self.done = True

        # this is to prevent the agent making big moves
        if self.prev_action is not None:
            reward -= np.linalg.norm(action - self.prev_action)/1000

        # Get observation of the racket and ball state
        ob = np.array(racket_pose + ball_pose, dtype=np.float32)
        return ob, reward, self.done, dict()

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self):
        print("Reset environment!")
        p.resetSimulation(self.client)
        p.setGravity(0, 0, -10)

        # Reload the tennis court and racket
        self.court = Court(self.client)

        # TODO: Randomly set the location of the racket and ball
        self.racket = Racket(self.client, [6.5, 0.5, 0.5], ENABLE_ORIENTATION)
        self.ball = Ball(self.client, pos=[-9,0,1])

        self.done = False
        self.step_count = 0
        self.court_ball_contact_count = 0

        # Get observation to return
        ball_pose = self.racket.get_observation()
        racket_pose = self.ball.get_observation()

        self.prev_ball_racket_yz_dist = \
            math.sqrt(((ball_pose[2] - racket_pose[2]) ** 2 +
                       (ball_pose[1] - racket_pose[1]) ** 2))
        return np.array(racket_pose + ball_pose, dtype=np.float32)

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

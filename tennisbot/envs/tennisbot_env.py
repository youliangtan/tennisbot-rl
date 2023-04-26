
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
from tennisbot.resources.objects import Court, Ball


##############################################################################
# Configurations

DELAY_MODE = False
BALL_SHOOT_FRAMES = 5
BALL_FORCE = 25
ENABLE_ORIENTATION = False

##############################################################################

class TennisbotEnv(gym.Env):
    """
    Setup Gym environment for tennisbot
    refer to api: 
        https://gymnasium.farama.org/api/env/
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, use_gui=False, is_sparse_reward=False):
        # Define action and observation space
        self.action_space = gym.spaces.box.Box(
            ## NOTE: This is the simplified action space in 3DOF x, y, z axis
            # low=np.array([-4.0, -4.0, -2.0], dtype=np.float32),
            # high=np.array([4.0, 4.0, 2.0], dtype=np.float32))
            
            # action is normalized
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32))

            ## NOTE: This is the original action space in 6DOF
            # low=np.array([5.0, -5.0, 0.0, -PI, -PI, -PI], dtype=np.float32),
            # high=np.array([20.0, 5.0, 5.0, PI, PI, PI], dtype=np.float32))
        
        # the observation space is the racket pos and vel + ball pos and vel
        self.observation_space = gym.spaces.box.Box(
            low=np.array([-20, -20, -5, -5, -5, -5] + [-20, -20, 0, -10, -10, -10],
                         dtype=np.float32),
            high=np.array([20, 20, 5, 5, 5, 5]  + [20, 20, 10, 10, 10, 10],
                          dtype=np.float32)
            # low=np.array([-20, -20, -5, -PI, -PI, -PI, -4, -4, -4],
            #              dtype=np.float32),
            # high=np.array([20, 20, 5, PI, PI, PI, 4, 4, 4],
            #               dtype=np.float32)
        )
        self.np_random, _ = gym.utils.seeding.np_random()
        self.step_count = 0

        if use_gui:
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)

        # Reduce length of episodes for RL algorithms
        # p.setTimeStep(1/30, self.client) # 30 fps TODO

        # model in the world
        self.is_sparse_reward = is_sparse_reward
        self.court = None
        self.racket = None
        self.ball = None
        self.done = False
        self.prev_ball_racket_yz_dist = 9
        self.rendered_img = None
        self.render_rot_matrix = None
        self.court_ball_contact_count = 0
        self.prev_action = None
        self.step_count = 0
        self.racket_scale = 1.0
        print("init done")

        self.reset()

    def step(self, action):
        # Feed action to the racket and get observation of racket's state
        # action = np.append(action, [1.5, 0, 0, 0])
        # self.racket.apply_action(action)
        
        # print("action: ", action)
       
        # action[2] += 4 # add a constant z-axis force to make it float
        action = np.append(action, 4*9.81)
        action[0] *= 10  # action is normalized
        action[1] *= 10  # action is normalized
        self.racket.apply_target_action(action)

        # shoot the ball out
        if self.step_count < BALL_SHOOT_FRAMES:
            self.ball.apply_force(self.ball_shoot_force)

        p.stepSimulation()
        self.step_count += 1

        if DELAY_MODE:
            # time.sleep(1./24000.)
            time.sleep(1/240.)

        ball_pose = self.ball.get_observation()
        racket_pose = self.racket.get_observation()
        
        reward = 0

        # Get observation of the racket and ball state
        ob = np.array(
            racket_pose[:3] + self.racket.get_vel() + ball_pose + self.ball.get_vel(),
            dtype=np.float32)

        if self.step_count < BALL_SHOOT_FRAMES:
            return ob, reward, False, dict()

        # # end the episode if the ball hits the court
        # contacts_ball_ground = p.getContactPoints(self.court.id, self.ball.id)
        # if len(contacts_ball_ground) > 0:
        #     # print("Ball hits the court!")
        #     self.court_ball_contact_count += 1

        #     if self.court_ball_contact_count >= 2: # arbitrary number
        #         ball_pose_ground = self.ball.get_observation()
        #         print("second hit, x", ball_pose_ground[0])
        #         if (ball_pose_ground[0] < 0.0): # hit the opposite court
        #             print("Hit the ball to the opposite side of the court!")
        #             reward += 200
        #         self.done = True

        # # penalize the racket if it hits the court
        # contact_racket_ground = p.getContactPoints(self.court.id, self.racket.id)
        # if len(contact_racket_ground) > 0:
        #     # print("Racket hits the court!")
        #     reward -= 2

        # # penalize the racket if it is getting to high
        # if racket_pose[2] > 2.5:
        #     reward -= 1

        # # get collision info
        contacts_ball_racket = p.getContactPoints(self.racket.id, self.ball.id)
        if len(contacts_ball_racket) > 0:
            print("   BINGO!!!! Ball hits the racket!")
            reward += 20
            # self.done = True

        # # this is to prevent the agent making big moves
        # # if self.prev_action is not None:
        # #     reward -= np.linalg.norm(action - self.prev_action)/1000

        # reward the racket to follow the ball
        x_ball_to_racket = ball_pose[0] - racket_pose[0]
        if (x_ball_to_racket < 0.5):
            dist = (racket_pose[1] - ball_pose[1])
            # print(self.prev_ball_racket_yz_dist - dist)
            # reward += max(self.prev_ball_racket_yz_dist - dist, 0)
            reward += self.prev_ball_racket_yz_dist - dist
            self.prev_ball_racket_yz_dist = dist
            pass
        else:
            delta_dist = math.sqrt(((ball_pose[2] - racket_pose[2]) ** 2 +
                                (ball_pose[1] - racket_pose[1]) ** 2))
            print(f"Ball passed the racket step [{self.step_count}]"
                  f"with y-z distance: {delta_dist}")
            self.done = True

        # # end the episode after 1000 steps
        if self.step_count > 1000:
            print("Episode ends after 1000 steps!, racket pose: ", racket_pose)
            self.done = True

        # print("reward", reward)
        # reward = 1
        return ob, reward, self.done, dict()

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def set_racket_scale(self, scale):
        self.racket_scale = scale
        print("Set racket scale to: ", scale)

    def reset(self):
        # print("Reset environment!")
        p.resetSimulation(self.client)
        p.setGravity(0, 0, -9.81)
        self.step_count = 0

        # Reload the tennis court and racket
        self.court = Court(self.client)

        # TODO: Randomly set the location of the racket and ball
        _rand_x = random.uniform(7.5, 12.5)
        _rand_y = random.uniform(-5, 5)
        _rand_z = random.uniform(0.2, 0.21)
        self.racket = Racket(self.client,
                            #  [9.5, 0, 0.2],
                            [_rand_x, _rand_y, _rand_z],
                             ENABLE_ORIENTATION,
                             scale=self.racket_scale)

        # ball shooting force during init
        self.ball_shoot_force = [
            random.uniform(BALL_FORCE, BALL_FORCE*1.5),
            random.uniform(-BALL_FORCE*0.4, BALL_FORCE*0.4),
            BALL_FORCE*0.8
        ]

        self.ball = Ball(self.client, pos=[-9,0,1])
        x, y, z = self.ball.get_observation()
        self.ball.random_pos(
            range_x=[x-3, x+3], range_y=[y-1, y+1], range_z=[z, z+0.5])

        self.done = False
        self.step_count = 0
        self.court_ball_contact_count = 0

        # Get observation to return
        ball_pose = self.ball.get_observation()
        racket_pose = self.racket.get_observation()

        self.prev_ball_racket_yz_dist = 0
       
        racket_vel = self.racket.get_vel()
        return np.array(
            racket_pose[0:3] + racket_vel + ball_pose + self.ball.get_vel(),
            dtype=np.float32)

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

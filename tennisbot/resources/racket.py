#!/usr/bin/env python3

import pybullet as p
import os
import math
from typing import Tuple, List
from simple_pid import PID
import random

##############################################################################
def set_angle(angle: float) -> float:
    """
    Ensure the angle is within the range of [-pi, pi] radian convention
    """
    return math.atan2(math.sin(angle), math.cos(angle))

##############################################################################
class Racket:
    def __init__(self, 
                 client,
                 pos=[0, 0, 0], 
                 enable_orientation=True,
                 rpy = [0, 0, 0],
                 time_step=1/240,
                 scale = 1.0):
        """
        init the racket object with default position, orientation, 
        and time step for PID
        """
        self.enable_orientation = enable_orientation
        self.client = client
        f_name = os.path.join(os.path.dirname(__file__), 'racket.urdf')
        
        # rpy to quaternion
        quat = p.getQuaternionFromEuler(rpy)
        self.id = p.loadURDF(fileName=f_name,
                              basePosition=pos,
                              baseOrientation=quat,
                              physicsClientId=client,
                              globalScaling=scale)

        # Make bouncey racket
        p.changeDynamics(self.id, -1, restitution=0.9)
        p.changeDynamics(self.id, -1, lateralFriction=0.2)
        p.changeDynamics(self.id, -1, rollingFriction=0.001)

        # Define PID controller gains
        # NOTE: this is tested in playground.py
        kp = 3.0
        kd = 0.1
        ki = 0.01
        maxForce = 10.0
        maxTorque = 3.0
        self.time_step = time_step

        self.pos_controller = [
            PID(kp, ki, kd, 
                output_limits=(-maxForce, maxForce), sample_time=1/240) for i in range(3)
        ]

        self.ori_controller = [
            PID(kp, ki, kd,
                output_limits=(-maxTorque, maxTorque), sample_time=1/240) for i in range(3)
        ]


    def get_ids(self) -> Tuple:
        return self.id, self.client

    def apply_action(self, action):
        """
        Applying position control to the racket
        """
        self.set_target_location(action)
        self.apply_pid_force_torque()

    def set_target_location(self, pose):
        """
        This is to set the target location of where we want the racket to be
        """
        for i in range(3):
            self.pos_controller[i].setpoint = pose[i]
            # self.ori_controller[i].setpoint = pose[i+3]

        # This is some hack to disable orientation control
        # TODO: not really working, fix this
        if not self.enable_orientation:
            for i in range(3):
                self.ori_controller[i].setpoint = 0
                
    
    def apply_target_action(self, forces, torques=None):
        """
        Apply directly target force
        """
        pose = self.get_observation()
        p.applyExternalForce(
                self.id, -1, forces[:3], pose[:3], p.WORLD_FRAME)
        if torques is not None:
            p.applyExternalTorque(self.id, -1, torques[:3], p.WORLD_FRAME)


    def apply_pid_force_torque(self):
        """
        Applying force and torque control to the racket
        """       
        # for i_step in range(2):
        pose = self.get_observation()

        # Control the racket position, default z-force to compensate gravity
        apply_force = [0, 0, 4]
        # apply_torque = [0, 0, 0]
        for i in range(3):
            apply_force[i] += self.pos_controller[i](pose[i], self.time_step)
            # apply_torque[i] = self.ori_controller[i](pose[i+3], self.time_step)
            
        # print("apply_force ", apply_force)
        
        p.applyExternalForce(
            self.id, -1, apply_force, pose[:3], p.WORLD_FRAME)
        
        # p.applyExternalTorque(self.id, -1, apply_torque, p.WORLD_FRAME)
    
    def get_observation(self) -> List[float]:
        """
        Get the position and orientation of the racket in the simulation
        return observation
        # Concatenate position, orientation in size 6 vector pose
        # [x, y, z, roll, pitch, yaw]
        """
        pos, ang = p.getBasePositionAndOrientation(self.id, self.client)
        ori = p.getEulerFromQuaternion(ang)

        # return pos + ori
        return pos

    def get_vel(self) -> List[float]:
        """
        Get the velocity of the racket in the simulation
        return velocity
        """
        vel, ang_vel = p.getBaseVelocity(self.id, self.client)
        return vel
    
    def reset_pos(self, pos):
        """reset position for the racket
        Args:
            pos (list): the position [x,y,z]
        """
        p.resetBasePositionAndOrientation(self.id, pos, [0, 0, 0, 1], self.client)


    def random_pos(self, range_x, range_y, range_z):
        """set a random position of the racket within given ranges.

        Args:
            range_x (list): rnage of x randomness
            range_y (list): range of y randomness
            range_z (list): range of z randomness
        """
        # random position 
        rand_x = random.uniform(range_x[0], range_x[1])
        rand_y = random.uniform(range_y[0], range_y[1])
        rand_z = random.uniform(range_z[0], range_z[1])
        # random force
        randomlist = [rand_x, rand_y, rand_z]
        self.reset_pos(randomlist)

        
    def update_pid(self, kp, ki, kd, maxForce = 10.0, maxTorque = 3.0):
        """
        Update the pid parameters
        """
        self.pos_controller = [
            PID(kp, ki, kd,
                output_limits=(-maxForce, maxForce),
                sample_time=self.time_step) for i in range(3)
        ]

        self.ori_controller = [
            PID(kp, ki, kd,
                output_limits=(-maxTorque, maxTorque),
                sample_time=self.time_step) for i in range(3)
        ]

#!/usr/bin/env python3

import pybullet as p
from typing import Tuple, List
import os

##############################################################################
class Plane:
    def __init__(self, client, pos=[0, 0, 0]):
        f_name = os.path.join(os.path.dirname(__file__), 'simpleplane.urdf')
        p.loadURDF(
            fileName=f_name, basePosition=pos, physicsClientId=client)

##############################################################################

class Court:
    def __init__(self, client, pos=[0, 0, 0]):
        f_name = os.path.join(os.path.dirname(__file__), 'court.urdf')
        self.id = p.loadURDF(
            fileName=f_name, basePosition=pos, physicsClientId=client)

        # Make bouncey court
        p.changeDynamics(self.id, -1, restitution=0.9)
        p.changeDynamics(self.id, -1, lateralFriction=0.2)
        p.changeDynamics(self.id, -1, rollingFriction=0.001)
        
        # add court texture
        f_name = os.path.join(os.path.dirname(__file__), 'court_mod.png')
        texture = p.loadTexture(f_name)
        p.changeVisualShape(self.id, -1, textureUniqueId=texture)

##############################################################################
class Ball:
    def __init__(self, client, pos=[0, 0, 0]):
        f_name = os.path.join(os.path.dirname(__file__), 'ball.urdf')
        self.id = p.loadURDF(  
            fileName=f_name, basePosition=pos, physicsClientId=client)

        # Make bouncey ball
        p.changeDynamics(self.id, -1, restitution=0.9)
        p.changeDynamics(self.id, -1, lateralFriction=0.2)
        p.changeDynamics(self.id, -1, rollingFriction=0.001)

    def get_observation(self) -> List[float]:
        """
        Get the position and orientation of the racket in the simulation
        return observation
        # Concatenate position, orientation in size 6 vector 
        # [x, y, z, roll, pitch, yaw]
        """
        pos, ang = p.getBasePositionAndOrientation(self.id, self.client)
        ori = p.getEulerFromQuaternion(ang)
        for i in range(3):
            self.pos_controller[i].setpoint = pos[i]
            self.ori_controller[i].setpoint = ori[i]
        return pos + ori

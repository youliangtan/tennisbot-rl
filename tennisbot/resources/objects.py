#!/usr/bin/env python3

import pybullet as p
from typing import Tuple, List
import os
import random

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
        self.born_pos = pos
        self.id = p.loadURDF(  
            fileName=f_name, basePosition=pos, physicsClientId=client)
        self.client = client

        # Make bouncey ball
        p.changeDynamics(self.id, -1, restitution=0.9)
        p.changeDynamics(self.id, -1, lateralFriction=0.2)
        p.changeDynamics(self.id, -1, rollingFriction=0.001)

    def get_observation(self) -> List[float]:
        """
        Get the position of the ball in the simulation
        @return ball pos in [x, y, z]
        """
        pos, _ = p.getBasePositionAndOrientation(self.id, self.client)
        return pos

    def apply_force(self, force: Tuple[float, float, float]):
        """
        Apply force to the ball in the simulation
        """
        pos, ang = p.getBasePositionAndOrientation(self.id, self.client)
        p.applyExternalForce(self.id, -1, force, pos, p.WORLD_FRAME)
        
    def set_pos(self, pos):
        """
        Set a new pos to the ball.
        Args:
            pos (list): ball position
        """
        f_name = os.path.join(os.path.dirname(__file__), 'ball.urdf')
        self.born_pos = pos
        self.id = p.loadURDF(  
            fileName=f_name, basePosition=pos, physicsClientId=self.client)

        # Make bouncey ball
        p.changeDynamics(self.id, -1, restitution=0.9)
        p.changeDynamics(self.id, -1, lateralFriction=0.2)
        p.changeDynamics(self.id, -1, rollingFriction=0.001)
        
    def random_pos(self, range_x, range_y, range_z):
        """set a random position to the ball within given ranges.

        Args:
            range_x (list): rnage of x randomness
            range_y (list): range of y randomness
            range_z (list): range of z randomness
        """
        # random position 
        rand_x = random.randint(range_x[0], range_x[1])
        rand_y = random.randint(range_y[0], range_y[1])
        rand_z = random.randint(range_z[0], range_z[1])

        randomlist = [rand_x, rand_y, rand_z]
        self.set_pos(randomlist)
        
    def get_pos(self):
        """get the position the ball was born at.
        Returns:
            list: the position [x,y,z]
        """
        return self.born_pos
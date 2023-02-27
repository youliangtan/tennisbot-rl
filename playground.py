#!/usr/bin/env python3

import pybullet as p
import time
import pybullet_data
from simple_pid import PID

from tennisbot.resources.objects import Ball, Court
from tennisbot.resources.racket import Racket


##############################################################################
# Configurations

BALL_START_SHOOT_FRAME = 300
BALL_END_SHOOT_FRAME = 750
BALL_FORCE = 5

##############################################################################

def main():
    pybullet_client = p.connect(p.GUI) # or p.DIRECT for non-graphical version

    p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
    p.resetDebugVisualizerCamera( 
        cameraDistance=10, cameraYaw=0, cameraPitch=-30, cameraTargetPosition=[3,0,2])
    p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
    p.setGravity(0,0,-10)

    # Load model in world
    planeId = p.loadURDF("plane.urdf")

    court_obj = Court(pybullet_client)
    court = court_obj.id

    ball_obj = Ball(pybullet_client, pos=[-9,0,1])
    ball = ball_obj.id

    # TODO: fix the wam7 robot with weird joint physics dynamics
    # tennisbot = p.loadURDF("robots/wam7.urdf", basePosition=[2, 0 , 1])

    racket_obj = Racket(pybullet_client, pos=[3,0.1,0.8])
    racket = racket_obj.id

    targetPos = [6.5, -0.2, 5]
    targetOri = [-1.57, 0, 0]
    racket_obj.set_target_location(targetPos + targetOri)

    ##################################################################################
    # Run Simulation

    for i in range (10000):
        # shoot ball
        if BALL_START_SHOOT_FRAME < i < BALL_END_SHOOT_FRAME:
            pos, ori = p.getBasePositionAndOrientation(ball)
            p.applyExternalForce(ball, -1, [BALL_FORCE, 0, BALL_FORCE*2.2], pos, p.WORLD_FRAME)
        
        racket_obj.apply_pid_force_torque()

        p.stepSimulation()
        time.sleep(1./240.)

    p.disconnect()

if __name__ == '__main__':
    print("start playground")
    main()

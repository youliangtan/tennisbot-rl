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
BALL_SHOOT_FRAMES = 450
BALL_FORCE = 5.5

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

    court = Court(pybullet_client)
    ball = Ball(pybullet_client, pos=[-9,0,1])

    # TODO: fix the wam7 robot with weird joint physics dynamics
    # tennisbot = p.loadURDF("robots/wam7.urdf", basePosition=[2, 0 , 1])

    racket = Racket(pybullet_client, pos=[3,0.1,0.8])

    targetPos = [12, 0.1, 1]
    targetOri = [-1.57, 0, 0]
    racket.set_target_location(targetPos + targetOri)
    
    kp = 10.0
    kd = 2.0
    ki = 0.001
    
    racket.update_pid(kp, ki, kd)

    ##################################################################################
    # Run Simulation
    for i_sim in range(3):
        # remove the last ball and racket
        p.removeBody(bodyUniqueId=ball.id)
        p.removeBody(bodyUniqueId=racket.id)
        
        # random pos of the ball
        ball.random_pos([-12,-3],[-4,4],[0,3])      
        
        # random pos of the racket
        racket.random_pos([5,12],[-3,3],[0,2])
        
        # set target pos related to the ball born pos
        ball_pos = ball.get_pos()
        
        targetPos = [min(13, ball_pos[0]+20), ball_pos[1], 0.5]
        print("--- targetPos ---")
        print(targetPos)
        racket.set_target_location(targetPos + targetOri)
        
        print(" ============ ith simulation ============ ")
        for i in range (3000):
            
            # shoot ball
            if BALL_START_SHOOT_FRAME < i < BALL_SHOOT_FRAMES+BALL_START_SHOOT_FRAME:
                ball.apply_force([BALL_FORCE, 0, BALL_FORCE*2.2])
            
            racket.apply_pid_force_torque()

            # get collision info
            contacts = p.getContactPoints(racket.id, ball.id)
            if len(contacts) > 0:
                print("Racket and ball are in collision!!")

            # contacts = p.getContactPoints(court.id, ball.id)
            # if len(contacts) > 0:
                # print("Court and ball are in collision!!")

            p.stepSimulation()
            time.sleep(1./240.)
    p.disconnect()

if __name__ == '__main__':
    print("start playground")
    main()

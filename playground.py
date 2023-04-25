#!/usr/bin/env python3

import pybullet as p
import time
import pybullet_data
from simple_pid import PID
import argparse

from tennisbot.resources.objects import Ball, Court, Goal
from tennisbot.resources.racket import Racket


##############################################################################

# Configurations
BALL_START_SHOOT_FRAME = 300
BALL_SHOOT_FRAMES = 450
BALL_FORCE = 5.5
DELAY_TIME = 1/240

# Config for swing
SWING_FRAME_COUNT = 20


##############################################################################

def main(args):
    pybullet_client = p.connect(p.GUI) # or p.DIRECT for non-graphical version

    p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
    p.resetDebugVisualizerCamera( 
        cameraDistance=5, cameraYaw=0, cameraPitch=-50, cameraTargetPosition=[3,0,2])
    p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
    p.setGravity(0,0,-9.81)
    court = Court(pybullet_client)

    ##################################################################################
    if args.swing:
        # Load model in world
        position = [3, 0.1, 0.5]
        ball = Ball(pybullet_client, pos=[position[0]-0.1, position[1], position[2]+0.7])
        racket = Racket(pybullet_client, pos=position, rpy=[0, 0.5, 0])

        goal = Goal(pybullet_client, base=[-12, 0])

        # this is just for debugging
        time.sleep(3)

        for i in range (3000):
            if i < SWING_FRAME_COUNT:
                # define the action space as [x, y, z, roll, pitch, yaw]
                action_space = [-400, 50, 400, 0, 0.3, -0.2]
                racket.apply_target_action(
                        [action_space[0], action_space[1], action_space[2]+4*9.81],
                        [action_space[3], action_space[4], action_space[5]]
                    )
            else:
                racket.apply_target_action([0, 0, 4*9.81])
            p.stepSimulation()
            time.sleep(DELAY_TIME)

    ##################################################################################
    else:
        # Load model in world
        ball = Ball(pybullet_client, pos=[-9,0,1])

        # TODO: fix the wam7 robot with weird joint physics dynamics
        # tennisbot = p.loadURDF("robots/wam7.urdf", basePosition=[2, 0 , 1])

        racket = Racket(pybullet_client, pos=[3,0.1,0.8], time_step=DELAY_TIME)

        targetPos = [12, 0.1, 1]
        targetOri = [-1.57, 0, 0]
        racket.set_target_location(targetPos + targetOri)
        
        kp = 10.0
        kd = 2.0
        ki = 0.001

        racket.update_pid(kp, ki, kd)
        print("target location: ", racket.get_observation())

        # Run Simulation
        for i_sim in range(3):
            # random pos of the ball
            ball.random_pos([-12,-3],[-4,4],[0,3])

            # random pos of the racket
            racket.random_pos([5,12],[-3,3],[1, 1.1])

            # set target pos
            racket.set_target_location(targetPos + targetOri)
            print("target location: ", racket.get_observation())

            print(" ============ ith simulation ============ ")
            for i in range (3000):
                
                # shoot ball
                if BALL_START_SHOOT_FRAME < i < BALL_SHOOT_FRAMES+BALL_START_SHOOT_FRAME:
                    ball.apply_force([BALL_FORCE, 0, BALL_FORCE*2])
                
                racket.apply_pid_force_torque()

                # get collision info
                contacts = p.getContactPoints(racket.id, ball.id)
                if len(contacts) > 0:
                    print("Racket and ball are in collision!!")

                # contacts = p.getContactPoints(court.id, ball.id)
                # if len(contacts) > 0:
                    # print("Court and ball are in collision!!")

                p.stepSimulation()
                time.sleep(DELAY_TIME)

    p.disconnect()

##############################################################################

if __name__ == '__main__':
    print("start playground")
    parser = argparse.ArgumentParser()
    parser.add_argument('--swing',
                        action='store_true', help="select swing racket scene")
    args = parser.parse_args()
    main(args)

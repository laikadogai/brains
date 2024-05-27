"""
| Joint          | Min    | Max   | Servo ID(s) |
|----------------|--------|-------|-------------|
| Waist          | -180   | 180   | 1           |
| Shoulder       | -108   | 114   | 2+3         |
| Elbow          | -123   | 92    | 4+5         |
| Forearm Roll   | -180   | 180   | 6           |
| Wrist Angle    | -100   | 123   | 7           |
| Wrist Rotate   | -180   | 180   | 8           |
| Gripper        | 30mm   | 74mm  | 9           |
"""

import time

import numpy as np
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
from loguru import logger

bot = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper", gripper_pressure=1.0, moving_time=1.5)
# bot.shutdown()

def pick_cup():

    bot.arm.go_to_sleep_pose()
    bot.gripper.release()

    bot.arm.set_ee_pose_components(z=0.45, x=0.1)
    bot.arm.set_ee_pose_components(z=0.45, x=0.5)
    bot.gripper.grasp(delay=3.0)
    bot.arm.set_ee_pose_components(z=0.45, x=0.1)
    bot.arm.set_ee_pose_components(z=0.25, x=0.1)

def empty_cup():

    bot.arm.set_ee_pose_components(z=0.25, x=0.1)
    bot.arm.set_ee_pose_components(z=0.45, x=0.1)
    bot.arm.set_ee_pose_components(z=0.45, x=0.1)
    bot.arm.set_ee_pose_components(z=0.45, x=0.5)
    bot.arm.set_ee_pose_components(z=0.45, x=0.5, roll=np.pi)
    time.sleep(1)
    bot.arm.set_ee_pose_components(z=0.45, x=0.5)
    bot.arm.set_ee_pose_components(z=0.45, x=0.1)
    bot.arm.set_ee_pose_components(z=0.25, x=0.1)

def place_cup_to_dishwasher():

    bot.arm.set_ee_pose_components(z=0.25, x=0.1)
    bot.arm.set_ee_pose_components(z=0.3, x=0.5, roll=-np.pi)
    bot.arm.set_ee_pose_components(z=0.1, x=0.5, roll=-np.pi)
    bot.gripper.release()
    bot.arm.set_ee_pose_components(z=0.3, x=0.5, roll=-np.pi)
    bot.arm.set_ee_pose_components(z=0.3, x=0.5)
    bot.arm.go_to_sleep_pose()

def pick_cup_from_dishwasher():

    bot.gripper.release()
    bot.arm.go_to_sleep_pose()

    bot.arm.set_ee_pose_components(z=0.25, x=0.1)
    bot.arm.set_ee_pose_components(z=0.2, x=0.55, y=-0.1, yaw=-np.pi/4)
    bot.arm.set_ee_pose_components(z=0.0, x=0.55, y=-0.1, yaw=-np.pi/4)
    bot.gripper.grasp(delay=2.0)
    bot.arm.set_ee_pose_components(z=0.3, x=0.55, y=-0.1, yaw=-np.pi/4)
    bot.arm.set_ee_pose_components(z=0.25, x=0.1, roll=np.pi)

def place_cup_to_desk():
    bot.arm.set_ee_pose_components(z=0.25, x=0.1, roll=np.pi)
    bot.arm.set_ee_pose_components(z=0.6, x=0.1)
    bot.arm.set_ee_pose_components(z=0.57, x=0.37)
    bot.gripper.release()
    bot.arm.set_ee_pose_components(z=0.6, x=0.1)
    bot.arm.go_to_sleep_pose()

# def place_cup_to_desk():
    # bot.arm.set_ee_pose_components(z=0.25, x=0.1, roll=np.pi)
    # bot.gripper.release()
    # bot.gripper.grasp(delay=2.0)


def pick_cup_from_dishwasher_2():
    bot.gripper.release()
    bot.arm.go_to_sleep_pose()
    bot.arm.set_ee_pose_components(z=0.25, x=0.1)
    bot.arm.set_ee_pose_components(z=0.2, x=0.6, pitch=np.pi/6)
    bot.arm.set_ee_pose_components(z=0.02, x=0.6, pitch=np.pi/6)
    bot.gripper.grasp(delay=2.0)
    bot.arm.set_ee_pose_components(z=0.3, x=0.6)
    bot.arm.set_ee_pose_components(z=0.25, x=0.1, roll=np.pi)



# def pick_cup_from_dishwasher_2():
#     bot.gripper.release()
#     bot.arm.go_to_sleep_pose()

    # bot.arm.set_ee_pose_components(z=0.25, x=0.1)
    # bot.arm.set_ee_pose_components(z=0.2, x=0.6, pitch=np.pi/6)
    # bot.arm.set_ee_pose_components(z=0.0, x=0.6, pitch=np.pi/6)
    # bot.gripper.grasp(delay=2.0)
    # bot.arm.set_ee_pose_components(z=0.3, x=0.6)
    # bot.arm.set_ee_pose_components(z=0.25, x=0.1, roll=np.pi)

# def place_cup_to_desk():
#     # bot.arm.set_ee_pose_components(z=0.25, x=0.1, roll=np.pi)
#     bot.gripper.grasp()

def pick_up_from_dishwasher():

    bot.arm.go_to_sleep_pose()
    bot.gripper.release()

    bot.arm.set_ee_pose_components(z=0.25, x=0.1)
    bot.arm.set_ee_pose_components(z=0.3, x=0.5, roll=-np.pi)
    bot.arm.set_ee_pose_components(z=0.1, x=0.5, roll=-np.pi)
    bot.gripper.release()
    bot.arm.set_ee_pose_components(z=0.3, x=0.5, roll=-np.pi)
    bot.arm.set_ee_pose_components(z=0.3, x=0.5)
    bot.arm.go_to_sleep_pose()


def pick_leaf():
    logger.info(f"Picking up leaf!")

    bot.arm.go_to_sleep_pose()
    bot.gripper.release()

    bot.arm.go_to_home_pose()
    bot.arm.set_single_joint_position(joint_name="wrist_angle", position=np.pi / 7)
    bot.arm.set_single_joint_position(joint_name="shoulder", position=np.pi / 3.6)
    bot.gripper.grasp()

    bot.arm.set_single_joint_position(joint_name="shoulder", position=-np.pi / 2.5)
    bot.arm.set_single_joint_position(joint_name="waist", position=np.pi / 2 + np.pi / 5)
    bot.arm.set_single_joint_position(joint_name="elbow", position=np.pi / 2.5)
    bot.arm.set_single_joint_position(joint_name="wrist_angle", position=np.pi / 4.5)
    bot.gripper.release()

    bot.arm.set_single_joint_position(joint_name="elbow", position=np.pi / 3)
    bot.arm.set_single_joint_position(joint_name="waist", position=0.0)
    bot.arm.go_to_sleep_pose()


def pick_clothes():
    logger.info(f"Picking up clothes!")

    bot.arm.go_to_sleep_pose()
    bot.gripper.set_pressure(1.0)
    # bot.gripper.release(3)
    # bot.gripper.grasp()

    bot.arm.set_joint_positions([0.0, np.pi / 3.7, 0.0, np.pi / 7, 0.0])
    bot.gripper.grasp(2)

    bot.arm.set_joint_positions([0.0, -np.pi / 4, 0.0, np.pi / 4, 0.0])
    bot.arm.set_joint_positions([np.pi / 2 + np.pi / 5, -np.pi / 4, 0.0, np.pi / 4, 0.0])
    bot.arm.set_joint_positions([np.pi / 2 + np.pi / 5, -np.pi / 3, np.pi / 3, np.pi / 7, 0.0])
    bot.gripper.release()

    bot.arm.set_joint_positions([0.0, -np.pi / 3, np.pi / 4, np.pi / 7, 0.0])
    bot.arm.go_to_sleep_pose()


def press_button():
    logger.info(f"Pressing button!")

    bot.arm.go_to_sleep_pose()
    bot.gripper.grasp()

    bot.arm.set_joint_positions([0.0, -np.pi / 12, 0.0, np.pi / 12, 0.0])

    bot.arm.set_joint_positions([0.0, 0.0, -np.pi / 40, 0.0, 0.0])

    bot.arm.go_to_sleep_pose()
    bot.gripper.release()


def rotate_grasp():
    logger.info(f"Rotating!")

    bot.arm.go_to_sleep_pose()
    bot.gripper.set_pressure(0.5)
    bot.gripper.release()

    bot.arm.set_joint_positions([0.0, -np.pi / 12, 0.0, np.pi / 12, 0.0])
    bot.gripper.grasp()

    bot.arm.set_joint_positions([0.0, -np.pi / 12, 0.0, np.pi / 12, np.pi / 2])
    bot.gripper.release()

    bot.arm.go_to_sleep_pose()
    bot.gripper.set_pressure(1.0)
    bot.gripper.release()


def wipe():
    logger.info(f"Wiping!")

    bot.arm.go_to_sleep_pose()
    bot.gripper.release()

    bot.arm.set_joint_positions([0.0, -np.pi / 4, 0.0, np.pi / 4, 0.0])
    bot.arm.set_joint_positions([np.pi / 2 + np.pi / 5, -np.pi / 8, np.pi / 3, np.pi / 5, 0.0])
    bot.gripper.grasp()

    bot.arm.set_joint_positions([0.0, -np.pi / 4, 0.0, np.pi / 4, 0.0])

    bot.arm.set_joint_positions([0.0, np.pi / 4.1, -np.pi / 4, np.pi / 2.5, 0.0])
    bot.arm.set_joint_positions([np.pi / 6, np.pi / 4.1, -np.pi / 4, np.pi / 2.5, 0.0])
    bot.arm.set_joint_positions([-np.pi / 6, np.pi / 3.8, -np.pi / 4, np.pi / 2.5, 0.0])
    bot.arm.set_joint_positions([np.pi / 6, np.pi / 4.1, -np.pi / 4, np.pi / 2.5, 0.0])
    bot.arm.set_joint_positions([0.0, np.pi / 4.1, -np.pi / 4, np.pi / 2.5, 0.0])

    bot.arm.set_joint_positions([0.0, -np.pi / 4, 0.0, np.pi / 4, 0.0])
    bot.arm.set_joint_positions([np.pi / 2 + np.pi / 5, -np.pi / 3, np.pi / 3, np.pi / 7, 0.0])
    bot.gripper.release()

    bot.arm.set_joint_positions([0.0, -np.pi / 3, np.pi / 4, np.pi / 7, 0.0])
    bot.arm.go_to_sleep_pose()



def wave():
    logger.info(f"Waving!")

    bot.arm.go_to_sleep_pose()
    bot.gripper.set_pressure(1.0)
    # bot.gripper.release(3)
    # bot.gripper.grasp()

    # bot.arm.set_joint_positions([0.0, np.pi / 3.7, 0.0, np.pi / 7, 0.0])
    # bot.gripper.grasp(2)

    # bot.arm.set_joint_positions([0.0, -np.pi / 4, 0.0, np.pi / 4, 0.0])
    # bot.arm.set_joint_positions([np.pi / 2 + np.pi / 5, -np.pi / 4, 0.0, np.pi / 4, 0.0])
    # bot.arm.set_joint_positions([np.pi / 2 + np.pi / 5, -np.pi / 3, np.pi / 3, np.pi / 7, 0.0])
    # bot.gripper.release()

    # bot.arm.set_joint_positions([0.0, -np.pi / 3, np.pi / 4, np.pi / 7, 0.0])
    bot.arm.go_to_home_pose()
    bot.arm.set_joint_positions([0.0, 0.0, 0.0, -np.pi / 2, 0.0], moving_time=1.0)
    bot.arm.set_joint_positions([np.pi / 4, 0.0, 0.0, -np.pi / 2, 0.0], moving_time=1.0)
    bot.arm.set_joint_positions([-np.pi / 4, 0.0, 0.0, -np.pi / 2, 0.0], moving_time=2.0)
    bot.arm.set_joint_positions([np.pi / 4, 0.0, 0.0, -np.pi / 2, 0.0], moving_time=2.0)
    bot.arm.set_joint_positions([-np.pi / 4, 0.0, 0.0, -np.pi / 2, 0.0], moving_time=2.0)
    bot.arm.set_joint_positions([0.0, 0.0, 0.0, -np.pi / 2, 0.0], moving_time=1.0)
    bot.arm.go_to_sleep_pose()

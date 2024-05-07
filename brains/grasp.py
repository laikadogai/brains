"""
Waist        -180 180
Shoulder     -108 114
Elbow        -108 93
Wrist Angle  -100 123
Wrist Rotate -180 180
Gripper
"""

import numpy as np
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
from loguru import logger

bot = InterbotixManipulatorXS(robot_model="wx250", group_name="arm", gripper_name="gripper", gripper_pressure=1.0)
# bot.shutdown()


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

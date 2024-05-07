import sys
import time

import numpy as np
from interbotix_xs_modules.xs_robot.arm import InterbotixGripperXSInterface, InterbotixManipulatorXS
from loguru import logger

"""
This script makes the end-effector perform pick, pour, and place tasks.
Note that this script may not work for every arm as it was designed for the wx250.
Make sure to adjust commanded joint positions and poses as necessary.

To get started, open a terminal and type:

    ros2 launch interbotix_xsarm_control xsarm_control.launch.py robot_model:=wx250
"""


def release_and_wait(gripper: InterbotixGripperXSInterface):
    """Command the gripper to release and wait until it stops moving."""
    gripper.release()  # command the gripper to release
    # while gripper.gripper_moving:  # check if the gripper is still moving
    #     logger.info("Waiting for gripper release...")
    #     time.sleep(0.1)  # wait briefly before checking again


def grasp_and_wait(gripper: InterbotixGripperXSInterface):
    """Command the gripper to grasp with specified force and wait until it stops moving."""
    gripper.set_pressure(1.0)
    gripper.grasp()  # command the gripper to grasp
    while gripper.gripper_moving:  # check if the gripper is still moving
        time.sleep(0.1)  # wait briefly before checking again
    time.sleep(3.0)


def forward_grasp():
    bot = InterbotixManipulatorXS(robot_model="wx250", group_name="arm", gripper_name="gripper")

    if bot.arm.group_info.num_joints < 5:
        bot.core.get_logger().fatal("This demo requires the robot to have at least 5 joints!")
        bot.shutdown()
        sys.exit()

    logger.info(f"Performing grasping!")

    # release_and_wait(bot.gripper)

    bot.arm.go_to_sleep_pose()
    bot.arm.go_to_home_pose()
    # bot.arm.go_to_sleep_pose()
    release_and_wait(bot.gripper)

    # bot.arm.set_joint_positions()
    bot.arm.set_single_joint_position(joint_name="wrist_angle", position=np.pi / 7)
    bot.arm.set_single_joint_position(joint_name="shoulder", position=np.pi / 3.6)
    bot.gripper.grasp(0)
    time.sleep(2)
    # bot.arm.go_to_home_pose()
    # bot.arm.set_single_joint_position(joint_name="shoulder", position=-np.pi / gcj)
    # bot.arm.set_single_joint_position(joint_name="elbow", position=np.pi / 2.5)

    bot.arm.set_single_joint_position(joint_name="shoulder", position=-np.pi / 2.5)
    # bot.arm.set_single_joint_position(joint_name="waist", position=np.pi / 2 + np.pi / 4)
    bot.arm.set_single_joint_position(joint_name="waist", position=np.pi / 2 + np.pi / 5)
    bot.arm.set_single_joint_position(joint_name="elbow", position=np.pi / 2.5)
    bot.arm.set_single_joint_position(joint_name="wrist_angle", position=np.pi / 4.5)
    release_and_wait(bot.gripper)
    bot.arm.set_single_joint_position(joint_name="elbow", position=np.pi / 3)
    # bot.arm.set_single_joint_position(joint_name="wrist_angle", position=np.pi / 4)
    bot.arm.set_single_joint_position(joint_name="waist", position=0.0)
    bot.arm.go_to_sleep_pose()
    bot.shutdown()

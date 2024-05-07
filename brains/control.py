import asyncio
import math
from typing import Tuple

import rclpy
from geometry_msgs.msg import Twist
from loguru import logger
from rclpy.duration import Duration
from rclpy.node import Node
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener
from tf_transformations import euler_from_quaternion

from brains import args
from brains.camera import find_object
from brains.grasp import forward_grasp
from brains.utils import play_text
from clients.sport_client import SportClient, SportState
from communicator.cyclonedds.ddsCommunicator import DDSCommunicator

communicator = DDSCommunicator(interface="eth0")
client = SportClient(communicator)
state = SportState(communicator)


# class ROS2BrainNode(Node):
#     def __init__(self):
#         super().__init__("laika_brain_node")
#         self.tf_buffer = Buffer()
#         self.transform_listener = TransformListener(self.tf_buffer, self)
#         self.acceleration_publisher = self.create_publisher(Twist, "cmd_vel", 10)
#         self.command_publisher = self.create_publisher(String, "command", 10)

#     def get_transform_once(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
#         timeout = Duration(seconds=10)
#         end_time = self.get_clock().now() + timeout

#         while rclpy.ok():
#             current_time = self.get_clock().now()
#             if current_time > end_time:
#                 logger.error("Timeout reached, failed to get transform.")
#                 raise Exception
#             try:
#                 # Look up the transformation (use a short timeout for lookup to keep trying)
#                 trans = self.tf_buffer.lookup_transform(
#                     "odom", "base_link", rclpy.time.Time(), timeout=Duration(seconds=1)
#                 )
#                 translation = trans.transform.translation
#                 rotation = trans.transform.rotation
#                 euler_rad: Tuple[float, float, float] = euler_from_quaternion(
#                     [rotation.x, rotation.y, rotation.z, rotation.w]
#                 )
#                 euler_deg = tuple(math.degrees(angle) for angle in euler_rad)

#                 # logger.info(f"Translation: {translation.x}, {translation.y}, {translation.z}")
#                 # logger.info(f"Rotation (Quaternion): {rotation.x}, {rotation.y}, {rotation.z}, {rotation.w}")
#                 # logger.info(f"Rotation in RPY (radian): {euler_rad}")
#                 # logger.info(f"Rotation in RPY (degree): {euler_deg}")
#                 return ((translation.x, translation.y, translation.z), euler_deg)
#             except Exception as e:
#                 logger.debug("Transform not available, retrying...")
#                 rclpy.spin_once(self, timeout_sec=0.01)  # Short wait before retrying

#     def rotate(self, n: int):
#         n *= -1
#         _, initial_rotation = self.get_transform_once()
#         target_angle = (initial_rotation[2] + n) % 360
#         current_angle = initial_rotation[2]

#         logger.info(f"Current angle: {current_angle}")
#         logger.info(f"Target angle: {target_angle}")

#         msg = Twist()
#         msg.angular.z = 0.2 * n / abs(n)

#         while abs(current_angle - target_angle) > 10:  # Allow for a small error margin
#             self.acceleration_publisher.publish(msg)
#             rclpy.spin_once(self, timeout_sec=0.05)  # Short wait to allow for new transform to be available
#             try:
#                 _, current_rotation = self.get_transform_once()
#                 current_angle = current_rotation[2] % 360
#                 logger.info(current_angle)
#             except Exception:
#                 continue

#         msg.angular.z = 0.0
#         for _ in range(10):
#             self.acceleration_publisher.publish(msg)

#     def move(self, n: float):
#         (x_initial, y_initial, _), _ = self.get_transform_once()
#         x_current, y_current = x_initial, y_initial

#         msg = Twist()
#         msg.linear.x = 0.1

#         while math.sqrt((x_current - x_initial) ** 2 + (y_current - y_initial) ** 2) < n - 0.2:
#             self.acceleration_publisher.publish(msg)
#             rclpy.spin_once(self, timeout_sec=0.05)  # Short wait to allow for new transform to be available
#             try:
#                 (x_current, y_current, _), _ = self.get_transform_once()
#             except Exception:
#                 continue

#         msg.linear.x = 0.0
#         for _ in range(10):
#             self.acceleration_publisher.publish(msg)

#     def send_command(self, command: str):
#         msg = String()
#         msg.data = command
#         self.command_publisher.publish(msg)
#         logger.info(f"Published command: {command}")


async def rotate(n: int, velocity: float = 0.4, tolerance: int = 5):
    n *= -1

    current_angle = None
    rotation_achieved = asyncio.Event()

    def on_state_update(x):
        nonlocal current_angle
        current_radians = x.imu_state.rpy[2]
        current_angle = current_radians * 180 / math.pi
        # logger.info(current_angle)
        if abs(current_angle - target_angle) < tolerance:
            rotation_achieved.set()

    state.add_callback(on_state_update)

    while current_angle is None:
        logger.debug("Waiting for angle to be read...")
        await asyncio.sleep(0.1)

    target_angle = (current_angle + n) % 360
    if target_angle > 180:
        target_angle -= 360

    logger.info(f"Rotating from {current_angle:.2f} degrees to {target_angle:.2f} degrees.")
    while not rotation_achieved.is_set():
        await client.Move({"z": (1, -1)[n < 0] * velocity})
        await asyncio.sleep(0.1)

    # state.remove_callback(on_state_update)

    logger.info(f"Target rotation achieved: {current_angle:.2f} degrees.")


async def move(n: float, velocity: float = 0.2, offset: float = 0.15):

    current_position = None
    position_achieved = asyncio.Event()

    def on_state_update(x):
        nonlocal current_position
        current_position = x.position
        if math.sqrt((current_position[0] - initial_position[0]) ** 2 + (current_position[1] - initial_position[1]) ** 2) >= n - offset:
            position_achieved.set()

    state.add_callback(on_state_update)

    while current_position is None:
        logger.debug("Waiting for position to be read...")
        await asyncio.sleep(0.1)

    initial_position = current_position

    logger.info(f"Moving from {initial_position} forward for {n:.2f} meters.")
    while not position_achieved.is_set():
        await client.Move({"x": velocity})
        await asyncio.sleep(0.1)

    # state.remove_callback(on_state_update)

    logger.info(f"Ended up in {current_position} after moving for forward for {n:.2f} meters.")

async def collect_leaves():

    # await client.StandDown()
    # forward_grasp()
    # return

    await client.StandUp()
    await client.BalanceStand()
    # return

    degrees_rotate = 45

    play_text(f"I'm picking up all the leaves!")

    cnt, position = 0, None
    while cnt < math.ceil(360 / abs(degrees_rotate)):
        play_text(f"Searching for a leaf!")
        position = find_object(args.object_search_string)
        if not position:
            cnt += 1
            if cnt < math.ceil(360 / abs(degrees_rotate)):
                await rotate(degrees_rotate)
        else:
            logger.info(position)
            x, _, z = position
            angle = int(math.atan2(x, z) * 180 / math.pi)
            distance = math.sqrt(x**2 + z**2)
            logger.info(f"angle: {angle}, distance: {distance}")
            play_text(f"Found a leaf in {distance:.1f} meters. Approaching it!")
            await rotate(int(angle), velocity=0.1, tolerance=8)
            if distance > 0.8:
                distance = min(distance - 0.7, 0.7)
                play_text(f"Will approach the leaf in a couple of chunks. Moving for {distance:.1f} meters.")
                await move(distance)
            else:
                await move(distance)
                await asyncio.sleep(2)
                await client.StandDown()
                play_text(f"Approached the leaf! Grasping it.")
                forward_grasp()
                await client.StandUp()
                await client.BalanceStand()
            cnt = 0

    play_text(f"No more leaves!")

    # rclpy.init()
    # node = ROS2BrainNode()
    # node.send_command("StandDown")
    # node.destroy_node()
    # rclpy.shutdown()

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


class ROS2BrainNode(Node):
    def __init__(self):
        super().__init__("laika_brain_node")
        self.tf_buffer = Buffer()
        self.transform_listener = TransformListener(self.tf_buffer, self)
        self.acceleration_publisher = self.create_publisher(Twist, "cmd_vel", 10)
        self.command_publisher = self.create_publisher(String, "command", 10)

    def get_transform_once(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        timeout = Duration(seconds=4)
        end_time = self.get_clock().now() + timeout

        while rclpy.ok():
            current_time = self.get_clock().now()
            if current_time > end_time:
                logger.error("Timeout reached, failed to get transform.")
                raise Exception
            try:
                # Look up the transformation (use a short timeout for lookup to keep trying)
                trans = self.tf_buffer.lookup_transform(
                    "odom", "base_link", rclpy.time.Time(), timeout=Duration(seconds=1)
                )
                translation = trans.transform.translation
                rotation = trans.transform.rotation
                euler_rad: Tuple[float, float, float] = euler_from_quaternion(
                    [rotation.x, rotation.y, rotation.z, rotation.w]
                )
                euler_deg = tuple(math.degrees(angle) for angle in euler_rad)

                # logger.info(f"Translation: {translation.x}, {translation.y}, {translation.z}")
                # logger.info(f"Rotation (Quaternion): {rotation.x}, {rotation.y}, {rotation.z}, {rotation.w}")
                # logger.info(f"Rotation in RPY (radian): {euler_rad}")
                # logger.info(f"Rotation in RPY (degree): {euler_deg}")
                return ((translation.x, translation.y, translation.z), euler_deg)
            except Exception as e:
                logger.debug("Transform not available, retrying...")
                rclpy.spin_once(self, timeout_sec=0.01)  # Short wait before retrying

    def rotate(self, n: int):
        n *= -1
        _, initial_rotation = self.get_transform_once()
        target_angle = (initial_rotation[2] + n) % 360
        current_angle = initial_rotation[2]

        logger.info(f"Current angle: {current_angle}")
        logger.info(f"Target angle: {target_angle}")

        msg = Twist()
        msg.angular.z = 0.2 * n / abs(n)

        while abs(current_angle - target_angle) > 10:  # Allow for a small error margin
            self.acceleration_publisher.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.05)  # Short wait to allow for new transform to be available
            try:
                _, current_rotation = self.get_transform_once()
                current_angle = current_rotation[2] % 360
                logger.info(current_angle)
            except Exception:
                continue

        msg.angular.z = 0.0
        for _ in range(10):
            self.acceleration_publisher.publish(msg)

    def move(self, n: float):
        (x_initial, y_initial, _), _ = self.get_transform_once()
        x_current, y_current = x_initial, y_initial

        msg = Twist()
        msg.linear.x = 0.1

        while math.sqrt((x_current - x_initial) ** 2 + (y_current - y_initial) ** 2) < n - 0.2:
            self.acceleration_publisher.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.05)  # Short wait to allow for new transform to be available
            try:
                (x_current, y_current, _), _ = self.get_transform_once()
            except Exception:
                continue

        msg.linear.x = 0.0
        for _ in range(10):
            self.acceleration_publisher.publish(msg)

    def send_command(self, command: str):
        msg = String()
        msg.data = command
        self.command_publisher.publish(msg)
        logger.info(f"Published command: {command}")


def search():
    forward_grasp()
    # play_text(f"Found a leaf in {0.2} meters")

    rclpy.init()
    node = ROS2BrainNode()

    # node.send_command("StandUp")
    # node.send_command("StandDown")

    # degrees_rotate = -90
    # position = None
    # cnt = 0
    # while not position and cnt < math.ceil(360 / abs(degrees_rotate)):
    #     position = find_object(args.object_search_string)
    #     if not position:
    #         node.rotate(degrees_rotate)
    #     cnt += 1
    # if position:
    #     logger.info(position)
    #     x, _, z = position
    #     angle = int(math.atan2(x, z) * 180 / math.pi)
    #     distance = math.sqrt(x**2 + z**2)
    #     logger.info(f"angle: {angle}, distance: {distance}")
    #     play_text(f"Found a leaf in {distance:.2f} meters")
    #     if angle != 0:
    #         node.rotate(angle)
    #     node.move(distance)
    #     node.send_command("StandDown")
    # else:
    #     logger.info(f"'{args.object_search_string}' was not found.")

    node.destroy_node()
    rclpy.shutdown()

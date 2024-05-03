import math
from typing import Tuple

import rclpy
from geometry_msgs.msg import Twist
from loguru import logger
from rclpy.duration import Duration
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener
from tf_transformations import euler_from_quaternion

from brains import args
from brains.camera import find_object


class TF2ListenerNode(Node):
    def __init__(self):
        super().__init__("tf2_listener")
        self.tf_buffer = Buffer()
        self.listener = TransformListener(self.tf_buffer, self)
        self.publisher = self.create_publisher(Twist, "cmd_vel", 10)  # Create a Twist message publisher

    def get_transform_once(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        timeout = Duration(seconds=4)  # Adjust timeout duration as needed.
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
                euler_rad = euler_from_quaternion([rotation.x, rotation.y, rotation.z, rotation.w])
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
        _, initial_rotation = self.get_transform_once()
        target_angle = (initial_rotation[2] + n) % 360
        current_angle = initial_rotation[2]

        logger.info(f"Current angle: {current_angle}")
        logger.info(f"Target angle: {target_angle}")

        msg = Twist()
        msg.angular.z = 0.2  # angular velocity

        while abs(current_angle - target_angle) > 10:  # Allow for a small error margin
            self.publisher.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.05)  # Short wait to allow for new transform to be available
            try:
                _, current_rotation = self.get_transform_once()
                current_angle = current_rotation[2] % 360
                logger.info(current_angle)
            except Exception:
                continue

        msg.angular.z = 0.0
        for _ in range(10):
            self.publisher.publish(msg)


def search():
    rclpy.init()
    node = TF2ListenerNode()

    degrees_rotate = 90
    position = None
    cnt = 0
    while not position and cnt < math.ceil(360 / degrees_rotate):
        position = find_object(args.object_search_string)
        if not position:
            node.rotate(degrees_rotate)

    if position:
        logger.info(position)
    else:
        logger.info(f"'{args.object_search_string}' was not found.")

    node.destroy_node()
    rclpy.shutdown()

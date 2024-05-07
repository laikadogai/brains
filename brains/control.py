import asyncio
import math

from loguru import logger

from brains import args
from brains.camera import find_object
from brains.grasp import pick_clothes, pick_leaf, press_button, rotate_grasp
from brains.utils import play_text
from clients.sport_client import SportClient, SportState
from communicator.constants import WEBRTC_TOPICS
from communicator.cyclonedds.ddsCommunicator import DDSCommunicator
from communicator.idl.std_msgs.msg.dds_ import String_

communicator = DDSCommunicator(interface="eth0")
client = SportClient(communicator)
state = SportState(communicator)

communicator.publish(WEBRTC_TOPICS["ULIDAR_SWITCH"], String_(data='"OFF"'), String_)


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

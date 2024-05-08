import asyncio
import math
from typing import List

import inflect
from loguru import logger

from brains import args
from brains.camera import find_object, get_sorted_matches
from brains.grasp import pick_clothes
from brains.utils import play_text
from clients.sport_client import SportClient, SportState
from communicator.constants import WEBRTC_TOPICS
from communicator.cyclonedds.ddsCommunicator import DDSCommunicator
from communicator.idl.std_msgs.msg.dds_ import String_

communicator = DDSCommunicator(interface="eth0")
communicator.publish(WEBRTC_TOPICS["ULIDAR_SWITCH"], String_(data='"OFF"'), String_)
client = SportClient(communicator)
state = SportState(communicator)


p = inflect.engine()


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
        if (
            math.sqrt(
                (current_position[0] - initial_position[0]) ** 2 + (current_position[1] - initial_position[1]) ** 2
            )
            >= n - offset
        ):
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


async def collect_items(items: List[str]):

    await client.StandUp()
    await client.BalanceStand()

    items_plural_search_string = p.join([p.plural(item) for item in items])  # type: ignore
    play_text(f"I'm picking up all {items_plural_search_string}!")

    cnt = 0
    while cnt < math.ceil(360 / abs(args.search_degrees_rotate)):
        play_text(f"Searching for {items_plural_search_string}!")
        object_predictions = find_object(". ".join([p.a(item) for item in items]) + ".")  # type: ignore

        if not object_predictions:
            cnt += 1
            if cnt < math.ceil(360 / abs(args.search_degrees_rotate)):
                await rotate(args.search_degrees_rotate)
        else:
            sorted_matches = get_sorted_matches(object_predictions)
            label, distance, angle = sorted_matches[0]

            if len(sorted_matches) > 1:
                for lbl, dst, _ in sorted_matches:
                    play_text(f"Found {lbl} in {dst:.1f} meters")
                play_text(f"Will aproach {label} as it is closest to me")
            else:
                play_text(f"Found {label} in {distance:.1f} meters. Approaching it!")
            logger.info(f"object: {label}, angle: {angle}, distance: {distance}")
            await rotate(int(angle), velocity=0.1, tolerance=8)
            if distance > 0.8:
                distance = min(distance - 0.7, 0.7)
                play_text(f"Will approach {label} in a couple of chunks. Moving for {distance:.1f} meters.")
                await move(distance)
            else:
                await move(distance)
                await asyncio.sleep(2)
                await client.StandDown()
                play_text(f"Approached {label}! Grasping it.")
                pick_clothes()
                await client.StandUp()
                await client.BalanceStand()
            cnt = 0

    play_text(f"No more {items_plural_search_string}!")

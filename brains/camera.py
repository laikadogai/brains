import time
from typing import List, Tuple, TypedDict

import cv2
import numpy as np
import pyrealsense2 as rs
import torch
from loguru import logger
from numpy._typing import NDArray
from PIL import Image
from torch import Tensor
from transformers import GroundingDinoForObjectDetection, GroundingDinoProcessor

from brains import args

object_detection_processor = GroundingDinoProcessor.from_pretrained(args.object_detection_model_id)
object_detection_model = GroundingDinoForObjectDetection.from_pretrained(args.object_detection_model_id)
if (
    type(object_detection_processor) != GroundingDinoProcessor
    or type(object_detection_model) != GroundingDinoForObjectDetection
):
    raise TypeError


class ObjectDetectionResult(TypedDict):
    scores: Tensor
    labels: List[str]
    boxes: Tensor


def get_camera_frame() -> Tuple[NDArray[np.uint8], NDArray[np.uint16], NDArray[np.float64], NDArray[np.float64]]:

    pipeline = rs.pipeline()
    config = rs.config()

    # Setting up stream parameters
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device_product_line = str(pipeline_profile.get_device().get_info(rs.camera_info.product_line))
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    if device_product_line == "L500":
        config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
    else:
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    align = rs.align(rs.stream.color)
    pipeline.start(config)

    time.sleep(1)

    # Waiting for both color and depth frames to appear
    depth_frame, color_frame = None, None
    while not color_frame or not depth_frame:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())

    # Converting depth from uint16 to meters
    # https://github.com/IntelRealSense/librealsense/issues/2481#issuecomment-428651169
    h, w = color_image.shape[0], color_image.shape[1]
    depth_image_projected = np.zeros((h, w))
    for x in range(h):
        for y in range(w):
            depth_image_projected[x, y] = depth_frame.get_distance(y, x)

    # Creating camera matrix
    depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
    camera_matrix = np.array(
        [[depth_intrinsics.fx, 0, depth_intrinsics.ppx], [0, depth_intrinsics.fy, depth_intrinsics.ppy], [0, 0, 1]]
    )

    return color_image, depth_image, depth_image_projected, camera_matrix


def visualize_frame(color_image: NDArray[np.uint8], depth_image: NDArray[np.uint16]):

    # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

    # If depth and color resolutions are different, resize color image to match depth image for display
    depth_colormap_dim = depth_colormap.shape
    color_colormap_dim = color_image.shape
    if depth_colormap_dim != color_colormap_dim:
        resized_color_image = cv2.resize(
            color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA
        )
        images = np.hstack((resized_color_image, depth_colormap))
    else:
        images = np.hstack((color_image, depth_colormap))

    # Show images
    cv2.namedWindow("RealSense", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("RealSense", images)
    cv2.waitKey(0)


def draw_bounding_boxes(image: NDArray[np.uint8], bbox_data: List[ObjectDetectionResult]) -> NDArray[np.uint8]:
    for bbox in bbox_data:
        for box, label in zip(bbox["boxes"], bbox["labels"]):
            start_point = (int(box[0]), int(box[1]))
            end_point = (int(box[2]), int(box[3]))
            color = (255, 0, 0)  # Blue color in BGR
            thickness = 2
            image = cv2.rectangle(image, start_point, end_point, color, thickness)
            cv2.putText(image, label, (start_point[0], start_point[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image


def find_object(search_string: str) -> Tuple[float, float, float] | None:

    logger.info(f"Searching for {search_string}...")
    color_image, depth_image, depth_image_projected, camera_matrix = get_camera_frame()
    image = Image.fromarray(color_image)

    inputs = object_detection_processor(images=image, text=search_string, return_tensors="pt")
    with torch.no_grad():
        outputs = object_detection_model(**inputs)

    results: List[ObjectDetectionResult] = object_detection_processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids, box_threshold=0.4, text_threshold=0.3, target_sizes=[image.size[::-1]]
    )

    if not results[0]["labels"]:
        return None

    logger.info(f"Object detection results: {results}")
    if results[0]["boxes"][0][2] - results[0]["boxes"][0][0] > 300:
        logger.info(f"Bounding box too big, probably false positive")
        return None

    [[fx, _, ppx], [_, fy, ppy], [_, _, _]] = camera_matrix

    x_image, y_image = int((results[0]["boxes"][0][0] + results[0]["boxes"][0][2]) / 2), int(
        (results[0]["boxes"][0][1] + results[0]["boxes"][0][3]) / 2
    )

    z: float = depth_image_projected[y_image, x_image]
    logger.info(f"z={z}")
    x: float = (x_image - ppx) * z / fx
    y: float = (y_image - ppy) * z / fy

    logger.info(f"Camera coordinates of a {search_string}: x={x:.3f}, y={y:.3f}, z={z:.3f} meters")

    result_image = draw_bounding_boxes(color_image, results)
    visualize_frame(result_image, depth_image)

    return (x, y, z)

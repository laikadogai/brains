import os
import time
from typing import List, Tuple, TypedDict

import cv2
import numpy as np
import pyrealsense2 as rs
import rclpy
import torch
from cv_bridge import CvBridge, CvBridgeError
from loguru import logger
from numpy._typing import NDArray
from PIL import Image as PILImage
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from torch import Tensor
from transformers import GroundingDinoForObjectDetection, GroundingDinoProcessor

from brains import args

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
logger.info(f"Detected device: {device}")

object_detection_processor = GroundingDinoProcessor.from_pretrained(args.object_detection_model_id)
object_detection_model = GroundingDinoForObjectDetection.from_pretrained(args.object_detection_model_id).to(device)
if (
    type(object_detection_processor) != GroundingDinoProcessor
    or type(object_detection_model) != GroundingDinoForObjectDetection
):
    raise TypeError


class CameraSubscriberNode(Node):
    def __init__(self):
        super().__init__('camera_subscriber_node')
        self.bridge = CvBridge()
        self.latest_color_image = None
        self.latest_depth_image = None
        self.latest_depth_in_meters = None
        self.camera_matrix = None
        self.received_color = False
        self.received_depth = False

        # Subscriptions
        self.create_subscription(Image, '/camera/camera/color/image_raw', self.color_callback, 10)
        self.create_subscription(Image, '/camera/camera/depth/image_rect_raw', self.depth_callback, 10)
        self.create_subscription(CameraInfo, '/camera/camera/depth/camera_info', self.camera_info_callback, 10)

    def color_callback(self, msg):
        try:
            self.latest_color_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.received_color = True
            logger.info(f"Got color image with shape: {self.latest_color_image.shape}")
        except CvBridgeError as e:
            logger.error(f'Error converting color image: {e}')

    def depth_callback(self, msg):
        try:
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='16UC1')
            if self.camera_matrix is not None:
                self.latest_depth_image = depth_image
                self.latest_depth_in_meters = depth_image * 0.001  # Assume depth scale is 0.001
                # self.latest_depth_in_meters = depth_image
                self.received_depth = True
                logger.info(f"Got depth image with shape: {depth_image.shape}")
        except CvBridgeError as e:
            logger.error(f'Error converting depth image: {e}')

    def camera_info_callback(self, msg):
        fx, fy = msg.k[0], msg.k[4]
        cx, cy = msg.k[2], msg.k[5]
        self.camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])


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
    config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
    if device_product_line == "L500":
        config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
    else:
        config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)

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


def get_camera_frame_ros() -> Tuple[NDArray[np.uint8], NDArray[np.uint16], NDArray[np.float64], NDArray[np.float64]]:
    rclpy.init(domain_id=int(os.environ["ROS_DOMAIN_ID"]))
    node = CameraSubscriberNode()

    while rclpy.ok():
        rclpy.spin_once(node)
        if node.received_color and node.received_depth:
            break

    color_image = node.latest_color_image
    depth_image = node.latest_depth_image
    depth_image_projected = node.latest_depth_in_meters
    camera_matrix = node.camera_matrix

    node.destroy_node()
    rclpy.shutdown()

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
    color_image, depth_image, depth_image_projected, camera_matrix = get_camera_frame_ros()
    # color_image, depth_image, depth_image_projected, camera_matrix = get_camera_frame()
    # logger.info(camera_matrix)
    # logger.info(depth_image.shape)
    # logger.info(f"Nonzero: {np.count_nonzero(depth_image)} / {depth_image.size}")
    # logger.info(f"Mean of nonzero: {np.mean(depth_image[depth_image != 0])}")
    image = PILImage.fromarray(color_image)

    inputs = object_detection_processor(images=image, text=search_string, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = object_detection_model(**inputs)

    results: List[ObjectDetectionResult] = object_detection_processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.4,
        text_threshold=0.3,
        target_sizes=[image.size[::-1]],
        # outputs, inputs.input_ids, box_threshold=0.4, text_threshold=0.3, target_sizes=[image.size[::-1]]
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

    depth_image_projected_bb = depth_image_projected[int(results[0]["boxes"][0][1]) : int(results[0]["boxes"][0][3]), int(results[0]["boxes"][0][0]) : int(results[0]["boxes"][0][2])]
    # logger.info(depth_image_projected_bb)
    # logger.info(f"Nonzero bb: {np.count_nonzero(depth_image_projected_bb)} / {depth_image_projected_bb.size}")

    # z: float = depth_image_projected[y_image, x_image]
    z = np.mean(depth_image_projected_bb[depth_image_projected_bb != 0])
    x: float = (x_image - ppx) * z / fx
    y: float = (y_image - ppy) * z / fy

    logger.info(f"Camera coordinates of a {search_string}: x={x:.3f}, y={y:.3f}, z={z:.3f} meters")

    result_image = draw_bounding_boxes(color_image, results)
    visualize_frame(result_image, depth_image)

    return (x, y, z)

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

if not rclpy.ok():
    rclpy.init(domain_id=int(os.environ["ROS_DOMAIN_ID"]))

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
        super().__init__("camera_subscriber_node")
        self.bridge = CvBridge()
        self.latest_color_image = None
        self.latest_depth_image = None
        self.latest_depth_in_meters = None
        self.camera_matrix = None
        self.received_color = False
        self.received_depth = False

        # Subscriptions
        self.create_subscription(Image, "/camera/camera/color/image_raw", self.color_callback, 10)
        self.create_subscription(Image, "/camera/camera/depth/image_rect_raw", self.depth_callback, 10)
        self.create_subscription(CameraInfo, "/camera/camera/depth/camera_info", self.camera_info_callback, 10)

    def color_callback(self, msg):
        try:
            self.latest_color_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.received_color = True
            logger.info(f"Got color image with shape: {self.latest_color_image.shape}")
        except CvBridgeError as e:
            logger.error(f"Error converting color image: {e}")

    def depth_callback(self, msg):
        try:
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="16UC1")
            if self.camera_matrix is not None:
                self.latest_depth_image = depth_image
                self.latest_depth_in_meters = depth_image * 0.001  # Assume depth scale is 0.001
                # self.latest_depth_in_meters = depth_image
                self.received_depth = True
                logger.info(f"Got depth image with shape: {depth_image.shape}")
        except CvBridgeError as e:
            logger.error(f"Error converting depth image: {e}")

    def camera_info_callback(self, msg):
        fx, fy = msg.k[0], msg.k[4]
        cx, cy = msg.k[2], msg.k[5]
        self.camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])


class ObjectDetectionResult(TypedDict):
    scores: Tensor
    labels: List[str]
    boxes: Tensor


class ObjectDetectionResultTransformed(TypedDict):
    score: Tensor
    label: str
    box: Tensor


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

    camera_subscriber_node = CameraSubscriberNode()

    while not (camera_subscriber_node.received_color and camera_subscriber_node.received_depth):
        rclpy.spin_once(camera_subscriber_node)

    color_image = camera_subscriber_node.latest_color_image
    depth_image = camera_subscriber_node.latest_depth_image
    depth_image_projected = camera_subscriber_node.latest_depth_in_meters
    camera_matrix = camera_subscriber_node.camera_matrix

    camera_subscriber_node.destroy_node()

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


def draw_bounding_boxes(image: NDArray[np.uint8], bbox: ObjectDetectionResult) -> NDArray[np.uint8]:
    for box, label in zip(bbox["boxes"], bbox["labels"]):
        start_point = (int(box[0]), int(box[1]))
        end_point = (int(box[2]), int(box[3]))
        color = (255, 0, 0)  # Blue color in BGR
        thickness = 2
        image = cv2.rectangle(image, start_point, end_point, color, thickness)
        cv2.putText(image, label, (start_point[0], start_point[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image


def process_single_bounding_box(
    result: ObjectDetectionResultTransformed,
    depth_image_projected: NDArray[np.float64],
    camera_matrix: NDArray[np.float64],
) -> Tuple[Tuple[float, float, float], str] | None:

    if not result["label"]:
        return None

    if result["box"][2] - result["box"][0] > 300:
        logger.info(f"Bounding box too big, probably false positive")
        return None

    x_image, y_image = int((result["box"][0] + result["box"][2]) / 2), int((result["box"][1] + result["box"][3]) / 2)

    depth_image_projected_bb = depth_image_projected[
        int(result["box"][1]) : int(result["box"][3]),
        int(result["box"][0]) : int(result["box"][2]),
    ]

    [[fx, _, ppx], [_, fy, ppy], [_, _, _]] = camera_matrix

    z = float(np.mean(depth_image_projected_bb[depth_image_projected_bb != 0]))
    x: float = (x_image - ppx) * z / fx
    y: float = (y_image - ppy) * z / fy

    logger.info(f"Camera coordinates of {result['label']}: x={x:.3f}, y={y:.3f}, z={z:.3f} meters")

    return (x, y, z), result["label"]


def find_object(
    search_string: str, visualize_results: bool = False
) -> List[Tuple[Tuple[float, float, float], str]] | None:

    logger.info(f"Searching for '{search_string}'...")
    color_image, depth_image, depth_image_projected, camera_matrix = get_camera_frame_ros()
    image = PILImage.fromarray(color_image)

    inputs = object_detection_processor(images=image, text=search_string, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = object_detection_model(**inputs)

    # visualize_frame(color_image, depth_image)

    results: ObjectDetectionResult = object_detection_processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.4,
        text_threshold=0.3,
        target_sizes=[image.size[::-1]],
    )[0]

    logger.info(f"Object detection results: {results}")

    if visualize_results:
        result_image = draw_bounding_boxes(color_image, results)
        visualize_frame(result_image, depth_image)

    results_transformed: List[ObjectDetectionResultTransformed] = [
        {"score": score, "label": label, "box": box}
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"])
    ]

    results_processed = [
        process_single_bounding_box(result_transformed, depth_image_projected, camera_matrix)
        for result_transformed in results_transformed
    ]
    results_processed = list(filter(lambda x: x, results_processed))  # Filtering out None's
    if not results_processed:
        return None
    return results_processed


def get_sorted_matches(predictions: List[Tuple[Tuple[float, float, float], str]]) -> List[Tuple[str, float, int]]:

    processed_predictions = []
    for (x, _, z), label in predictions:
        angle = int(np.arctan2(x, z) * 180 / np.pi)
        distance = float(np.sqrt(x**2 + z**2))
        processed_predictions.append((label, distance, angle))

    return sorted(processed_predictions, key=lambda x: x[1])

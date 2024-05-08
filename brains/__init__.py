from typing import Literal

from loguru import logger
from pydantic_settings import BaseSettings


class Args(BaseSettings):
    """These values may be overriden by envs"""

    clip_duration: int = 5
    phrase_detection_cooldown: int = 3

    elevenlabs_voice_id: str = "pbjffAFCjozOIegLuBKn"

    openai_completions_model: str = "gpt-4-turbo-preview"
    openai_temperature: float = 0.0
    openai_max_function_calls: int = 3
    openai_max_tokens: int = 600

    openwakeword_detected_phrases_output_dir: str = "./data/phrases"
    openwakeword_model_path: str = "./models/openwakeword/hey_laiika.tflite"
    openwakeword_phrase_detection_threshold: float = 0.05
    openwakeword_vad_threshold: float = 0.0
    openwakeword_noise_suppression: bool = False
    openwakeword_inference_framework: Literal["onnx", "tflite"] = "tflite"

    object_detection_model_id: str = "IDEA-Research/grounding-dino-tiny"
    search_degrees_rotate: int = 45


args = Args()

logger.info(args)

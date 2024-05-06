import io
import os
import time
from collections import defaultdict
from datetime import datetime

import numpy as np
import openwakeword
import pyaudio
from loguru import logger
from openwakeword.model import Model
from openwakeword.utils import download_models
from scipy.io import wavfile

from brains import args
from brains.brain import submit_request
from brains.camera import find_object
from brains.control import collect_leaves
from brains.utils import get_transcription, play_text

CHUNK_SIZE = 1280
RATE = 16000

download_models()

async def active_listening_loop():

    # Get microphone stream
    audio = pyaudio.PyAudio()
    mic_stream = audio.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK_SIZE,
    )

    # Load pre-trained openwakeword models
    if args.openwakeword_model_path:
        owwModel = Model(
            wakeword_models=[args.openwakeword_model_path],
            enable_speex_noise_suppression=args.openwakeword_noise_suppression,
            vad_threshold=args.openwakeword_vad_threshold,
            inference_framework=args.openwakeword_inference_framework,
        )
    else:
        owwModel = Model(
            enable_speex_noise_suppression=args.openwakeword_noise_suppression,
            vad_threshold=args.openwakeword_vad_threshold,
            inference_framework=args.openwakeword_inference_framework,
        )

    os.makedirs(args.openwakeword_detected_phrases_output_dir, exist_ok=True)

    # Predict continuously on audio stream
    last_save = time.time()
    activation_times = defaultdict(list)
    last_time_activated = time.time()

    logger.info("Listening for wakewords...")
    while True:
        mic_audio = np.frombuffer(mic_stream.read(CHUNK_SIZE), dtype=np.int16)

        prediction = owwModel.predict(mic_audio)
        if type(prediction) != dict:
            continue

        for phrase, confidence in prediction.items():
            if confidence >= args.openwakeword_phrase_detection_threshold and time.time() - last_time_activated >= 1:
                logger.debug(f"Heard activation phrase")
                play_text("What's up buddy?")
                last_time_activated = time.time()
                activation_times[phrase].append(time.time())

            if (
                activation_times.get(phrase)
                and (time.time() - last_save) >= args.phrase_detection_cooldown
                and (time.time() - activation_times[phrase][0]) >= args.clip_duration - 1
            ):
                last_save = time.time()
                activation_times[phrase] = []
                detect_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
                logger.debug(f"Finished recording context of activation phase '{phrase}'")

                audio = np.array(list(owwModel.preprocessor.raw_data_buffer)[-RATE * args.clip_duration :]).astype(
                    np.int16
                )

                audio_buffer = io.BytesIO()
                audio_buffer.name = "audio.wav"
                wavfile.write(
                    filename=os.path.join(
                        os.path.abspath(args.openwakeword_detected_phrases_output_dir), detect_time + f"_{phrase}.wav"
                    ),
                    rate=RATE,
                    data=audio,
                )
                wavfile.write(filename=audio_buffer, rate=RATE, data=audio)

                text = get_transcription(audio_buffer)
                await submit_request(text)


import asyncio

if __name__ == "__main__":

    # Download preprocessing models in case
    # they were not already loaded

    # download_models()
    # active_listening_loop()

    # result = None
    # while not result:
    #     result = find_object(search_string=args.object_search_string)
    # print(result)

    asyncio.run(active_listening_loop())

    # asyncio.run(collect_leaves())

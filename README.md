# Brains

[![Build and push to registry](https://github.com/laikadogai/brains/actions/workflows/build.yaml/badge.svg)](https://github.com/laikadogai/brains/actions/workflows/build.yaml)

It is software that is made to be run on a [Laika the Dog](https://laikadog.ai/) — robodog for household chores. The robot listens for a wake phrase (e.g. "Hey Laika") and can understand natural commands through speech recognition and GPT-4, responding with a synthesized voice using ElevenLabs. Using an Intel RealSense camera and computer vision algorithms, it can locate objects in 3D space, approach them, and manipulate them using an Interbotix robotic arm.

<img src="https://github.com/user-attachments/assets/35051d70-e1af-47e7-ba6c-0ec24b491557" width="512">

https://github.com/user-attachments/assets/b5f2642e-a4f5-4e42-b35e-03732e672a9c

## Prerequisites

Prerequisites consist of hardware and packages that should be installed on this hardware.

### Hardware

- Quadruped: [Unitree Go2](https://www.unitree.com/go2), cheapest version can work if you will [jailbreak](https://wiki.theroboverse.com/) it
- Arm: any [Trossen Robotics arm](https://www.trossenrobotics.com/robotic-arms), we had [WidowX 250 S](https://www.trossenrobotics.com/widowx-250)
- Camera: Intel RealSense depth camera, in our case, it is [D435i](https://www.intelrealsense.com/depth-camera-d435i/)
- Computer (if your version of Go2 is not EDU then you need one): some NVIDIA device, we have [Jetson Orin Nano](https://developer.nvidia.com/embedded/learn/get-started-jetson-orin-nano-devkit)
- Speaker: we used JBL lol
- Microphone
- DC/DC converter
- Optionally you can print the [body which we designed](./stl) to glue everything nicely

### Software

You have two options: work within Docker or install dependencies on bare metal.

#### Docker

Build image:

```bash
sudo docker buildx build --tag brains .
```

Run it:

```bash
sudo docker run -it --tty --name brains --rm \
    --env OPENAI_API_KEY=${OPENAI_API_KEY} \
    --env ELEVEN_API_KEY=${ELEVEN_API_KEY} \
    brains
```

You will enter tmux environment with everything configured. You might have to forward devices inside the container so they can be seen from the inside.

#### Bare Metal

- [ROS 2](https://docs.ros.org/en/humble/index.html), we used Humble
- [ROS 2 package for the arm](https://github.com/Interbotix/interbotix_ros_manipulators)
- [ROS 2 package for the camera](https://github.com/IntelRealSense/realsense-ros)
- Optionally you can train your own wake word model using [openWakeWord](https://github.com/dscripka/openWakeWord?tab=readme-ov-file#training-new-models)
- [Poetry](https://python-poetry.org/docs/#installation)
- This package (`poetry install`)

## How to Run

Launch arm and camera ROS nodes:

```bash
ros2 launch interbotix_xsarm_control xsarm_control.launch.py robot_model:=wx250
ros2 launch realsense2_camera rs_launch.py
```

Make sure you have `OPENAI_API_KEY` and `ELEVEN_API_KEY` environment variables, and run the service via:

```bash
poetry run python app.py
```

It is possible to control script parameters via environment variables listed in [Args class](./brains/__init__.py), for example:

```bash
openai_temperature=0.7 poetry run python app.py
```

<!---

## ToDo

- Make service work in Docker by figuring out hardware forwarding
- Listen until no sound and get rid of the hardcoded window
- Write messaging state to local storage to preserve between restarts
- When within an ongoing conversation thread, don't wait for the "activation phrase"
- Remove "Hey, robot" from messages history to avoid clutter (maybe)
- Add some kind of cron jobs, e.g. rescan the environment

## Miscellaneous

### Local Modules vs Online APIs

It is possible to replase ASR with Whisper and TTS with the best open model from [TTS Arena](https://huggingface.co/spaces/TTS-AGI/TTS-Arena).

-->

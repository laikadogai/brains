FROM osrf/ros:humble-desktop

RUN apt update && apt upgrade -y && apt install iproute2 iputils-ping portaudio19-dev python3-pyaudio -y

# Setting up DotFiles and useful programs
WORKDIR /root
RUN git clone https://github.com/furiousteabag/DotFiles.git && cd DotFiles && ./setup.sh debian

# Creating colcon workspace
ENV COLCON_WORKSPACE /root/colcon-workspace
RUN mkdir -p $COLCON_WORKSPACE/src
WORKDIR $COLCON_WORKSPACE

# Installing arm packages
# https://docs.trossenrobotics.com/interbotix_xsarms_docs/ros_interface/ros2/software_setup.html
RUN curl 'https://raw.githubusercontent.com/Interbotix/interbotix_ros_manipulators/main/interbotix_ros_xsarms/install/amd64/xsarm_amd64_install.sh' > xsarm_amd64_install.sh && \
    chmod +x xsarm_amd64_install.sh && \
    ./xsarm_amd64_install.sh -d $ROS_DISTRO -p $COLCON_WORKSPACE -n

# Official Unitree ROS2 bindings
RUN apt install ros-$ROS_DISTRO-rmw-cyclonedds-cpp ros-$ROS_DISTRO-rosidl-generator-dds-idl -y
RUN git clone https://github.com/unitreerobotics/unitree_ros2 src/unitree_ros2 && \
    git clone https://github.com/ros2/rmw_cyclonedds src/rmw_cyclonedds -b $ROS_DISTRO && \
    git clone https://github.com/eclipse-cyclonedds/cyclonedds src/cyclonedds -b releases/0.10.x
RUN colcon build --packages-select cyclonedds
ENV RMW_IMPLEMENTATION rmw_cyclonedds_cpp

# Unofficial Unitree Go2 ROS2 SDK
RUN apt install python3-pip clang -y
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
RUN git clone --recurse-submodules https://github.com/abizovnuralem/go2_ros2_sdk.git src/go2_ros2_sdk && pip install -r src/go2_ros2_sdk/requirements.txt
# Remove the official 'unitree_go' package folder (official and unofficial packages have the same conflicting name 'unitree_go')
RUN rm -rf src/unitree_ros2/cyclonedds_ws/src/unitree/unitree_go
RUN rosdep install --from-paths src --ignore-src -r -y
RUN . /root/.cargo/env && . /opt/ros/$ROS_DISTRO/setup.sh && colcon build --packages-ignore go2_demo

# Realsense
# https://github.com/IntelRealSense/realsense-ros?tab=readme-ov-file#installation-on-ubuntu
RUN mkdir -p /etc/apt/keyrings && curl -sSf https://librealsense.intel.com/Debian/librealsense.pgp | sudo tee /etc/apt/keyrings/librealsense.pgp > /dev/null
RUN apt install apt-transport-https -y
RUN echo "deb [signed-by=/etc/apt/keyrings/librealsense.pgp] https://librealsense.intel.com/Debian/apt-repo `lsb_release -cs` main" | tee /etc/apt/sources.list.d/librealsense.list
# librealsense2-dkms
RUN apt update && apt install librealsense2-utils librealsense2-dev librealsense2-dbg "ros-humble-realsense2-*" -y

# Go2 python SDK
# export CMAKE_PREFIX_PATH=/home/jetson/colcon-workspace/install/cyclonedds:$CMAKE_PREFIX_PATH
WORKDIR /root
RUN git clone https://github.com/legion1581/go2_python_sdk.git && cd go2_python_sdk && pip install -r requirements.txt

# Poetry
ENV POETRY_VERSION=1.8.2
ENV POETRY_HOME=/opt/poetry
ENV POETRY_VENV=/opt/poetry-venv
ENV POETRY_CACHE_DIR=/opt/.cache
RUN apt install python3-venv portaudio19-dev -y && python3 -m venv $POETRY_VENV \
	&& $POETRY_VENV/bin/pip install -U pip setuptools \
	&& $POETRY_VENV/bin/pip install poetry==${POETRY_VERSION}
ENV PATH="${PATH}:${POETRY_VENV}/bin"

# Project
WORKDIR /app
COPY poetry.lock pyproject.toml ./
RUN poetry install
# Preload and cache openwakeword preprocessing models
RUN poetry run python -c "import openwakeword; openwakeword.utils.download_models()"
COPY . /app

# Adding interactive setup scripts for completions and colcon_cd.
# If starting bash interactive shell change those accordingly.
# https://answers.ros.org/question/394021/no-completion-for-ros2-run-in-zsh/
RUN ZSHRC_PATH=$( [ -d "/root/.config/zsh" ] && echo "/root/.config/zsh/.zshrc" || echo "/root/.zshrc" ) && \
    echo "source /opt/ros/$ROS_DISTRO/setup.zsh" >> $ZSHRC_PATH && \
    echo "source $COLCON_WORKSPACE/install/setup.zsh" >> $ZSHRC_PATH && \
    echo "source /usr/share/colcon_argcomplete/hook/colcon-argcomplete.zsh" >> $ZSHRC_PATH && \
    echo 'eval "$(register-python-argcomplete3 ros2)"' >> $ZSHRC_PATH && \
    echo "source /usr/share/colcon_cd/function/colcon_cd.sh" >> $ZSHRC_PATH && \
    echo "source /usr/share/gazebo/setup.sh" >> $ZSHRC_PATH && \
    echo 'ulimit -n 1024' >> $ZSHRC_PATH # https://github.com/ros2/ros2/issues/1531

ENV ROS_DOMAIN_ID 0

CMD ["zsh", "--login"]
# CMD [ "poetry", "run", "python", "app.py" ]

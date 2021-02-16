FROM arm64v8/ros:noetic

WORKDIR /root/ws/
COPY . .

# INSTALL ROS PACKAGES
RUN apt-get update && apt-get install -y \
  ros-${ROS_DISTRO}-ros-tutorials \
  ros-${ROS_DISTRO}-common-tutorials \
  python3-pip && \
  rm -rf /var/lib/apt/lists/*

# INSTALL PYTHON PACKAGES
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install -r requirements.txt

# EXTEND BASHRC
RUN echo "source /root/ws/devel_isolated/setup.bash" >> ~/.bashrc

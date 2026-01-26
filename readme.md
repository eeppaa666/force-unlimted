# 启动teleop server
python3 src/teleop/src/teleop_server.py --pub_frequency=60
## 启动 teleop sub查看消息
python3 src/teleop/test/test_teleop_sub.py

# 启动ik
python3 src/ik/src/ik_node.py --frequency=60

# 启动fk
python3 src/fk/src/fk_node.py --frequency=60

# 启动controller
export CYCLONEDDS_HOME=/root/code/cyclonedds-0.10.2/cyclonedds_install

## 启动 unitree controller
conda activate controller
python3 src/controller/src/unitree_g1_29_controller.py --unitree_dds_fps 60 --ros_msg_fps 60  --open_img_pub True --image_fps 10 --img_server_ip 10.106.1.95

## 启动 fourier controller
conda activate fourier
python3 src/controller/src/fourier_gr1_t1_controller.py

# 启动 foxglove 可视化
python3 src/foxglove/src/server.py

# docker
**镜像地址**：

\<your code path\> 是宿主机代码绝对路径
## 启动 infra ros docker
docker run -it --rm --name ros_container --net=host --privileged -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v /dev:/dev -v <your code path>:/root/workspace  infra_ros:v1.0 /bin/bash

## 启动 fourier ros docker
docker run -it --rm --name ros_container --net=host --privileged -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v /dev:/dev -v <your code path>:/root/workspace  fourier_ros:v1.0 /bin/bash

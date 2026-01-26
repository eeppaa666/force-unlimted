# 摇操手册

# 架构

所有节点基于ros2 进行通信，使用PICO等VR设备进行摇操

![framework.jpeg](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/a2QnV4j3kWrBoO4X/img/a4aa5359-f446-4423-9e5b-abe0ad2713f2.jpeg)

# 环境准备

**代码仓库地址**：

**infra docker 镜像：**

**镜像启动命令：**

<your code path>是宿主机代码绝对路径，需要用户根据自己情况修改

```shell
docker run -it --rm --name infra_ros --net=host --privileged -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v /dev:/dev -v <your code path>:/root/workspace  -w /root/workspace infra_ros:v1.0 /bin/bash

```

# 节点启动

启动节点之前需要进入code 根目录，所有节点都依赖docker 镜像环境启动

```shell
# 进入docker 容器
docekr exec -it infra_ros /bin/bash

# 激活conda env环境
conda activate infra

# 进入代码工作目录
cd /root/workspace/force-unlimited
```

## teleop节点

负责采集各个VR设备的摇操数据（hand, controller,  head 等等），然后通过ros2 发送到总线上

**启动命令**：

```shell
# 参数介绍
# --use_hand_track 是否开启手势跟踪
# --pub_frequency 发送摇操数据的HZ,默认30HZ

python3 src/teleop/src/teleop_server.py
```

## IK节点

负责将摇操跟踪的数据，通过IK转换为机器人本体的qpos数据

**启动命令**：

```shell
# --robot robot类型 目前支持 unitree_g1_29、fourier_gr1t1

python3 src/ik/src/ik_node.py --frequency=60

```

## FK节点

负责将IK 或 机器人本体的state 信号通过FK输出 frametransform

**启动命令：**

```shell
# --robot robot类型 目前支持 unitree_g1_29、fourier_gr1t1
# --use_ik_sol 是否使用IK输出的qpos，否则使用本体/sim state返回状态

python3 src/fk/src/fk_node.py --frequency=60

```

## foxglove可视化节点

负责观测机器人3d实时pose，以及查看总线上各个topic的raw message。启动后使用客户端连接 ws://<ip>:8765

[请至钉钉文档查看附件《飞书20260124-170308.mp4》](https://alidocs.dingtalk.com/i/nodes/1DKw2zgV2PzQBZ01SD99YwYM8B5r9YAn?iframeQuery=anchorId%3DX02mks4zpazi72gr3ji0o)

**启动命令：**

```shell
python3 src/foxglove/src/server.py
```

## controller节点

通过接受IK节点的qpos来控制不同的机器人本体，不同的本体有自己不同的controller

### fourier controller

**docker 镜像地址:**

**启动命令：**

```shell
python3 src/controller/src/unitree_g1_29_controller.py --unitree_dds_fps 60 --ros_msg_fps 60

```

### unitree controller

**docker 镜像地址:**

**启动命令：**

```shell
python3 src/controller/src/fourier_gr1_t1_controller.py
```
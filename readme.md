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
python3 src/controller/src/unitree_g1_29_controller.py --unitree_dds_fps 60 --ros_msg_fps 60  --open_img_pub True --image_fps 10 --img_server_ip 10.106.1.95

# 启动 fourier controller
python3 src/controller/src/fourier_gr1_t1_controller.py

# 启动 foxglove 可视化
python3 src/foxglove/src/server.py

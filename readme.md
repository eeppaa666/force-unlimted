# 启动teleop server
python3 src/teleop/src/teleop_server.py --pub_frequency=60
## 启动 teleop sub查看消息
python3 src/teleop/test/test_teleop_sub.py

# 启动ik
python3 src/ik/src/ik_node.py --frequency=60

# 启动fk
python3 src/ik/src/fk_node.py --frequency=60

# 启动controller
python3 src/controller/src/unitree_g1_29_controller.py --unitree_dds_fps 60 --ros_msg_fps 60  --open_img_pub True

# 启动 foxglove 可视化
python3 src/foxglove/src/server.py

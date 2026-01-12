# 启动ik
python3 src/ik/src/ik_node.py --frequency=60

# 启动teleop server
python3 src/teleop/src/teleop_server.py --pub_frequency=60
python3 src/teleop/test/test_teleop_sub.py

# 启动controller
python3 src/controller/src/unitree_g1_29_controller.py --unitree_dds_fps 60 --ros_msg_fps 60

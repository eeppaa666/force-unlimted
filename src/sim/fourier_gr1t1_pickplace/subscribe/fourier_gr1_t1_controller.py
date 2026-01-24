"""
FOURIER G1-29 双臂控制桥接节点
------------------------------------------------------
功能：
1. 通过 FOURIER DDS SDK 控制真实/仿真 G1 机器人双臂
2. 订阅 ROS2 上位机发送的 IK 解算结果 (protobuf over UInt8MultiArray)
3. 周期性发布机器人当前双臂状态 (protobuf over UInt8MultiArray)
4. 可选：获取头部相机图像并发布到 ROS2
5. 内部启动 DDS 控制线程 + ROS2 spin 线程

数据流：
ROS2 IK_SOL_TOPIC  --->  本节点  ---> FOURIER DDS ArmController
FOURIER DDS State  --->  本节点  ---> ROS2 LOW_STATE_TOPIC
Teleimager Head Camera ---> 本节点 ---> ROS2 HEAD_FRAME_TOPIC
"""

import os
import sys
import time
import logging
import threading
from functools import partial
from array import array

# =========================
# 路径设置：确保 protobuf 生成文件可被导入
# =========================
this_file = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(os.path.dirname(this_file), '../../../'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, '../proto/generate'))

# =========================
# 引入 Protobuf 消息
# =========================
from controller.state_pb2 import UnitTreeLowState
from ik.ik_sol_pb2 import IKSol
from teleop.tele_pose_pb2 import TeleState
from image.image_pb2 import ImageFrame


# =========================
# 多进程日志与 ROS2
# =========================
from multiprocessing import Value, Array, Lock
from multiprocessing_logging import install_mp_handler
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, UInt8MultiArray

# =========================
# ROS2 Topic 定义
# =========================
# sub_topic
FOURIER_IK_SOL_TOPIC    = "/fourier/ik/sol"     # 上位机 IK 求解结果
TRACK_STATE_TOPIC       = '/teleop/track_state' # 遥操作/状态机控制信号
# pub topic
FOURIER_LOW_STATE_TOPIC = "/fourier/controller/low_state"  # 机器人双臂当前状态
FOURIER_HEAD_FRAME      = "/fourier/controller/head_frame"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(processName)s] %(filename)s:%(lineno)d - %(message)s' # 这里设置格式
)
install_mp_handler()


class FourierGR1T1Controller():
    def __init__(self, config):
        self._msg_count       = 0
        self._publisher_control = {}
        self._ik_sol      = IKSol()
        self._low_state   = UnitTreeLowState()
        temp_image_state = ImageFrame()
        self._image_msg   = UInt8MultiArray()
        self.__publisher_list_  = [FOURIER_LOW_STATE_TOPIC, FOURIER_HEAD_FRAME]
        self.__subscriber_list_ = [FOURIER_IK_SOL_TOPIC, TRACK_STATE_TOPIC]
        self._ros_msg_fps = 60

    def Start(self):
        self.__initRos()
        self._running = True
        logging.info(f"Start FOURIERG129Controller")

    def Stop(self):
        logging.info("Stopping FOURIERG129Controller...")
        # self._arm_ctrl.ctrl_dual_arm_go_home()
        if rclpy.ok():
            self.ros_node.get_logger().info("Stopping ROS spin...")
            rclpy.shutdown()
        if hasattr(self, "_ros_thread") and self._ros_thread.is_alive():
            self._ros_thread.join(timeout=2.0)
        if hasattr(self, "ros_node"):
            self.ros_node.destroy_node()
        self._running = False
        logging.info("FOURIERG129Controller stopped cleanly.")

    def IsRunnning(self):
        return self._running

    def GetIKMsg(self):
        return list(self._ik_sol)

    def __initRos(self):
        rclpy.init()
        self.ros_node = Node("FOURIER_g1_29_controller")
        for topic in self.__subscriber_list_:
            self.ros_node.create_subscription(UInt8MultiArray, topic, partial(self.__messageCallback, topic), 10)
            logging.info(f"create {topic} ros message sub")
        for topic in self.__publisher_list_:
            self._publisher_control[topic] = self.ros_node.create_publisher(UInt8MultiArray, topic, 10)
            logging.info(f"create {topic} ros message pub")
        self._ros_thread = threading.Thread(target=self.__rosSpin, daemon=True)
        self._ros_thread.start()

        # self._FOURIER_timer = self.ros_node.create_timer(1.0 / self._FOURIER_dds_fps, self.__FOURIERMsgPub)
        self._msg_timer = self.ros_node.create_timer(1.0 / self._ros_msg_fps, self.__rosMsgPub)

    def __messageCallback(self, topic_name, msg):
        self._msg_count += 1

        if topic_name is FOURIER_IK_SOL_TOPIC:
            state = IKSol()
            try:
                binary_data = bytes(msg.data)
                state.ParseFromString(binary_data)
                self._ik_sol.CopyFrom(state)
                if self._msg_count % 100 > 95:
                    logging.info(f"Get {FOURIER_IK_SOL_TOPIC} msg")
            except Exception as e:
                logging.error(f'ParseFromString Protobuf failed: {e}')

        if topic_name is TRACK_STATE_TOPIC:
            state = TeleState()
            try:
                binary_data = bytes(msg.data)
                state.ParseFromString(binary_data)
                if self._msg_count % 100 > 95:
                    logging.info(f"Get {TRACK_STATE_TOPIC} msg")
                # from google.protobuf import json_format
                # json_string = json_format.MessageToJson(state,
                #     always_print_fields_with_no_presence=True, # 强制包含默认值字段
                #     preserving_proto_field_name=True    # 建议同时开启，保持字段名和 .proto 一致
                # )
                # print(json_string)
                # ts = state.timestamp
                # ik_ms = ts.seconds * 1000000000 + ts.nanos
                # now_ms = time.time_ns()
                # delay_ms = now_ms - ik_ms
                # logging.info(f"Get {TRACK_STATE_TOPIC} msg, {ik_ms} ms, {delay_ms / 1000000} ms")
                # self._ik_sol.CopyFrom(state)
                # self.__FOURIERMsgPub()
            except Exception as e:
                logging.error(f'ParseFromString Protobuf failed: {e}')

    def __rosSpin(self):
        rclpy.spin(self.ros_node)

    def __rosMsgPub(self):
        pass
        # msg = UInt8MultiArray()
        # current_lr_arm_q   = self._arm_ctrl.get_current_dual_arm_q()
        # current_lr_arm_dq  = self._arm_ctrl.get_current_dual_arm_dq()
        # current_lr_motor_q = self._arm_ctrl.get_current_motor_q()

        # try:
        #     timestamp = time.time_ns()
        #     self._low_state.timestamp.seconds = timestamp // 1_000_000_000
        #     self._low_state.timestamp.nanos = timestamp % 1_000_000_000
        #     self._low_state.dual_arm_q.extend(current_lr_arm_q)
        #     self._low_state.dual_arm_dq.extend(current_lr_arm_dq)
        #     self._low_state.motor_q.extend(current_lr_motor_q)
        #     binary_data = self._low_state.SerializeToString()
        #     msg.data = list(binary_data)
        #     self._publisher_control[FOURIER_LOW_STATE_TOPIC].publish(msg)
        # except Exception as e:
        #     logging.error(f'SerializeToString Protobuf failed: {e}')
        # self._low_state.Clear()
        # logging.info(f"{FOURIER_LOW_STATE_TOPIC} Msg pub")


def main():
    controller = FourierGR1T1Controller("")
    controller.Start()

    while(controller.IsRunnning()):
        time.sleep(5)
        logging.info(f"doning...")

if __name__ == "__main__":
    main()



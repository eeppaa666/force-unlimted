"""
Unitree G1-29 双臂控制桥接节点
------------------------------------------------------
功能：
1. 通过 Unitree DDS SDK 控制真实/仿真 G1 机器人双臂
2. 订阅 ROS2 上位机发送的 IK 解算结果 (protobuf over UInt8MultiArray)
3. 周期性发布机器人当前双臂状态 (protobuf over UInt8MultiArray)
4. 内部启动 DDS 控制线程 + ROS2 spin 线程

整体架构:
ROS2 IK_SOL_TOPIC  --->  本节点  ---> Unitree DDS ArmController
Unitree DDS State  --->  本节点  ---> ROS2 LOW_STATE_TOPIC
"""

import os
import sys
import time
import logging
import threading
from functools import partial

# =========================
# 路径设置：确保 protobuf 生成文件可被导入
# =========================
this_file = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(os.path.dirname(this_file), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, '../proto/generate'))
from controller.state_pb2 import UnitTreeLowState
from ik.ik_sol_pb2 import UnitTreeIkSol
from teleop.tele_pose_pb2 import TeleState

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize # dds
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import ( LowCmd_  as hg_LowCmd, LowState_ as hg_LowState) # idl for g1, h1_2
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from thirdparty_sdk.unitree.robot_arm import G1_29_ArmController, G1_23_ArmController, H1_2_ArmController, H1_ArmController
from thirdparty_sdk.unitree.robot_hand_unitree import Dex1_1_Gripper_Controller

from multiprocessing import Value, Array, Lock
from multiprocessing_logging import install_mp_handler
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, UInt8MultiArray


from base.controller_interface import ControllerInterface
from base.utils import BuildArgParser, RegisterShutDownHook

# from loguru import logger
# logger.debug(f"asdfasd {self.subscriber_list_}")
# logger.info(f"asdfasd {self.subscriber_list_}")
# logger.warning(f"asdfasd {self.subscriber_list_}")
# logger.error(f"asdfasd {self.subscriber_list_}")

# =========================
# ROS2 Topic 定义
# =========================
# sub_topic
UNITREE_IK_SOL_TOPIC    = "/unitree/ik_sol"     # 上位机 IK 求解结果
TRACK_STATE_TOPIC       = '/teleop/track_state' # 遥操作/状态机控制信号
# pub topic
UNITREE_LOW_STATE_TOPIC = "/unitree/low_state"  # 机器人双臂当前状态

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(processName)s] %(filename)s:%(lineno)d - %(message)s' # 这里设置格式
)
install_mp_handler()


class UnitreeG129Controller(ControllerInterface):
    def __init__(self, config):
        super().__init__(config)

        self._msg_count       = 0
        self._publisher_control = {}
        self._ik_sol      = UnitTreeIkSol()
        self._low_state   = UnitTreeLowState()

        self._motion_mode     = config.get("motion_mode", bool)
        self._simulation_mode = config.get("simulation_mode", bool)
        self._unitree_dds_fps = config.get("unitree_dds_fps", int)
        self._ros_msg_fps     = config.get("ros_msg_fps", int)

    def Start(self):
        self.__initUnitree()
        self.__initRos()
        self._running = True
        logging.info(f"Start UnitreeG129Controller")

    def Stop(self):
        logging.info("Stopping UnitreeG129Controller...")
        # self._arm_ctrl.ctrl_dual_arm_go_home()
        if rclpy.ok():
            self.ros_node.get_logger().info("Stopping ROS spin...")
            rclpy.shutdown()
        if hasattr(self, "_ros_thread") and self._ros_thread.is_alive():
            self._ros_thread.join(timeout=2.0)
        if hasattr(self, "ros_node"):
            self.ros_node.destroy_node()
        self._running = False
        logging.info("UnitreeG129Controller stopped cleanly.")

    def IsRunnning(self):
        return self._running

    def __initUnitree(self):
        if self._simulation_mode:
            ChannelFactoryInitialize(1)
        else:
            ChannelFactoryInitialize(0)
        # 接口	                             类型	    作用
        # __init__	                        构造函数	初始化 DDS、线程、PID 参数、控制消息
        # ctrl_dual_arm(q, tau)	            外部控制	设置双臂目标关节角度/力矩
        # ctrl_dual_arm_go_home()	        外部控制	将双臂归零位
        # speed_gradual_max(t)	            外部控制	速度渐进到最大值
        # speed_instant_max()	            外部控制	速度立即最大
        # get_current_motor_q()	            状态查询	获取全身电机角度
        # get_current_dual_arm_q()	        状态查询	获取双臂角度
        # get_current_dual_arm_dq()	        状态查询	获取双臂速度
        # get_mode_machine()	            状态查询	获取当前 DDS 模式
        # _subscribe_motor_state()	        内部线程	持续读取 DDS 电机状态，更新缓存
        # _ctrl_motor_state()	            内部线程	循环发送控制命令到 DDS
        # clip_arm_q_target(target, vel)	内部工具	对目标角度做速度限制
        # _Is_weak_motor(id)	            内部工具	判断弱电机
        # _Is_wrist_motor(id)	            内部工具	判断手腕电机
        self._arm_ctrl = G1_29_ArmController(motion_mode=self._motion_mode, simulation_mode=self._simulation_mode)
        self._arm_ctrl.speed_gradual_max()

        self._left_gripper_value = Value('d', 0.0, lock=True)        # [input]
        self._right_gripper_value = Value('d', 0.0, lock=True)       # [input]
        self._dual_gripper_data_lock = Lock()
        self._dual_gripper_state_array = Array('d', 2, lock=False)   # current left, right gripper state(2) data.
        self._dual_gripper_action_array = Array('d', 2, lock=False)  # current left, right gripper action(2) data.
        self._gripper_ctrl = Dex1_1_Gripper_Controller(self._left_gripper_value, self._right_gripper_value, self._dual_gripper_data_lock, 
                                                     self._dual_gripper_state_array, self._dual_gripper_action_array, simulation_mode=self._simulation_mode)

    def __initRos(self):
        rclpy.init()
        self.ros_node = Node("unitree_g1_29_controller")
        for topic in self.subscriber_list_:
            self.ros_node.create_subscription(UInt8MultiArray, topic, partial(self.__messageCallback, topic), 10)
            logging.info(f"create {topic} ros message sub")
        for topic in self.publisher_list_:
            self._publisher_control[topic] = self.ros_node.create_publisher(UInt8MultiArray, topic, 10)
            logging.info(f"create {topic} ros message pub")
        self._ros_thread = threading.Thread(target=self.__rosSpin, daemon=True)
        self._ros_thread.start()

        # self._unitree_timer = self.ros_node.create_timer(1.0 / self._unitree_dds_fps, self.__unitreeMsgPub)
        self._msg_timer = self.ros_node.create_timer(1.0 / self._ros_msg_fps, self.__rosMsgPub)

    def __messageCallback(self, topic_name, msg):
        self._msg_count += 1

        if topic_name is UNITREE_IK_SOL_TOPIC:
            state = UnitTreeIkSol()
            try:
                binary_data = bytes(msg.data)
                state.ParseFromString(binary_data)
                self._arm_ctrl.ctrl_dual_arm(state.dual_arm_sol_q, state.dual_arm_sol_tauff)
                if self._msg_count % 100 > 95:
                    logging.info(f"Get {UNITREE_IK_SOL_TOPIC} msg")
                # ts = state.timestamp
                # ik_ms = ts.seconds * 1000000000 + ts.nanos
                # now_ms = time.time_ns()
                # delay_ms = now_ms - ik_ms
                # logging.info(f"Get {UNITREE_IK_SOL_TOPIC} msg, {ik_ms} ms, {delay_ms / 1000000} ms")
                # self._ik_sol.CopyFrom(state)
                # self.__unitreeMsgPub()
            except Exception as e:
                logging.error(f'ParseFromString Protobuf failed: {e}')

        if topic_name is TRACK_STATE_TOPIC:
            state = TeleState()
            try:
                binary_data = bytes(msg.data)
                state.ParseFromString(binary_data)
                left_ctrl_triggerValue  = 10.0 - state.left_ctrl_trigger_value * 10,
                right_ctrl_triggerValue = 10.0 - state.right_ctrl_trigger_value * 10,
                with self._left_gripper_value.get_lock():
                    self._left_gripper_value.value = left_ctrl_triggerValue[0]
                with self._right_gripper_value.get_lock():
                    self._right_gripper_value.value = right_ctrl_triggerValue[0]
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
                # self.__unitreeMsgPub()
            except Exception as e:
                logging.error(f'ParseFromString Protobuf failed: {e}')

    def __rosSpin(self):
        rclpy.spin(self.ros_node)

    # def __unitreeMsgPub(self):
    #     # get unitree msg and ros send
    #     if len(self._ik_sol.dual_arm_sol_q) >= 1:
    #         self._arm_ctrl.ctrl_dual_arm(self._ik_sol.dual_arm_sol_q, self._ik_sol.dual_arm_sol_tauff)
    #         ts = self._ik_sol.timestamp
    #         ik_ms = ts.seconds * 1000000000 + ts.nanos
    #         now_ms = time.time_ns()
    #         delay_ms = now_ms - ik_ms
    #         logging.info(f"Send Unitree msg, {ik_ms} ns, {delay_ms / 1000000} ms")
    #         # logging.info(f"Unitree Msg pub")

    def __rosMsgPub(self):
        msg = UInt8MultiArray()
        current_lr_arm_q  = self._arm_ctrl.get_current_dual_arm_q()
        current_lr_arm_dq = self._arm_ctrl.get_current_dual_arm_dq()

        try:
            timestamp = time.time_ns()
            self._low_state.timestamp.seconds = timestamp // 1_000_000_000
            self._low_state.timestamp.nanos = timestamp % 1_000_000_000
            self._low_state.dual_arm_q.extend(current_lr_arm_q)
            self._low_state.dual_arm_dq.extend(current_lr_arm_dq)
            binary_data = self._low_state.SerializeToString()
            msg.data = list(binary_data)
            self._publisher_control[UNITREE_LOW_STATE_TOPIC].publish(msg)
        except Exception as e:
            logging.error(f'SerializeToString Protobuf failed: {e}')
        self._low_state.Clear()
        # logging.info(f"{UNITREE_LOW_STATE_TOPIC} Msg pub")


def ExtraContrilerArgs(parser):
    parser.description = "Unitree G1 specific controller"
    parser.add_argument( "--motion_mode", type=bool, default=False, choices=["True", "False"], help="Unitree arm control motion_mode")
    parser.add_argument( "--simulation_mode", type=bool, default=True, choices=["True", "False"], help="Unitree arm control simulation_mode, control Unitree dds, True mean dimain_id 1 else 0")
    parser.add_argument( "--unitree_dds_fps", type=int, default=30, help="Unitree dds fps for simulation")
    parser.add_argument( "--ros_msg_fps", type=int, default=30, help="Ros msg fps for other subscribe")

    parser.set_defaults(rate_hz=30)
    parser.set_defaults(pub_topic_list=[UNITREE_LOW_STATE_TOPIC])
    parser.set_defaults(sub_topic_list=[UNITREE_IK_SOL_TOPIC, TRACK_STATE_TOPIC])


def main():
    parser = BuildArgParser(ExtraContrilerArgs)
    args = parser.parse_args()

    logging.info(args)
    controller = UnitreeG129Controller(vars(args))
    controller.Start()
    RegisterShutDownHook(controller.Stop)

    while(controller.IsRunnning()):
        time.sleep(5)
        logging.info(f"doning...")

if __name__ == "__main__":
    main()



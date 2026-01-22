import os
import sys
import time
import logging
import threading
import hydra
import numpy as np
from omegaconf import OmegaConf, DictConfig
from functools import partial
from array import array
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# =========================
# 路径设置：确保 protobuf 生成文件可被导入
# =========================
this_file = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(os.path.dirname(this_file), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, '../proto/generate'))

# =========================
# 引入 Protobuf 消息
# =========================
from controller.state_pb2 import UnitTreeLowState
from ik.ik_sol_pb2 import UnitTreeIkSol
from teleop.tele_pose_pb2 import TeleState
from image.image_pb2 import ImageFrame
from controller.teleoperation_pb2 import FourierTeleoperation

# =========================
# 多进程日志与 ROS2
# =========================
from multiprocessing import Value, Array, Lock
from multiprocessing_logging import install_mp_handler
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, UInt8MultiArray

# =========================
# 工具函数
# =========================
from base.controller_interface import ControllerInterface
from base.utils import BuildArgParser, RegisterShutDownHook, SaveImage, Str2Bool

# =========================
# fourier teleoperation package
# =========================
from teleoperation.adapter.hands import DummyDexHand, HandAdapter
from teleoperation.adapter.robots import DummyRobot, RobotAdapter
from teleoperation.retarget.robot import DexRobot
from teleoperation.upsampler import Upsampler


# =========================
# ROS2 Topic 定义
# =========================
# sub_topic
FOURIER_IK_SOL_TOPIC    = "/fourier/ik/sol"     # 上位机 IK 求解结果
TRACK_STATE_TOPIC       = '/teleop/track_state' # 遥操作/状态机控制信号
# pub topic
FOURIER_LOW_STATE_TOPIC = "/fourier/controller/low_state"  # 机器人双臂当前状态
FOURIER_HEAD_FRAME      = "/fourier/controller/head_frame"
FOURIER_DEBUG_VISIABLE  = "/fourier/controller/debug_visiable"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(processName)s] %(filename)s:%(lineno)d - %(message)s' # 这里设置格式
)
install_mp_handler()

class FourierGR1T1Controller(ControllerInterface):
    def __init__(self, config):
        super().__init__(config)
        self._cfg = self.__load_config(config.get("config_path", str), config.get("config_name", str))
        self._sim = self._cfg.sim
        self._dt = 1 / self._cfg.frequency

        self._msg_count       = 0
        self._publisher_control = {}
        self._ik_sol      = UnitTreeIkSol()
        self._low_state   = UnitTreeLowState()
        temp_image_state = ImageFrame()
        self._image_msg   = UInt8MultiArray()
        self._init_command_sent = False

        self._ros_msg_fps     = config.get("ros_msg_fps", int)
        self._open_img_pub    = config.get("open_img_pub", bool)
        self._image_encode    = config.get("img_encode", bool)
        self._image_fps       = config.get("image_fps", int)
        self._img_server_ip   = config.get("img_server_ip", str)

        self.joint_names = [
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_hip_pitch_joint",
            "left_knee_pitch_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_hip_pitch_joint",
            "right_knee_pitch_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
            "waist_yaw_joint",
            "waist_pitch_joint",
            "waist_roll_joint",
            "head_yaw_joint",
            "head_roll_joint",
            "head_pitch_joint",
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_pitch_joint",
            "left_wrist_yaw_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_pitch_joint",
            "right_wrist_yaw_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
        ]

        # --- 自动算索引 ---
        self.left_arm_joints = [
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_pitch_joint",
            "left_wrist_yaw_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
        ]

        self.right_arm_joints = [
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_pitch_joint",
            "right_wrist_yaw_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
        ]

        self.left_idx  = [self.joint_names.index(n) for n in self.left_arm_joints]
        self.right_idx = [self.joint_names.index(n) for n in self.right_arm_joints]


    def Start(self):
        self.__initFourier()
        self.__initRos()
        self._running = True
        logging.info(f"Start FourierGR1T1Controller")

    def Stop(self):
        logging.info("Stopping FourierGR1T1Controller...")
        self.end()
        if rclpy.ok():
            self.ros_node.get_logger().info("Stopping ROS spin...")
            rclpy.shutdown()
        if hasattr(self, "_ros_thread") and self._ros_thread.is_alive():
            self._ros_thread.join(timeout=2.0)
        if hasattr(self, "ros_node"):
            self.ros_node.destroy_node()
        self._running = False
        logging.info("FourierGR1T1Controller stopped cleanly.")

    def IsRunnning(self):
        return self._running

    def __load_config(self, config_path: str, config_name: str) -> DictConfig:
        """
        使用 Hydra 解析配置文件
        :param config_path: 配置目录
        :param config_name: 配置文件名
        :return: DictConfig 对象
        """
        # 确保路径是 Path 对象
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config path {config_path} does not exist")

        logging.info(f"Loading config from {config_path}/{config_name}.yaml")
        # 使用 Hydra 的 initialize_config_dir + compose 加载
        with hydra.initialize_config_dir(config_dir=str(config_path)):
            cfg = hydra.compose(config_name=config_name)
        logging.info(f"Config loaded:\n{OmegaConf.to_yaml(cfg)}")
        return cfg

    def __initFourier(self):
        # 双臂控制是通过 DexRobot + RobotAdapter/Upsampler
        # 夹爪 _fourier_left_hand _fourier_right_hand
        self._fourier_dex_reboot = DexRobot(self._cfg)
        # update joint positions in pinocchio（Player中DexRobot初始化后补充）
        self._fourier_dex_reboot.set_joint_positions(
            [self._fourier_dex_reboot.config.joint_names[i] for i in self._fourier_dex_reboot.config.controlled_joint_indices],
            self._fourier_dex_reboot.config.default_qpos,
            degrees=False,
        )
        self._fourier_dex_reboot.set_posture_target_from_current_configuration()
        if not self._sim:  # 真实机器人
            self._fourier_client: RobotAdapter = hydra.utils.instantiate(self._cfg.robot.instance)
            self._fourier_client.connect()
            self._fourier_upsampler = Upsampler(
                self._fourier_client,
                target_hz=self._cfg.upsampler.frequency,
                dimension=self._cfg.robot.num_joints,
                initial_command=self._fourier_client.joint_positions,
                gravity_compensation=self._cfg.upsampler.gravity_compensation,
            )
            self._fourier_left_hand: HandAdapter = hydra.utils.instantiate(self._cfg.hand.left_hand)
            self._fourier_right_hand: HandAdapter = hydra.utils.instantiate(self._cfg.hand.right_hand)
        else:  # 仿真
            self._fourier_client: RobotAdapter = DummyRobot(self._cfg.robot.num_joints)
            self._fourier_upsampler = Upsampler(self._fourier_client, dimension=self._cfg.robot.num_joints, target_hz=self._cfg.upsampler.frequency)
            hand_dimension = self._cfg.hand.left_hand.get("dimension", 6)
            self._fourier_left_hand: HandAdapter = DummyDexHand(hand_dimension)
            self._fourier_right_hand: HandAdapter = DummyDexHand(hand_dimension)
        self.init_control_joints()

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
        self._msg_timer = self.ros_node.create_timer(1.0 / self._ros_msg_fps, self.__rosMsgPub)

    def __messageCallback(self, topic_name, msg):
        self._msg_count += 1

        if topic_name is FOURIER_IK_SOL_TOPIC:
            state = UnitTreeIkSol()
            debug_visiable_state = FourierTeleoperation()
            try:
                binary_data = bytes(msg.data)
                state.ParseFromString(binary_data)
                dual_q = np.array(state.dual_arm_sol_q, dtype=np.float64, copy=True)
                q_cur  = np.array(self._fourier_dex_reboot.q_real, dtype=np.float64, copy=True)
                q_cmd = self.buildQCmd(dual_q, q_cur)

                # debug visiable
                native = debug_visiable_state.native_pos
                native.q_real.extend(self._fourier_dex_reboot.q_real.flatten().tolist())

                self._fourier_dex_reboot.q_real = q_cmd

                # debug visiable
                solve = debug_visiable_state.solve_pos
                solve.q_real.extend(self._fourier_dex_reboot.q_real.flatten().tolist())
                qpos = self.control_joints()

                filter = debug_visiable_state.filter_pos
                filter.filter_qpos.extend(qpos.flatten().tolist())
                filter.q_real.extend(self._fourier_dex_reboot.q_real.flatten().tolist())

                if self._msg_count % 100 > 95:
                    logging.info(f"Get {FOURIER_IK_SOL_TOPIC} msg")

                timestamp = time.time_ns()
                self._low_state.timestamp.seconds = timestamp // 1_000_000_000
                self._low_state.timestamp.nanos = timestamp % 1_000_000_000
                binary_data = debug_visiable_state.SerializeToString()
                msg = UInt8MultiArray()
                msg.data = list(binary_data)
                self._publisher_control[FOURIER_DEBUG_VISIABLE].publish(msg)
                # logging.info(f"pub {FOURIER_DEBUG_VISIABLE} msg")
            except Exception as e:
                logging.error(f'ParseFromString Protobuf failed: {e}')

        if topic_name is TRACK_STATE_TOPIC:
            state = TeleState()
            try:
                binary_data = bytes(msg.data)
                state.ParseFromString(binary_data)
                if self._msg_count % 100 > 95:
                    logging.info(f"Get {TRACK_STATE_TOPIC} msg")

                if state.start_track:
                    from fourier.utils import LEFT_HAND_TRIGGER_QPOS, RIGHT_HAND_TRIGGER_QPOS
                    if state.left_ctrl_trigger_value >= 0:
                        scale = state.left_ctrl_trigger_value if state.left_ctrl_trigger_value > 0 else 0.1
                        left = np.array(LEFT_HAND_TRIGGER_QPOS)
                        left[:8] = left[:8] *  (abs(scale) if abs(scale) <= 1.0 else 0)

                        left, _ = self._fourier_dex_reboot.hand_action_convert(left, np.array(RIGHT_HAND_TRIGGER_QPOS))
                        self._fourier_left_hand.set_positions(left)

                    if state.right_ctrl_trigger_value >= 0:
                        right = np.array(RIGHT_HAND_TRIGGER_QPOS)
                        right[:8] = right[:8] *  (abs(state.right_ctrl_trigger_value) if abs(state.right_ctrl_trigger_value) <= 1.0 else 0)

                        _, right = self._fourier_dex_reboot.hand_action_convert(np.array(LEFT_HAND_TRIGGER_QPOS), right)
                        # self._fourier_right_hand.set_positions(right)

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

    def __rosMsgPub(self):
        msg = UInt8MultiArray()
        # current_lr_arm_q   = self._arm_ctrl.get_current_dual_arm_q()
        # current_lr_arm_dq  = self._arm_ctrl.get_current_dual_arm_dq()
        # current_lr_motor_q = self._arm_ctrl.get_current_motor_q()

        try:
            timestamp = time.time_ns()
            self._low_state.timestamp.seconds = timestamp // 1_000_000_000
            self._low_state.timestamp.nanos = timestamp % 1_000_000_000
            # self._low_state.dual_arm_q.extend(current_lr_arm_q)
            # self._low_state.dual_arm_dq.extend(current_lr_arm_dq)
            # self._low_state.motor_q.extend(current_lr_motor_q)
            # binary_data = self._low_state.SerializeToString()
            # msg.data = list(binary_data)
            # self._publisher_control[FOURIER_LOW_STATE_TOPIC].publish(msg)
        except Exception as e:
            logging.error(f'SerializeToString Protobuf failed: {e}')
        self._low_state.Clear()
        # logging.info(f"{FOURIER_LOW_STATE_TOPIC} Msg pub")


    def __ImageMsgPub(self):
        img, fps  = self._img_client.get_head_frame()

        if img is not None:
            temp_image_state = ImageFrame()
            timestamp = time.time_ns()
            temp_image_state.timestamp.seconds = timestamp // 1_000_000_000
            temp_image_state.timestamp.nanos = timestamp % 1_000_000_000

            temp_image_state.frame_id = FOURIER_HEAD_FRAME
            temp_image_state.width    = img.shape[1]
            temp_image_state.height   = img.shape[0]
            temp_image_state.channels = img.shape[2]

            if not self._image_encode:
                temp_image_state.pixel_format = ImageFrame.BGR8
                temp_image_state.encoding = ImageFrame.ENCODING_H265
                temp_image_state.data = img.tobytes()
            else:
                temp_image_state.pixel_format = ImageFrame.BGR8
                temp_image_state.encoding     = ImageFrame.ENCODING_JPEG
                temp_image_state.data = img.tobytes()

            temp_image_state.fps = fps
            temp_image_state.camera_model = "teleimager"

            try:
                self._image_msg.data = array('B', temp_image_state.SerializeToString())
                self._publisher_control[FOURIER_HEAD_FRAME].publish(self._image_msg)
            except Exception as e:
                logging.error(f'SerializeToString Protobuf failed: {e}')
            temp_image_state.Clear()
            # SaveImage(img)
        if self._msg_count % 100 > 90:
            logging.info(f"Pub {FOURIER_HEAD_FRAME} msg")

    def control_joints(self):
        qpos = self._fourier_dex_reboot.joint_filter.next(time.time(), self._fourier_dex_reboot.q_real)
        self._fourier_upsampler.put(qpos)
        return qpos

    def init_control_joints(self):
        if self._init_command_sent:
            return

        self._fourier_upsampler.start()
        self._fourier_upsampler.put(self._fourier_dex_reboot.q_real)
        self._init_command_sent = True
        logging.info("Init command sent.")

    def pause_robot(self):
        logging.info("Pausing robot...")
        self._fourier_upsampler.pause()
        # self.client.move_joints(ControlGroup.ALL, self.client.joint_positions, gravity_compensation=False)

    def end(self):
        self._fourier_upsampler.stop()
        self._fourier_upsampler.join()
        self.client.disconnect()
        with ThreadPoolExecutor(max_workers=2) as executor:
            executor.submit(self.left_hand.reset)
            executor.submit(self.left_hand.reset)

    def control_hands(self, left_qpos: np.ndarray, right_qpos: np.ndarray, return_real=False):
        """Control real hands

        Args:
            left_qpos (np.ndarray): Hand qpos in radians
            right_qpos (np.ndarray): Hand qpos in radians

        Returns:
            np.ndarray: concatenated and filtered hand qpos in real steps
        """
        left, right = self.hand_action_convert(left_qpos, right_qpos, filtering=True)
        self.left_hand.set_positions(left)  # type: ignore
        self.right_hand.set_positions(right)  # type: ignore

        if return_real:
            return np.hstack([left, right])
        else:
            actuated_indices = self.hand_retarget.cfg.actuated_indices
            return np.hstack([left_qpos[actuated_indices], right_qpos[actuated_indices]])

    def buildQCmd(self, dual_arm_q, q_current):
        """
        dual_arm_q: np.ndarray shape (14,)
        q_current:  np.ndarray shape (32,)  当前整机关节状态
        return:     np.ndarray shape (32,)  新指令
        """
        q_cmd = q_current.copy()

        q_cmd[self.left_idx]  = dual_arm_q[:7]
        q_cmd[self.right_idx] = dual_arm_q[7:]

        return q_cmd

def ExtraContrilerArgs(parser):
    parser.description = "Unitree G1 specific controller"
    parser.add_argument("--ros_msg_fps", type=int, default=30, help="Ros msg fps for other subscribe")
    parser.add_argument("--open_img_pub", type=Str2Bool, default=False, help="If open image pub")
    parser.add_argument("--image_encode", type=Str2Bool, default=False, help="Image_encode for H265")
    parser.add_argument("--image_fps", type=int, default=10, help="Image pub fps")
    parser.add_argument("--config_name", type=str, default="teleop_gr1_t1_sim", help="Hydra config name")

    parser.set_defaults(rate_hz=30)
    parser.set_defaults(pub_topic_list=[FOURIER_LOW_STATE_TOPIC, FOURIER_HEAD_FRAME, FOURIER_DEBUG_VISIABLE])
    parser.set_defaults(sub_topic_list=[FOURIER_IK_SOL_TOPIC, TRACK_STATE_TOPIC])
    parser.set_defaults(config_path=os.path.abspath(os.path.join(project_root, "controller/src/fourier/configs")))


def main():
    parser = BuildArgParser(ExtraContrilerArgs)
    args = parser.parse_args()

    logging.info(args)
    controller = FourierGR1T1Controller(vars(args))
    controller.Start()
    RegisterShutDownHook(controller.Stop)

    while(controller.IsRunnning()):
        time.sleep(5)
        logging.info(f"doning...")

if __name__ == "__main__":
    main()


# if you are using Apple Vision Pro:
# python -m teleoperation
#        --config-name teleop
#        robot=gr1t1_legacy
#        hand=fourier_dexpilot_dhx
#        camera=oak_97
#        robot.visualize=true
####     mocap=avp
#        sim=true

# OR if you are using Meta Quest 3
# python -m teleoperation
#        --config-name teleop
#        robot=gr1t1_legacy
#        hand=fourier_dexpilot_dhx
#        camera=oak_97
#        robot.visualize=true
####     mocap=quest
#        sim=true
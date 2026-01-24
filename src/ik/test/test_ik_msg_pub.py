import os, sys
this_file = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(os.path.dirname(this_file), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, '../proto/generate'))

# proto
from teleop.tele_pose_pb2 import TeleState
from controller.state_pb2 import UnitTreeLowState
from ik.ik_sol_pb2 import IKSol

import logging
import argparse
import threading
import numpy as np
# proto
from teleop.tele_pose_pb2 import TeleState
from controller.state_pb2 import UnitTreeLowState
from ik.ik_sol_pb2 import IKSol
from multiprocessing_logging import install_mp_handler
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(processName)s] %(filename)s:%(lineno)d - %(message)s' # 这里设置格式
)
install_mp_handler()

import rclpy
from rclpy.node import Node
from std_msgs.msg import UInt8MultiArray

from teleop.tele_pose_pb2 import TeleState
from teleop.src.common import TRACK_STATE_TOPIC
# from ik.src.ik_processor_base import IKProcessor

# from ik.src.ik_register import IK_PROCESSOR_MAP
# from ik.src.fourier.gr1t1_processor import Gr1T1Processor
IK_PROCESSOR_MAP={
    # "unitree_g1_29": G129IkProcessor,
    # "fourier_gr1t1": Gr1T1Processor,
}

FOURIER_IK_SOL_TOPIC = "/fourier/ik/sol"
FOURIER_LOW_STATE_TOPIC = "/fourier/controller/low_state"

class IkNode(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__('ik_node')
        logging.info(f"IK Node frequency set to: {args.frequency} Hz")
        self._subscription = self.create_subscription(
            UInt8MultiArray,
            TRACK_STATE_TOPIC,
            self.trackStateCallback,
            10)
        self._publisher = self.create_publisher(UInt8MultiArray, FOURIER_IK_SOL_TOPIC, 10)

        self._subscription  # prevent unused variable warning
        self._state_lock = threading.Lock()
        self._state = TeleState()
        self._state_flush_cnt: np.uint = 0
        self._state_prev_cnt: np.uint = 0

        self._ik_timer = self.create_timer(1.0 / args.frequency, self.ikProcessCallback)
        # from ik.src.unitree.g1_29_ik_processor import G129IkProcessor
        # self._ik_processor: IKProcessor = G129IkProcessor(self)
        # if args.robot not in IK_PROCESSOR_MAP:
            # raise ValueError(f"not support this robot type {args.robot}")
        # self._ik_processor: IKProcessor = IK_PROCESSOR_MAP[args.robot](self)

    def ikProcessCallback(self):
        import time
        msg = IKSol()
        msg.timestamp.seconds = time.time_ns() // 1_000_000_000
        msg.timestamp.nanos = time.time_ns() % 1_000_000_000
        sol_q, sol_tauff, left_ee_mat, right_ee_mat = self.generate_fake_dual_arm_ik()

        msg.dual_arm_sol_q.extend(sol_q.tolist())
        msg.dual_arm_sol_tauff.extend(sol_tauff.tolist())
        msg.debug_info.left_ee_pose.extend(left_ee_mat.flatten().tolist())
        msg.debug_info.right_ee_pose.extend(right_ee_mat.flatten().tolist())
        self._publisher.publish(UInt8MultiArray(data=msg.SerializeToString()))
        print(f"{FOURIER_IK_SOL_TOPIC} pub sol_q {sol_q}")

    def trackStateCallback(self, msg: UInt8MultiArray):
        state = TeleState()
        try:
            binary_data = bytes(msg.data)
            state.ParseFromString(binary_data)
            with self._state_lock:
                self._state.CopyFrom(state)
                self._state_flush_cnt += 1
        except Exception as e:
            self.get_logger().error(f'解析 Protobuf 失败: {e}')

    def generate_fake_dual_arm_ik(self):
        """
        生成一条符合真实双臂机器人取值范围的随机 IK 消息数据
        返回:
            sol_q:        (14,)  左7 + 右7 关节角(rad)
            sol_tauff:   (14,)  左7 + 右7 力矩(Nm)
            left_ee_mat: (4,4)  左末端位姿矩阵
            right_ee_mat:(4,4) 右末端位姿矩阵
        """

        # --- 关节角 (rad)
        sol_q = np.random.uniform(low=-2.5, high=2.5, size=14)

        # --- 关节力矩 (Nm)
        sol_tauff = np.random.uniform(low=-20.0, high=20.0, size=14)

        # --- 末端位姿矩阵生成
        def random_pose():
            # 随机位置 ±0.8m
            pos = np.random.uniform(low=[0.2, -0.5, 0.3],
                                    high=[0.8,  0.5, 1.2])

            # 随机小范围旋转
            axis = np.random.randn(3)
            axis /= np.linalg.norm(axis)
            angle = np.random.uniform(-1.0, 1.0)  # ±1 rad
            K = np.array([[0, -axis[2], axis[1]],
                        [axis[2], 0, -axis[0]],
                        [-axis[1], axis[0], 0]])
            R = np.eye(3) + np.sin(angle)*K + (1-np.cos(angle))*(K@K)

            T = np.eye(4)
            T[:3,:3] = R
            T[:3, 3] = pos
            return T

        left_ee_mat  = random_pose()
        right_ee_mat = random_pose()

        return sol_q, sol_tauff, left_ee_mat, right_ee_mat

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frequency", type=float, default=30.0, help="Publishing frequency in Hz")
    parser.add_argument('--log_level', type=str, default='info', help='Logging level')
    parser.add_argument('--robot', type=str, default='unitree_g1_29', help="Robot type")

    args, other_args = parser.parse_known_args()

    logging.getLogger().setLevel(args.log_level.upper())
    try:
        rclpy.init(args=other_args)
        node = IkNode(args)
        rclpy.spin(node)
        rclpy.shutdown()
    except KeyboardInterrupt:
        pass
    finally:
        print("Shutting down ROS2 node...")
        node.destroy_node()

if __name__ == '__main__':
    main()

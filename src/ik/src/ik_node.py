import os, sys
this_file = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(os.path.dirname(this_file), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, '../proto/generate'))

import logging
import argparse
import threading
import numpy as np

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
from ik.src.ik_processor_base import IKProcessor

from ik_register import IK_PROCESSOR_MAP

class IkNode(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__('ik_node')
        logging.info(f"IK Node frequency set to: {args.frequency} Hz")
        self._subscription = self.create_subscription(
            UInt8MultiArray,
            TRACK_STATE_TOPIC,
            self.trackStateCallback,
            10)
        self._subscription  # prevent unused variable warning
        self._state_lock = threading.Lock()
        self._state = TeleState()
        self._state_flush_cnt: np.uint = 0
        self._state_prev_cnt: np.uint = 0

        self._ik_timer = self.create_timer(1.0 / args.frequency, self.ikProcessCallback)
        # from ik.src.unitree.g1_29_ik_processor import G129IkProcessor
        # self._ik_processor: IKProcessor = G129IkProcessor(self)
        if args.robot not in IK_PROCESSOR_MAP:
            raise ValueError(f"not support this robot type {args.robot}")
        self._ik_processor: IKProcessor = IK_PROCESSOR_MAP[args.robot](self)

    def ikProcessCallback(self):
        with self._state_lock:
            if self._state_flush_cnt <= 0:
                return
            if self._state_prev_cnt == self._state_flush_cnt:
                return
            self._state_prev_cnt = self._state_flush_cnt

            current_state = TeleState()
            current_state.CopyFrom(self._state)

        self._ik_processor.Process(current_state)

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

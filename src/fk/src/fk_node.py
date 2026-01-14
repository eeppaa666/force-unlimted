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

from fk_processor_base import FKProcessor
from fk_register import FK_PROCESSOR_MAP

class FkNode(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__('fk_node')
        logging.info(f"FK Node frequency set to: {args.frequency} Hz")
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
        self.args = args
        self._ik_timer = self.create_timer(1.0 / args.frequency, self.fkProcessCallback)
        # from fk.src.unitree.g1_29_fk_processor import G129FKProcessor
        # self._ik_processor: FKProcessor = G129FKProcessor(self)
        if args.robot not in FK_PROCESSOR_MAP:
            raise ValueError(f"not support this robot type {args.robot}")
        self._ik_processor: FKProcessor = FK_PROCESSOR_MAP[args.robot](self)

    def fkProcessCallback(self):
        with self._state_lock:
            if not self._state.start_track:
                return
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
    parser.add_argument("--use_ik_sol", action='store_true', help="use ik solve node state for FK")
    parser.add_argument('--log_level', type=str, default='info', help='Logging level')
    parser.add_argument('--robot', type=str, default='unitree_g1_29', help="Robot type")

    args, other_args = parser.parse_known_args()

    logging.getLogger().setLevel(args.log_level.upper())
    try:
        rclpy.init(args=other_args)
        node = FkNode(args)
        rclpy.spin(node)
        rclpy.shutdown()
    except KeyboardInterrupt:
        pass
    finally:
        print("Shutting down ROS2 node...")
        node.destroy_node()

if __name__ == '__main__':
    main()

import os, sys
this_file = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(os.path.dirname(this_file), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, '../proto/generate'))

import logging
import threading
import numpy as np
from omegaconf import DictConfig, OmegaConf
import hydra

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

from ik.src.ik_register import IK_PROCESSOR_MAP

class IkNode(Node):
    def __init__(self, args: DictConfig):
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
        self._args = args

        self._ik_timer = self.create_timer(1.0 / args.frequency, self.ikProcessCallback)
        # from ik.src.unitree.g1_29_ik_processor import G129IkProcessor
        # self._ik_processor: IKProcessor = G129IkProcessor(self)
        if args.robot.type not in IK_PROCESSOR_MAP:
            raise ValueError(f"not support this robot type {args.robot.type}")
        self._ik_processor: IKProcessor = IK_PROCESSOR_MAP[args.robot.type](self)

    @property
    def args(self):
        return self._args

    def ikProcessCallback(self):
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


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg, resolve=True))

    logging.getLogger().setLevel(cfg.log_level.upper())
    try:
        rclpy.init(args=sys.argv[1:])
        node = IkNode(cfg)
        rclpy.spin(node)
        rclpy.shutdown()
    except KeyboardInterrupt:
        pass
    finally:
        print("Shutting down ROS2 node...")
        node.destroy_node()

if __name__ == '__main__':
    main()

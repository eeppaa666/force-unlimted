import time
import hydra
from omegaconf import DictConfig, OmegaConf

import os, sys
this_file = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(os.path.dirname(this_file), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, '../proto/generate'))

import yaml
import rclpy
from rclpy.node import Node
from std_msgs.msg import UInt8MultiArray

import logging
from multiprocessing_logging import install_mp_handler
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(processName)s] %(filename)s:%(lineno)d - %(message)s' # 这里设置格式
)
install_mp_handler()

from foxglove.src.common.processor_base import FoxgloveProcessor
from foxglove.src.common.message import OutMessage, InMessage
from foxglove.src.common.async_loop import AsyncManager

class FoxgloveNode(Node):
    def __init__(self, args: DictConfig):
        super().__init__('foxglove_node')

        # self._server = FoxgloveServer(args.ip, args.port, "Foxglove ROS2 Bridge")
        self._processor: FoxgloveProcessor = None  # TODO: initialize your processor here
        if args.mode == 'live':
            from foxglove.src.common.websocket_processor import WebsocketProcessor
            self._processor = WebsocketProcessor()
            self._processor.Init(args)
        elif args.mode == 'replay':
            from foxglove.src.common.mcap_processor import MCAPProcessor
            self._processor = MCAPProcessor()
            self._processor.Init(args)
        else:
            raise ValueError(f"Unknown mode: {args.mode}")

        self._async_manager = AsyncManager()
        self._async_manager.start()

        self._subs = []
        for topic in args['topics']:
            sub = self.create_subscription(
                UInt8MultiArray,
                topic,
                self.cretateCallback(topic),
                10
            )
            self._subs.append(sub)
            logging.info(f"Subscribed to topic: {topic}")

    def cretateCallback(self, topic: str):
        def callback(msg: UInt8MultiArray):
            # logging.info(f"msg {topic}")
            in_msg = InMessage()
            in_msg.topic = topic
            in_msg.data = bytes(msg.data)
            in_msg.timestamp_ns = time.time_ns()
            from foxglove.src.converter.converter import Converter
            converter = Converter()
            def cb(out_msg: OutMessage):
                # 如果 Process 也是耗时的，可以考虑将其也放入异步
                self._processor.Process(out_msg)

            self._async_manager.run(converter.Convert(in_msg, cb))

        return callback

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(args: DictConfig):
    print(OmegaConf.to_yaml(args, resolve=True))

    logging.getLogger().setLevel(args.log_level.upper())
    try:
        rclpy.init(args=sys.argv[1:])
        node = FoxgloveNode(args)
        rclpy.spin(node)
        rclpy.shutdown()
    except KeyboardInterrupt:
        pass
    finally:
        print("Shutting down ROS2 node...")
        node.destroy_node()

if __name__ == '__main__':
    main()
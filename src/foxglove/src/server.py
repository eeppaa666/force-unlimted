import time
import asyncio
import argparse
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

from common.processor_base import FoxgloveProcessor
from common.message import OutMessage, InMessage
class FoxgloveNode(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__('foxglove_node')

        # self._server = FoxgloveServer(args.ip, args.port, "Foxglove ROS2 Bridge")
        self._processor: FoxgloveProcessor = None  # TODO: initialize your processor here
        if args.mode == 'live':
            from common.websocket_processor import WebsocketProcessor
            self._processor = WebsocketProcessor()
            self._processor.Init(args)
        elif args.mode == 'replay':
            from common.mcap_processor import MCAPProcessor
            self._processor = MCAPProcessor()
            self._processor.Init(args)
        else:
            raise ValueError(f"Unknown mode: {args.mode}")
        try:
            with open(args.interested_topics, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)  # 推荐使用safe_load避免执行任意代码
        except Exception as e:
            raise ValueError(f"Failed to load interested topics config {e}")

        # 获取当前的 asyncio loop
        # try:
        #     self._loop = asyncio.get_event_loop()
        # except RuntimeError as e:
        #     print(f"{e}")
        #     self._loop = asyncio.new_event_loop()

        self._subs = []
        for topic in config['topics']:
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
            from converter.converter import Converter
            converter = Converter()
            def cb(out_msg: OutMessage):
                # 如果 Process 也是耗时的，可以考虑将其也放入异步
                self._processor.Process(out_msg)

            converter.Convert(in_msg, cb)
            # if self._loop.is_running():
            #     # 我们调用之前写好的非阻塞 Convert
            #     self._loop.call_soon_threadsafe(
            #         lambda: converter.Convert(in_msg, cb)
            #     )
            # else:
            #     # 如果 loop 没跑起来（比如纯同步环境），降级直接运行
            #     # 但这通常不符合你的预期，建议确保 loop 在后台运行
            #     print("Warning: asyncio loop is not running")

        return callback


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--interested_topics', type=str, default=os.path.join(project_root, 'foxglove/config/interested_topics.yaml'), help='Path to the configuration file')
    parser.add_argument('--mode', type=str, default='live', help='run mode: repaly / live')
    # live mode args
    parser.add_argument("--ip", type=str, default="0.0.0.0", help="server ip address")
    parser.add_argument("--port", type=int, default=8765, help="server port")
    # replay mode args
    parser.add_argument('--output', type=str, help='Path to output mcap file')
    # log args
    parser.add_argument('--log_level', type=str, default='info', help='Logging level')

    args, other_args = parser.parse_known_args()

    logging.getLogger().setLevel(args.log_level.upper())
    try:
        rclpy.init(args=other_args)
        node = FoxgloveNode(args)
        while rclpy.ok():
            # 驱动一次 ROS2 任务处理
            # timeout_sec=0 表示立即返回，不阻塞 asyncio 循环
            rclpy.spin_once(node, timeout_sec=0)

            # 交还控制权给 asyncio，让 Foxglove Server 有机会运行
            await asyncio.sleep(0.01)
        # rclpy.spin(node)
        rclpy.shutdown()
    except KeyboardInterrupt:
        pass
    finally:
        print("Shutting down ROS2 node...")
        node.destroy_node()

if __name__ == '__main__':
    asyncio.run(main())
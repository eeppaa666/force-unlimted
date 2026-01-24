import os, sys
this_file = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(os.path.dirname(this_file), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, '../proto/generate'))

import logging
from multiprocessing_logging import install_mp_handler

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(processName)s] %(filename)s:%(lineno)d - %(message)s' # 这里设置格式
)

install_mp_handler()

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, UInt8MultiArray

from ik.ik_sol_pb2 import IKSol
from ik.src.unitree.common import UNITREE_IK_SOL_TOPIC

class MinimalSubscriber(Node):
    def __init__(self):
        super().__init__('minimal_subscriber')
        self.subscription = self.create_subscription(
            UInt8MultiArray,
            UNITREE_IK_SOL_TOPIC,
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg: UInt8MultiArray):
        # 1. 创建一个空的 Protobuf 消息对象
        state = IKSol()

        try:
            # 2. 关键步骤：将 ROS 消息的 data 字段转换为 bytes
            # msg.data 在 rclpy 中通常是一个 array.array 或 list
            # Protobuf 的 ParseFromString 必须接收 bytes 对象
            binary_data = bytes(msg.data)

            # 3. 反序列化
            state.ParseFromString(binary_data)

            from google.protobuf import json_format
            json_string = json_format.MessageToJson(state,
                # always_print_fields_with_no_presence=True, # 强制包含默认值字段
                preserving_proto_field_name=True    # 建议同时开启，保持字段名和 .proto 一致
            )
            print(json_string)

        except Exception as e:
            self.get_logger().error(f'解析 Protobuf 失败: {e}')


def main(args=None):
    rclpy.init(args=args)

    minimal_subscriber = MinimalSubscriber()

    rclpy.spin(minimal_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
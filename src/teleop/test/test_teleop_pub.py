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

from teleop.tele_pose_pb2 import TeleState
from teleop.src.common import TRACK_STATE_TOPIC

class MinimalSubscriber(Node):
    def __init__(self):
        super().__init__('minimal_subscriber')
        self.pub = self.create_publisher(
            UInt8MultiArray,
            TRACK_STATE_TOPIC,
            10)
        self.create_timer(1 / 30, self.listener_callback)

    def listener_callback(self):
        # 1. 创建一个空的 Protobuf 消息对象
        state = TeleState()

        try:
            state.start_track = True
                    # 2. 序列化为二进制字符串
            binary_data = state.SerializeToString()

            # 3. 封装进 ROS 消息
            ros_msg = UInt8MultiArray()
            # 注意：Python 的 bytes 需要转换成 list(int) 供 ROS 2 使用
            ros_msg.data = list(binary_data)
            self.pub.publish(ros_msg)

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
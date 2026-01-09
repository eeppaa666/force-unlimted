import os, sys
this_file = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(os.path.dirname(this_file), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, '../proto/generate'))

import argparse
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

from teleop.src.utils import matrix_to_pose
from teleop.src.televuer.televuer import TeleVuer
from teleop.tele_pose_pb2 import TeleState
from teleop.src.common import TRACK_STATE_TOPIC

class TeleopPublisher(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__('teleop_publisher')

        # teleimager, if you want to test real image streaming, make sure teleimager server is running
        from  image_manager.src.teleimager.image_client import ImageClient
        self.img_client = ImageClient(host=args.image_server_host)
        camera_config = self.img_client.get_cam_config()

        # teleimager + televuer
        self.tv = TeleVuer(use_hand_tracking=args.use_hand_track,
                    binocular=camera_config['head_camera']['binocular'],
                    img_shape=camera_config['head_camera']['image_shape'],
                    display_fps=camera_config['head_camera']['fps'],
                    display_mode=args.display_mode,   # "ego" or "immersive" or "pass-through"
                    zmq=camera_config['head_camera']['enable_zmq'],
                    webrtc=camera_config['head_camera']['enable_webrtc'],
                    webrtc_url=f"https://{args.image_server_host}:{camera_config['head_camera']['webrtc_port']}/offer"
        )

        self.args = args
        self._publisher = self.create_publisher(UInt8MultiArray, TRACK_STATE_TOPIC, 10)

        time_priod = 1.0 / args.pub_frequency
        self.timer = self.create_timer(time_priod, self._timer_pub_callback) #0.5 seconds

        image_time_priod = 1.0 / args.image_frequency
        self.image_timer = self.create_timer(image_time_priod, self._timer_image_callback)

        self.i = 0

    def __del__(self):
        self.tv.close()

    def _timer_image_callback(self):
        img, _= self.img_client.get_head_frame()
        if img is not None:
            self.tv.render_to_xr(img)
            # cv2.imshow("Head Camera", img)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
            # break

    def _timer_pub_callback(self):
        state = TeleState()
        # right arm
        state.right_ctrl_trigger = self.tv.right_ctrl_trigger
        state.right_ctrl_trigger_value = self.tv.right_ctrl_triggerValue
        state.right_ctrl_squeeze = self.tv.right_ctrl_squeeze
        state.right_ctrl_squeeze_value = self.tv.right_ctrl_squeezeValue
        state.right_ctrl_thumbstick = self.tv.right_ctrl_thumbstick
        state.right_ctrl_thumbstick_value.extend(self.tv.right_ctrl_thumbstickValue)
        state.right_ctrl_a_button = self.tv.right_ctrl_aButton
        state.right_ctrl_b_button = self.tv.right_ctrl_bButton
        pose = matrix_to_pose(self.tv.right_arm_pose)
        if pose is not None:
            state.right_ee_pose.CopyFrom(pose)

        # left arm
        state.left_ctrl_trigger = self.tv.left_ctrl_trigger
        state.left_ctrl_trigger_value = self.tv.left_ctrl_triggerValue
        state.left_ctrl_squeeze = self.tv.left_ctrl_squeeze
        state.left_ctrl_squeeze_value = self.tv.left_ctrl_squeezeValue
        state.left_ctrl_thumbstick = self.tv.left_ctrl_thumbstick
        state.left_ctrl_thumbstick_value.extend(self.tv.left_ctrl_thumbstickValue)
        state.left_ctrl_a_button = self.tv.left_ctrl_aButton
        state.left_ctrl_b_button = self.tv.left_ctrl_bButton
        pose = matrix_to_pose(self.tv.left_arm_pose)
        if pose is not None:
            state.left_ee_pose.CopyFrom(pose)

        # head pose
        pose = matrix_to_pose(self.tv.head_pose)
        if pose is not None:
            state.head_pose.CopyFrom(pose)

        #TODO hand pose

        # 2. 序列化为二进制字符串
        binary_data = state.SerializeToString()

        # 3. 封装进 ROS 消息
        ros_msg = UInt8MultiArray()
        # 注意：Python 的 bytes 需要转换成 list(int) 供 ROS 2 使用
        ros_msg.data = list(binary_data)
        self._publisher.publish(ros_msg)
        self.get_logger().debug('Publishing: "%s"' % ros_msg.data)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_hand_track', action='store_true', help='Enable hand tracking')
    parser.add_argument('--image_server_host', type=str, default='localhost', help='Image server host')
    parser.add_argument('--log_level', type=str, default='info', help='Logging level')
    parser.add_argument("--pub_frequency", type=float, default=30.0, help="Publishing frequency in Hz")
    parser.add_argument("--image_frequency", type=float, default=20.0, help="Image fetching frequency in Hz")
    parser.add_argument("--display_mode", type=str, default="ego", help="vuer display mode")

    args, other_args = parser.parse_known_args()

    logging.getLogger().setLevel(args.log_level.upper())
    try:
        rclpy.init(args=other_args)
        minimal_publisher = TeleopPublisher(args)
        rclpy.spin(minimal_publisher)
        rclpy.shutdown()
    except KeyboardInterrupt:
        pass
    finally:
        print("Shutting down ROS2 node...")
        minimal_publisher.destroy_node()

if __name__ == '__main__':
    main()
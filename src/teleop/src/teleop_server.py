import os, sys
this_file = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(os.path.dirname(this_file), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, '../proto/generate'))

import argparse
import time

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
from foxglove.Pose_pb2 import Pose
from foxglove.Quaternion_pb2 import Quaternion
from foxglove.Vector3_pb2 import Vector3

from teleop.src.utils import *
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
                    dds=camera_config['head_camera']['enable_zmq'],
                    webrtc=camera_config['head_camera']['enable_webrtc'],
                    webrtc_url=f"https://{args.image_server_host}:{camera_config['head_camera']['webrtc_port']}/offer"
        )

        self.args = args
        self._publisher = self.create_publisher(UInt8MultiArray, TRACK_STATE_TOPIC, 10)

        time_priod = 1.0 / args.pub_frequency
        self.timer = self.create_timer(time_priod, self._timer_pub_callback) #0.5 seconds

        image_time_priod = 1.0 / args.image_frequency
        self.image_timer = self.create_timer(image_time_priod, self._timer_image_callback)
        self.start_track = False
        self.base_link = Pose(position=Vector3(x=0, y=0, z=0), orientation=Quaternion(x=0, y=0, z=0, w=1))
        self.prev_left_track_button = False
        self.perv_right_hand_pinch = False
        self.prev_right_hand_pinch_y = 0

    def __del__(self):
        self.tv.close()

    def _timer_image_callback(self):
        img, _= self.img_client.get_head_frame()
        if img is not None:
            self.tv.render_to_xr(img)
            # cv2.imshow("Head Camera", img)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
            # break
    def _set_start_track(self, trigger_button: bool):
        cur_left_a: bool = trigger_button
        if cur_left_a != self.prev_left_track_button:
            if cur_left_a == True and not self.start_track:
                self.start_track = True
                logging.info("start teleop track")
            elif cur_left_a == True and self.start_track:
                self.start_track = False
                logging.info("stop teleop track")
            self.prev_left_track_button = cur_left_a

    def _timer_pub_callback(self):
        if not self.args.use_hand_track:
            self._set_start_track(self.tv.left_ctrl_aButton)
        else:
            self._set_start_track(self.tv.left_hand_pinch)

        state = TeleState()
        if not self.start_track:
            # reset base link pose
            if self.tv.left_ctrl_bButton == True:
                self.base_link = Pose(position=Vector3(x=0, y=0, z=0), orientation=Quaternion(x=0, y=0, z=0, w=1))
            self._move_base_link()
            state.start_track = False
        else:
            state.start_track = True

        state.use_hand_track = self.args.use_hand_track

        state.base_link.extend(Pose2Matrix(self.base_link).flatten())
        state.timestamp.seconds = time.time_ns() // 1_000_000_000
        state.timestamp.nanos = time.time_ns() % 1_000_000_000

        # right ee
        valid = valid_matrix(self.tv.right_arm_pose)
        if valid:
            state.right_ee_pose.extend(self.tv.right_arm_pose.flatten())
        # left ee
        valid = valid_matrix(self.tv.left_arm_pose)
        if valid:
            state.left_ee_pose.extend(self.tv.left_arm_pose.flatten())
        # head pose
        valid = valid_matrix(self.tv.head_pose)
        if valid:
            state.head_pose.extend(self.tv.head_pose.flatten())

        # right arm
        state.right_ctrl_trigger = self.tv.right_ctrl_trigger
        state.right_ctrl_trigger_value = self.tv.right_ctrl_triggerValue
        state.right_ctrl_squeeze = self.tv.right_ctrl_squeeze
        state.right_ctrl_squeeze_value = self.tv.right_ctrl_squeezeValue
        state.right_ctrl_thumbstick = self.tv.right_ctrl_thumbstick
        state.right_ctrl_thumbstick_value.extend(self.tv.right_ctrl_thumbstickValue)
        state.right_ctrl_a_button = self.tv.right_ctrl_aButton
        state.right_ctrl_b_button = self.tv.right_ctrl_bButton

        # left arm
        state.left_ctrl_trigger = self.tv.left_ctrl_trigger
        state.left_ctrl_trigger_value = self.tv.left_ctrl_triggerValue
        state.left_ctrl_squeeze = self.tv.left_ctrl_squeeze
        state.left_ctrl_squeeze_value = self.tv.left_ctrl_squeezeValue
        state.left_ctrl_thumbstick = self.tv.left_ctrl_thumbstick
        state.left_ctrl_thumbstick_value.extend(self.tv.left_ctrl_thumbstickValue)
        state.left_ctrl_a_button = self.tv.left_ctrl_aButton
        state.left_ctrl_b_button = self.tv.left_ctrl_bButton

        if self.args.use_hand_track:
            # hand pose
            ## left hand
            state.left_hand_position.extend(self.tv.left_hand_positions.flatten())
            state.left_hand_orientation.extend(self.tv.left_hand_orientations.flatten())
            state.left_hand_pinch = self.tv.left_hand_pinch
            state.left_hand_pinch_value = self.tv.left_hand_pinchValue
            state.left_hand_squeeze = self.tv.left_hand_squeeze
            state.left_hand_squeeze_value = self.tv.left_hand_squeezeValue
            ## right hand
            state.right_hand_position.extend(self.tv.right_hand_positions.flatten())
            state.right_hand_orientation.extend(self.tv.right_hand_orientations.flatten())
            state.right_hand_pinch = self.tv.right_hand_pinch
            state.right_hand_pinch_value = self.tv.right_hand_pinchValue
            state.right_hand_squeeze = self.tv.right_hand_squeeze
            state.right_hand_squeeze_value = self.tv.right_hand_squeezeValue

        # 2. 序列化为二进制字符串
        binary_data = state.SerializeToString()

        # 3. 封装进 ROS 消息
        ros_msg = UInt8MultiArray()
        # 注意：Python 的 bytes 需要转换成 list(int) 供 ROS 2 使用
        ros_msg.data = list(binary_data)
        self._publisher.publish(ros_msg)

    def _move_base_link(self):
        import numpy as np
        T_ROBOT_OPENXR = np.array([[ 0, 0,-1, 0],
                           [-1, 0, 0, 0],
                           [ 0, 1, 0, 0],
                           [ 0, 0, 0, 1]])
        step = 0.05
        # -----> X
        # |
        # |
        # Y
        # 控制robot base_link系 左右前后
        if not self.args.use_hand_track:
            right_xy = self.tv.right_ctrl_thumbstickValue
            # 控制robot base_link系 上下
            left_xy = self.tv.left_ctrl_thumbstickValue
        else:
            right_hand_pinch = self.tv.right_hand_pinch
            if right_hand_pinch != self.perv_right_hand_pinch:
                if right_hand_pinch == True:
                    self.prev_right_hand_pinch_y = self.tv.right_arm_pose[1, 3]

                self.perv_right_hand_pinch = right_hand_pinch

            right_xy = [0, 0]
            if right_hand_pinch:
                right_xy[1] = -(self.tv.right_arm_pose[1, 3] - self.prev_right_hand_pinch_y)

        head_mat = self.tv.head_pose
        head_mat = T_ROBOT_OPENXR @ head_mat
        # if left_xy[0] != 0:
        #     self.base_link.position.y -= step * left_xy[0]
        # if left_xy[1] != 0:
        #     self.base_link.position.x -= step * left_xy[1]
        self.base_link.position.x = head_mat[0, 3]
        self.base_link.position.y = head_mat[1, 3]

        if right_xy[1] != 0:
            self.base_link.position.z -= step * right_xy[1]


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
import threading
import time
import logging
import numpy as np
import pinocchio as pin
import argparse

from rclpy.node import Node
from std_msgs.msg import UInt8MultiArray

from thirdparty_sdk.fourier.robot_arm_ik_r1lite import Fourier_ArmIK

from ik.src.ik_processor_base import IKProcessor
from ik.src.common import *
from ik.src.fourier.hand import HandRetarget

# proto
from teleop.tele_pose_pb2 import TeleState
from controller.state_pb2 import UnitTreeLowState
from ik.ik_sol_pb2 import IKSol

def PoseProcessEE(ee_mat: np.ndarray, base_mat: np.ndarray):
    ee_mat[0:3, 3] = ee_mat[0:3, 3] - base_mat[0:3, 3]

    # 1. 假设你已经得到了当前的位姿 (4x4 matrix)
    # current_ee_pose 是你从 FK 或 Protobuf 转换出来的矩阵
    T_old = pin.SE3(ee_mat)

    # 2. 定义绕 Y 轴旋转 90 度 (pi/2) 的旋转矩阵
    # pin.utils.rotate(axis, angle)
    R_y_90 = pin.utils.rotate('y', -np.pi/2)

    # 3. 构造一个没有平移、只有旋转的 SE3 对象
    T_rot = pin.SE3(R_y_90, np.zeros(3))

    # 4. 右乘应用旋转 (绕局部坐标系旋转)
    T_new = T_old.act(T_rot)  # 或者直接 T_old * T_rot
    ee_mat = T_new.homogeneous

    trans = np.eye(4)
    trans[0:3, 3] = [0, 0, 0.15]
    ee_mat = ee_mat @ trans
    return ee_mat

def PoseProcessEEForLeftHand(ee_mat: np.ndarray, base_mat: np.ndarray):
    ee_mat[0:3, 3] = ee_mat[0:3, 3] - base_mat[0:3, 3]
    # ee_mat = ee_mat @ T_ROBOT_OPENXR

    wrist2left = np.array([[0, 0, -1, 0], [-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=float)
    ee_mat = ee_mat @ wrist2left

    trans = np.eye(4)
    # trans[0:3, 3] = [0.1, 0, -0.6]
    trans[0:3, 3] = [0, 0, 0.05]
    ee_mat = ee_mat @ trans
    return ee_mat

def PoseProcessEEForRightHand(ee_mat: np.ndarray, base_mat: np.ndarray):
    ee_mat[0:3, 3] = ee_mat[0:3, 3] - base_mat[0:3, 3]

    # R_z_90 = pin.utils.rotate('z', np.pi / 2)
    # T_rot = pin.SE3(R_z_90, np.zeros(3))
    # ee_mat = pin.SE3(ee_mat).act(T_rot).homogeneous

    # R_x_90 = pin.utils.rotate('x', -np.pi / 2)
    # T_rot = pin.SE3(R_x_90, np.zeros(3))
    # ee_mat = pin.SE3(ee_mat).act(T_rot).homogeneous

    wrist2right = np.array([[0, 0, -1, 0], [1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=float)
    ee_mat = ee_mat @ wrist2right

    trans = np.eye(4)
    trans[0:3, 3] = [0, 0, 0.05]
    ee_mat = ee_mat @ trans
    return ee_mat


class Gr1T1Processor(IKProcessor):
    def AddArgs(args: argparse.ArgumentParser) -> None:
        args.add_argument('--hand_config', type=str, default=PROJECT_PROOT + '/ik/config/fourier_dexpilot_dhx.yaml', help='Hand configuration for IK processing')

    def __init__(self, node: Node):
        super().__init__()
        self._node = node
        # low state
        self._subscription = node.create_subscription(
            UInt8MultiArray,
            FOURIER_LOW_STATE_TOPIC,
            self.lowStateCallback,
            10)

        # ik solution publisher
        self._publisher = node.create_publisher(UInt8MultiArray, FOURIER_IK_SOL_TOPIC, 10)

        self._low_state = UnitTreeLowState()
        self._low_state_lock = threading.Lock()
        self._arm_ik = Fourier_ArmIK()
        if node.args.robot.hand:
            self._hand_retarget = HandRetarget(cfg=node.args.robot.hand)
        self.dq_num = 14
        self.hand2fingers_left = np.array([[0, 0, -1, 0], [-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=float)
        self.hand2fingers_right = np.array([[0, 0, -1, 0], [1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=float)

        logging.info("Gr1T1Processor initialized.")

    def lowStateCallback(self, msg: UInt8MultiArray):
        try:
            low_state = UnitTreeLowState()
            low_state.ParseFromString(bytes(msg.data))
            with self._low_state_lock:
                self._low_state.CopyFrom(low_state)
        except Exception as e:
            self._node.get_logger().error(f"Failed to parse UnitreeLowState: {e}")
            return

    def Process(self, tele_state: TeleState):
        with self._low_state_lock:
            low_state_copy = UnitTreeLowState()
            low_state_copy.CopyFrom(self._low_state)
        cur_dual_arm_q = np.array(low_state_copy.dual_arm_q) if len(low_state_copy.dual_arm_q) == self.dq_num else None
        cur_dual_arm_dq = np.array(low_state_copy.dual_arm_dq) if len(low_state_copy.dual_arm_dq) == self.dq_num else None

        if len(tele_state.left_ee_pose) == 0 or \
            len(tele_state.right_ee_pose) == 0 or \
            len(tele_state.head_pose) == 0:
            logging.warning("TeleState is missing required fields for IK processing.")
            return

        # 将xr 坐标系转换为 robot坐标系
        left_ee_mat = np.array(tele_state.left_ee_pose, dtype=np.float64).reshape(4, 4)
        right_ee_mat = np.array(tele_state.right_ee_pose, dtype=np.float64).reshape(4, 4)
        head_ee_mat = np.array(tele_state.head_pose, dtype=np.float64).reshape(4, 4)
        base_link_robot = np.array(tele_state.base_link, dtype=np.float64).reshape(4, 4)

        left_ee_mat_robot = WebXR2RobotForEEPose(left_ee_mat)
        right_ee_mat_robot = WebXR2RobotForEEPose(right_ee_mat)
        head_ee_mat_robot = WebXR2RobotForEEPose(head_ee_mat)

        msg = IKSol()
        if tele_state.use_hand_track:
            left_wrist_mat_robot = PoseProcessEEForLeftHand(left_ee_mat_robot, base_link_robot)
            right_wrist_mat_robot = PoseProcessEEForRightHand(right_ee_mat_robot, base_link_robot)
            # process hand landmarks
            left_fingers = np.append(left_ee_mat[0:3, 3], np.array(tele_state.left_hand_position)).reshape(25, 3)
            right_fingers = np.append(right_ee_mat[0:3, 3], np.array(tele_state.right_hand_position)).reshape(25, 3)
            left_fingers = np.concatenate([left_fingers.T, np.ones((1, left_fingers.shape[0]))])
            right_fingers = np.concatenate([right_fingers.T, np.ones((1, right_fingers.shape[0]))])
            left_fingers = T_ROBOT_OPENXR @ left_fingers
            right_fingers = T_ROBOT_OPENXR @ right_fingers

            rel_left_fingers = fast_mat_inv(left_ee_mat_robot) @ left_fingers
            rel_right_fingers = fast_mat_inv(right_ee_mat_robot) @ right_fingers
            rel_left_fingers = (self.hand2fingers_left.T @ rel_left_fingers)[0:3, :].T
            rel_right_fingers = (self.hand2fingers_right.T @ rel_right_fingers)[0:3, :].T
            hand_dq_left, hand_dq_right = self._hand_retarget.retarget(rel_left_fingers, rel_right_fingers)

            msg.left_hand_q.extend(hand_dq_left)
            msg.right_hand_q.extend(hand_dq_right)
        else:
            left_wrist_mat_robot = PoseProcessEE(left_ee_mat_robot, base_link_robot)
            right_wrist_mat_robot = PoseProcessEE(right_ee_mat_robot, base_link_robot)

        # ik 求解
        sol_q, sol_tuaff = self._arm_ik.solve_ik(left_wrist_mat_robot, right_wrist_mat_robot, cur_dual_arm_q, cur_dual_arm_dq)

        msg.timestamp.seconds = time.time_ns() // 1_000_000_000
        msg.timestamp.nanos = time.time_ns() % 1_000_000_000
        msg.dual_arm_sol_q.extend(sol_q)
        msg.dual_arm_sol_tauff.extend(sol_tuaff)
        msg.debug_info.left_ee_pose.extend(left_wrist_mat_robot.flatten())
        msg.debug_info.right_ee_pose.extend(right_wrist_mat_robot.flatten())
        self._publisher.publish(UInt8MultiArray(data=msg.SerializeToString()))
        logging.debug(f'Published IK solution {sol_q}, {sol_tuaff}')
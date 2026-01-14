import threading
import time
import logging
import numpy as np
import pinocchio as pin

from rclpy.node import Node
from std_msgs.msg import UInt8MultiArray

from thirdparty_sdk.fourier.robot_arm_ik_r1lite import Fourier_ArmIK

from ik_processor_base import IKProcessor
from common import FOURIER_IK_SOL_TOPIC, FOURIER_LOW_STATE_TOPIC
from common import WebXR2RobotForEEPose, Pose2matrix

# proto
from teleop.tele_pose_pb2 import TeleState
from controller.state_pb2 import UnitTreeLowState
from ik.ik_sol_pb2 import UnitTreeIkSol

def PoseProcessEE(martrix: np.ndarray, head_matrix):
    # martrix[0:3, 3] = martrix[0:3, 3] - head_matrix[0:3, 3]
    # 1. 假设你已经得到了当前的位姿 (4x4 matrix)
    # current_ee_pose 是你从 FK 或 Protobuf 转换出来的矩阵
    T_old = pin.SE3(martrix)

    # 2. 定义绕 Y 轴旋转 90 度 (pi/2) 的旋转矩阵
    # pin.utils.rotate(axis, angle)
    R_y_90 = pin.utils.rotate('y', -np.pi/2)

    # 3. 构造一个没有平移、只有旋转的 SE3 对象
    T_rot = pin.SE3(R_y_90, np.zeros(3))

    # 4. 右乘应用旋转 (绕局部坐标系旋转)
    T_new = T_old.act(T_rot)  # 或者直接 T_old * T_rot
    return T_new.homogeneous

class Gr1T1Processor(IKProcessor):
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
        self.dq_num = 14

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

        if not tele_state.HasField("left_ee_pose") or \
            not tele_state.HasField("right_ee_pose") or \
            not tele_state.HasField("head_pose"):
            logging.warning("TeleState is missing required fields for IK processing.")
            return

        # 将xr 坐标系转换为 robot坐标系
        left_ee_mat = Pose2matrix(tele_state.left_ee_pose)
        right_ee_mat = Pose2matrix(tele_state.right_ee_pose)
        head_ee_mat = Pose2matrix(tele_state.head_pose)

        left_ee_mat_robot = WebXR2RobotForEEPose(left_ee_mat)
        right_ee_mat_robot = WebXR2RobotForEEPose(right_ee_mat)
        head_ee_mat_robot = WebXR2RobotForEEPose(head_ee_mat)

        left_ee_mat_robot = PoseProcessEE(left_ee_mat_robot, head_ee_mat_robot)
        right_ee_mat_robot = PoseProcessEE(right_ee_mat_robot, head_ee_mat_robot)
        # ik 求解
        sol_q, sol_tuaff = self._arm_ik.solve_ik(left_ee_mat_robot, right_ee_mat_robot, cur_dual_arm_q, cur_dual_arm_dq)

        msg = UnitTreeIkSol()
        msg.timestamp.seconds = time.time_ns() // 1_000_000_000
        msg.timestamp.nanos = time.time_ns() % 1_000_000_000
        msg.dual_arm_sol_q.extend(sol_q)
        msg.dual_arm_sol_tauff.extend(sol_tuaff)
        msg.debug_info.left_ee_pose.extend(left_ee_mat_robot.flatten())
        msg.debug_info.right_ee_pose.extend(right_ee_mat_robot.flatten())
        self._publisher.publish(UInt8MultiArray(data=msg.SerializeToString()))
        logging.debug(f'Published IK solution {sol_q}, {sol_tuaff}')
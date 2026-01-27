import threading
import time
import logging
import numpy as np
from rclpy.node import Node
from std_msgs.msg import UInt8MultiArray

from thirdparty_sdk.unitree.robot_arm_ik import G1_29_ArmIK

from ik.src.ik_processor_base import IKProcessor
from ik.src.common import *

# proto
from teleop.tele_pose_pb2 import TeleState
from controller.state_pb2 import UnitTreeLowState
from ik.ik_sol_pb2 import IKSol

CONST_HEAD_POSE = np.array([[1, 0, 0, 0],
                            [0, 1, 0, 1.5],
                            [0, 0, 1, -0.2],
                            [0, 0, 0, 1]])

# For Robot initial position
CONST_RIGHT_ARM_POSE = np.array([[1, 0, 0, 0.15],
                                 [0, 1, 0, 1.13],
                                 [0, 0, 1, -0.3],
                                 [0, 0, 0, 1]])

CONST_LEFT_ARM_POSE = np.array([[1, 0, 0, -0.15],
                                [0, 1, 0, 1.13],
                                [0, 0, 1, -0.3],
                                [0, 0, 0, 1]])

CONST_HAND_ROT = np.tile(np.eye(3)[None, :, :], (25, 1, 1))

def PoseProcessEE(martrix: np.ndarray, base_link: np.ndarray):
    martrix[0:3, 3] = martrix[0:3, 3] - base_link[0:3, 3]
    # # martrix[0, 3] += 0.15 # x
    # # martrix[2, 3] += 0.15 # z
    trans = np.eye(4)
    trans[0:3, 3] = [-0.15, 0, 0]
    martrix = martrix @ trans
    return martrix

class G129IkProcessor(IKProcessor):
    def __init__(self, node: Node):
        super().__init__()
        self._node = node

        # low state
        self._subscription = node.create_subscription(
            UInt8MultiArray,
            UNITREE_LOW_STATE_TOPIC,
            self.lowStateCallback,
            10)

        # ik solution publisher
        self._publisher = node.create_publisher(UInt8MultiArray, UNITREE_IK_SOL_TOPIC, 10)

        self._low_state = UnitTreeLowState()
        self._low_state_lock = threading.Lock()
        self._arm_ik = G1_29_ArmIK()

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
        cur_dual_arm_q = np.array(low_state_copy.dual_arm_q) if len(low_state_copy.dual_arm_q) == 14 else None
        cur_dual_arm_dq = np.array(low_state_copy.dual_arm_dq) if len(low_state_copy.dual_arm_dq) == 14 else None

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

        left_ee_mat, _ = safe_mat_update(CONST_LEFT_ARM_POSE, left_ee_mat)
        right_ee_mat, _ = safe_mat_update(CONST_RIGHT_ARM_POSE ,right_ee_mat)
        head_ee_mat, _ = safe_mat_update(CONST_HEAD_POSE, head_ee_mat)

        left_ee_mat_robot = WebXR2RobotForEEPose(left_ee_mat)
        right_ee_mat_robot = WebXR2RobotForEEPose(right_ee_mat)
        head_ee_mat_robot = WebXR2RobotForEEPose(head_ee_mat)

        # postprocess
        left_ee_mat_robot = PoseProcessEE(left_ee_mat_robot, base_link_robot)
        right_ee_mat_robot = PoseProcessEE(right_ee_mat_robot, base_link_robot)

        # ik 求解
        sol_q, sol_tuaff = self._arm_ik.solve_ik(left_ee_mat_robot, right_ee_mat_robot, cur_dual_arm_q, cur_dual_arm_dq)

        msg = IKSol()
        msg.timestamp.seconds = time.time_ns() // 1_000_000_000
        msg.timestamp.nanos = time.time_ns() % 1_000_000_000
        msg.dual_arm_sol_q.extend(sol_q)
        msg.dual_arm_sol_tauff.extend(sol_tuaff)
        msg.debug_info.left_ee_pose.extend(left_ee_mat_robot.flatten())
        msg.debug_info.right_ee_pose.extend(right_ee_mat_robot.flatten())
        self._publisher.publish(UInt8MultiArray(data=msg.SerializeToString()))
        logging.debug(f'Published IK solution {sol_q}, {sol_tuaff}')
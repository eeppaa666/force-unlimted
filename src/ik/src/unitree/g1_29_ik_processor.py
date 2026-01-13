import threading
import time
import logging
import numpy as np
from rclpy.node import Node
from std_msgs.msg import UInt8MultiArray

from thirdparty_sdk.unitree.robot_arm_ik import G1_29_ArmIK
from ik.src.ik_processor_base import IKProcessor
from ik.src.unitree.common import UNITREE_LOW_STATE_TOPIC, UNITREE_IK_SOL_TOPIC
from ik.src.unitree.common import Unitree2RobotForEEPose, Pose2matrix, Matrix2Pose
from foxglove.Pose_pb2 import Pose
from teleop.tele_pose_pb2 import TeleState
from controller.state_pb2 import UnitTreeLowState
from ik.ik_sol_pb2 import UnitTreeIkSol

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

        if not tele_state.HasField("left_ee_pose") or \
            not tele_state.HasField("right_ee_pose") or \
            not tele_state.HasField("head_pose"):
            logging.warning("TeleState is missing required fields for IK processing.")
            return

        # 将xr 坐标系转换为 robot坐标系
        left_ee_mat = Pose2matrix(tele_state.left_ee_pose)
        right_ee_mat = Pose2matrix(tele_state.right_ee_pose)
        left_ee_mat_robot = Unitree2RobotForEEPose(left_ee_mat, Pose2matrix(tele_state.head_pose))
        right_ee_mat_robot = Unitree2RobotForEEPose(right_ee_mat, Pose2matrix(tele_state.head_pose))
        # ik 求解
        sol_q, sol_tuaff = self._arm_ik.solve_ik(left_ee_mat_robot, right_ee_mat_robot, cur_dual_arm_q, cur_dual_arm_dq)

        msg = UnitTreeIkSol()
        msg.timestamp.seconds = time.time_ns() // 1_000_000_000
        msg.timestamp.nanos = time.time_ns() % 1_000_000_000
        msg.dual_arm_sol_q.extend(sol_q)
        msg.dual_arm_sol_tauff.extend(sol_tuaff)
        self._publisher.publish(UInt8MultiArray(data=msg.SerializeToString()))
        logging.debug(f'Published IK solution {sol_q}, {sol_tuaff}')
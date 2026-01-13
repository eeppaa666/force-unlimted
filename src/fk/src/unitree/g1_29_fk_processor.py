import threading
import time
import logging
import numpy as np
from rclpy.node import Node
from std_msgs.msg import UInt8MultiArray

from fk.src.fk_processor_base import FKProcessor
from ik.src.unitree.common import UNITREE_LOW_STATE_TOPIC, UNITREE_IK_SOL_TOPIC

from .root_arm_fk import G1_29_ArmFK
from .common import UNITREE_FK_TRANSFRAME

# proto
from ik.ik_sol_pb2 import UnitTreeIkSol
from teleop.tele_pose_pb2 import TeleState
from controller.state_pb2 import UnitTreeLowState
from foxglove.FrameTransforms_pb2 import FrameTransforms

class G129FKProcessor(FKProcessor):
    def __init__(self, node: Node):
        super().__init__()
        self._node = node

        # low state
        self._subscription = node.create_subscription(
            UInt8MultiArray,
            UNITREE_LOW_STATE_TOPIC,
            self.lowStateCallback,
            10)

        # ik sol
        self._subscription1 = node.create_subscription(
            UInt8MultiArray,
            UNITREE_IK_SOL_TOPIC,
            self.ikSolCallback,
            10)

        # ik solution publisher
        self._publisher = node.create_publisher(UInt8MultiArray, UNITREE_FK_TRANSFRAME, 10)

        self._low_state = UnitTreeLowState()
        self._low_state_lock = threading.Lock()
        self._ik_sol = UnitTreeIkSol()
        self._ik_sol_lock = threading.Lock()

        self._arm_fk = G1_29_ArmFK()

    def lowStateCallback(self, msg: UInt8MultiArray):
        try:
            low_state = UnitTreeLowState()
            low_state.ParseFromString(bytes(msg.data))
            with self._low_state_lock:
                self._low_state.CopyFrom(low_state)
        except Exception as e:
            self._node.get_logger().error(f"Failed to parse UnitreeLowState: {e}")
            return

    def ikSolCallback(self, msg: UInt8MultiArray):
        try:
            ik_sol = UnitTreeIkSol()
            ik_sol.ParseFromString(bytes(msg.data))
            with self._ik_sol_lock:
                self._ik_sol.CopyFrom(ik_sol)
        except Exception as e:
            self._node.get_logger().error(f"Failed to parse UnitreeLowState: {e}")
            return

    def Process(self, tele_state: TeleState):
        with self._low_state_lock:
            low_state_copy = UnitTreeLowState()
            low_state_copy.CopyFrom(self._low_state)
        cur_dual_arm_q = np.array(low_state_copy.dual_arm_q) if len(low_state_copy.dual_arm_q) == 14 else None
        # cur_dual_arm_dq = np.array(low_state_copy.dual_arm_dq) if len(low_state_copy.dual_arm_dq) == 14 else None
        with self._ik_sol_lock:
            ik_sol_q = self._ik_sol.dual_arm_sol_q

        # ik_sol fk
        tfs = self._arm_fk.compute_all_fk(ik_sol_q)
        print(tfs)

        msg = FrameTransforms()
        self._publisher.publish(UInt8MultiArray(data=msg.SerializeToString()))

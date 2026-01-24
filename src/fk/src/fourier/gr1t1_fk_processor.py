import threading
import time
import logging
import numpy as np
from rclpy.node import Node
from std_msgs.msg import UInt8MultiArray

from ik.src.common import FOURIER_IK_SOL_TOPIC, FOURIER_LOW_STATE_TOPIC

from fk_processor_base import FKProcessor
from .root_arm_fk import Gr1T1_ArmFK
from common import FOURIER_FK_TRANSFRAME, TimeNs2GoogleTs

# proto
from ik.ik_sol_pb2 import UnitTreeIkSol
from teleop.tele_pose_pb2 import TeleState
from controller.state_pb2 import UnitTreeLowState
from foxglove.FrameTransforms_pb2 import FrameTransforms
from foxglove.FrameTransform_pb2 import FrameTransform

class Gr1T1FKProcessor(FKProcessor):
    def __init__(self, node: Node):
        super().__init__()
        self._node = node
        # low state
        self._subscription = node.create_subscription(
            UInt8MultiArray,
            FOURIER_LOW_STATE_TOPIC,
            self.lowStateCallback,
            10)

        # ik sol
        self._subscription1 = node.create_subscription(
            UInt8MultiArray,
            FOURIER_IK_SOL_TOPIC,
            self.ikSolCallback,
            10)

        # ik solution publisher
        self._publisher = node.create_publisher(UInt8MultiArray, FOURIER_FK_TRANSFRAME, 10)

        self._low_state = UnitTreeLowState()
        self._low_state_lock = threading.Lock()
        self._ik_sol = UnitTreeIkSol()
        self._ik_sol_lock = threading.Lock()

        self._arm_fk = Gr1T1_ArmFK()

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
        if self._node.args.use_ik_sol:
            if not tele_state.start_track:
                tfs = self._arm_fk.get_init_tfs()
            else:
                with self._ik_sol_lock:
                    sol = UnitTreeIkSol()
                    sol.CopyFrom(self._ik_sol)
                input_q = np.array(sol.dual_arm_sol_q, dtype=np.float64)
                tfs = self._arm_fk.compute_all_fk(input_q, sol.left_hand_q, sol.right_hand_q)
        else:
            with self._low_state_lock:
                low_state_copy = UnitTreeLowState()
                low_state_copy.CopyFrom(self._low_state)
            input_q = np.array(low_state_copy.dual_arm_q) if len(low_state_copy.dual_arm_q) == 14 else None
                # cur_dual_arm_dq = np.array(low_state_copy.dual_arm_dq) if len(low_state_copy.dual_arm_dq) == 14 else None
            # ik_sol fk
            tfs = self._arm_fk.compute_all_fk(input_q)

        if tfs is None:
            return
        msg = FrameTransforms()
        for link_name, pose in tfs.items():
            translation = pose.translation  # numpy array [x, y, z]
            rotation = pose.rotation        # 3x3 rotation matrix
            # print(pose.rotation)
            # 将旋转矩阵转为四元数（可选）
            from scipy.spatial.transform import Rotation as R
            from foxglove.Quaternion_pb2 import Quaternion
            from foxglove.Vector3_pb2 import Vector3
            quat = R.from_matrix(rotation).as_quat()
            msg.transforms.append(
                FrameTransform(
                    parent_frame_id="base_link",
                    child_frame_id=link_name,
                    timestamp=TimeNs2GoogleTs(time.time_ns()),
                    translation=Vector3(x=translation[0], y=translation[1], z=translation[2]),
                    rotation=Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3]),
                )
            )
        self._publisher.publish(UInt8MultiArray(data=msg.SerializeToString()))

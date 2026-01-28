import threading
import time
import logging
import numpy as np
from rclpy.node import Node
from std_msgs.msg import UInt8MultiArray

from ik.src.common import *

from fk.src.fk_processor_base import FKProcessor
from fk.src.root_fk import RobotFK
from fk.src.common import FOURIER_FK_TRANSFRAME, TimeNs2GoogleTs

# proto
from ik.ik_sol_pb2 import IKSol
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
        self._ik_sol = IKSol()
        self._ik_sol_lock = threading.Lock()

        self.mixed_jonits = [
            "left_hip_roll_joint", "left_hip_yaw_joint", "left_hip_pitch_joint", "left_knee_pitch_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
            "right_hip_roll_joint", "right_hip_yaw_joint", "right_hip_pitch_joint", "right_knee_pitch_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
            "waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint",
            "head_yaw_joint", "head_roll_joint", "head_pitch_joint",
        ]

        self.active_joints = [
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_pitch_joint",
            "left_wrist_yaw_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_pitch_joint",
            "right_wrist_yaw_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",

            'L_index_proximal_joint', 'L_index_intermediate_joint', 'L_middle_proximal_joint',
            'L_middle_intermediate_joint', 'L_pinky_proximal_joint',
            'L_pinky_intermediate_joint', 'L_ring_proximal_joint', 'L_ring_intermediate_joint',
            'L_thumb_proximal_yaw_joint', 'L_thumb_proximal_pitch_joint', 'L_thumb_distal_joint',

            'R_index_proximal_joint', 'R_index_intermediate_joint', 'R_middle_proximal_joint',
            'R_middle_intermediate_joint', 'R_pinky_proximal_joint',
            'R_pinky_intermediate_joint', 'R_ring_proximal_joint', 'R_ring_intermediate_joint',
            'R_thumb_proximal_yaw_joint', 'R_thumb_proximal_pitch_joint', 'R_thumb_distal_joint',
        ]

        self.urdf_path = os.path.join(PROJECT_PROOT, '../assets/fourier/urdf/GR1T1_fourier_hand_6dof.urdf')
        self.model_dir = os.path.join(PROJECT_PROOT, '../assets/fourier/urdf')

        self._arm_fk = RobotFK(urdf_path=self.urdf_path, model_dir=self.model_dir, mixed_joints=self.mixed_jonits, activate_joints=self.active_joints)

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
            ik_sol = IKSol()
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
                    sol = IKSol()
                    sol.CopyFrom(self._ik_sol)

                input_q = np.array(sol.dual_arm_sol_q, dtype=np.float64)
                input_q = np.concatenate([input_q, sol.left_hand_q, sol.right_hand_q])
                tfs = self._arm_fk.compute_all_fk(input_q)
        else:
            with self._low_state_lock:
                low_state_copy = UnitTreeLowState()
                low_state_copy.CopyFrom(self._low_state)
            input_q = np.array(low_state_copy.dual_arm_q) if len(low_state_copy.dual_arm_q) == 14 else None

            # ik_sol fk
            tfs = self._arm_fk.compute_all_fk(input_q)

        if tfs is None:
            return
        msg = FrameTransforms()
        for link_name, pose in tfs.items():
            translation = pose.translation  # numpy array [x, y, z]
            rotation = pose.rotation        # 3x3 rotation matrix

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

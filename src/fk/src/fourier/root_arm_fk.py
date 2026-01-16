import casadi
import meshcat.geometry as mg
import numpy as np
import pinocchio as pin
import time
from pinocchio import casadi as cpin
from pinocchio.visualize import MeshcatVisualizer
import os
import pickle
import logging

import os, sys
this_file = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(os.path.dirname(this_file), '../../../'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'proto/generate'))

class Gr1T1_ArmFK:
    def __init__(self, Unit_Test = False, Visualization = False):
        np.set_printoptions(precision=5, suppress=True, linewidth=200)

        self.Unit_Test = Unit_Test
        self.Visualization = Visualization

        # fixed cache file path
        self.cache_path = "fourier_model_cache.pkl"

        self.urdf_path = os.path.join(project_root, '../assets/fourier/urdf/gr1t1_fourier_hand_6dof.urdf')
        self.model_dir = os.path.join(project_root, '../assets/fourier/')

        # Try loading cache first
        if os.path.exists(self.cache_path) and (not self.Visualization):
            logging.info(f"[Fourier_ArmIK] >>> Loading cached robot model: {self.cache_path}")
            self.robot, self.reduced_robot = self.load_cache()
        else:
            logging.info("[Fourier_ArmIK] >>> Loading URDF (slow)...")
            self.robot = pin.RobotWrapper.BuildFromURDF(self.urdf_path, self.model_dir)

            self.mixed_jointsToLockIDs = [
                "left_hip_roll_joint", "left_hip_yaw_joint", "left_hip_pitch_joint", "left_knee_pitch_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
                "right_hip_roll_joint", "right_hip_yaw_joint", "right_hip_pitch_joint", "right_knee_pitch_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
                "waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint",
                "head_yaw_joint", "head_roll_joint", "head_pitch_joint",
                "L_thumb_proximal_yaw_joint", "L_thumb_proximal_pitch_joint", "L_thumb_distal_joint",
                "L_index_proximal_joint", "L_index_intermediate_joint",
                "L_middle_proximal_joint", "L_middle_intermediate_joint",
                "L_ring_proximal_joint", "L_ring_intermediate_joint",
                "L_pinky_proximal_joint", "L_pinky_intermediate_joint",
                "R_thumb_proximal_yaw_joint", "R_thumb_proximal_pitch_joint", "R_thumb_distal_joint",
                "R_index_proximal_joint", "R_index_intermediate_joint",
                "R_middle_proximal_joint", "R_middle_intermediate_joint",
                "R_ring_proximal_joint", "R_ring_intermediate_joint",
                "R_pinky_proximal_joint", "R_pinky_intermediate_joint"
            ]

            self.reduced_robot = self.robot.buildReducedRobot(
                list_of_joints_to_lock=self.mixed_jointsToLockIDs,
                reference_configuration=np.array([0.0] * self.robot.model.nq),
            )

            self.reduced_robot.model.addFrame(
                pin.Frame('L_ee',
                        self.reduced_robot.model.getJointId('left_wrist_pitch_joint'),
                        pin.SE3(np.eye(3),
                                np.array([0,0,-0.05]).T),
                        pin.FrameType.OP_FRAME)
            )

            self.reduced_robot.model.addFrame(
                pin.Frame('R_ee',
                        self.reduced_robot.model.getJointId('right_wrist_pitch_joint'),
                        pin.SE3(np.eye(3),
                                np.array([0,0,-0.05]).T),
                        pin.FrameType.OP_FRAME)
            )

            # Save cache (only after everything is built)
            if not os.path.exists(self.cache_path):
                self.save_cache()
                logging.info(f">>> Cache saved to {self.cache_path}")

        self.data = self.reduced_robot.model.createData()

        # 获取详细的 q 向量布局
        model = self.reduced_robot.model
        for i in range(1, model.njoints):
            joint = model.joints[i]
            name = model.names[i]
            print(f"q[{joint.idx_q}] 对应关节: {name}")

     # Save both robot.model and reduced_robot.model
    def save_cache(self):
        data = {
            "robot_model": self.robot.model,
            "reduced_model": self.reduced_robot.model,
        }

        with open(self.cache_path, "wb") as f:
            pickle.dump(data, f)

    # Load both robot.model and reduced_robot.model
    def load_cache(self):
        with open(self.cache_path, "rb") as f:
            data = pickle.load(f)

        robot = pin.RobotWrapper()
        robot.model = data["robot_model"]
        robot.data = robot.model.createData()

        reduced_robot = pin.RobotWrapper()
        reduced_robot.model = data["reduced_model"]
        reduced_robot.data = reduced_robot.model.createData()

        return robot, reduced_robot

    def scale_arms(self, human_left_pose, human_right_pose, human_arm_length=0.60, robot_arm_length=0.75):
        scale_factor = robot_arm_length / human_arm_length
        robot_left_pose = human_left_pose.copy()
        robot_right_pose = human_right_pose.copy()
        robot_left_pose[:3, 3] *= scale_factor
        robot_right_pose[:3, 3] *= scale_factor
        return robot_left_pose, robot_right_pose

    def get_link_pose(self, q, link_name):
        """
        获取特定 Link 的位姿
        """
        pin.framesForwardKinematics(self.reduced_robot.model, self.data, q)
        if self.reduced_robot.model.existFrame(link_name):
            frame_id = self.reduced_robot.model.getFrameId(link_name)
            return self.data.oMf[frame_id].homogeneous()
        else:
            logging.error(f"Frame {link_name} not found in model.")
            return None
    def get_init_tfs(self):
        # 1. 获取 neutral 状态的 q (通常是全 0，对于浮动基座机器人，位姿部分为单位阵)
        q_neutral = pin.neutral(self.reduced_robot.model)

        # 2. 计算一次初始状态的 FK
        # 注意：这里使用 self.data，它会更新 self.data.oMf 数组
        pin.framesForwardKinematics(self.reduced_robot.model, self.data, q_neutral)

        frames_transforms = {}
        for i in range(self.reduced_robot.model.nframes):
            frame = self.reduced_robot.model.frames[i]
            frame_name = frame.name

            # oMf (Object-to-Frame) 是相对于世界坐标系的位姿 SE3
            # .homogeneous() 返回 4x4 numpy 矩阵
            # frames_transforms[frame_name] = self.data.oMf[i].homogeneous
            frames_transforms[frame_name] = self.data.oMf[i]
            # frames_transforms[frame_name] = self.reduced_robot.data.oMf[i]

        return frames_transforms

    def compute_all_fk(self, q):
        """
        根据给定的关节角 q 计算所有 Frame (Link) 的位姿
        :param q: 关节位置向量 (numpy array)
        :return: 字典 { frame_name: 4x4_transform_matrix }
        """
        # 1. 确保 q 的维度正确
        if q is None or len(q) != self.reduced_robot.model.nq:
            # 如果传入的 q 只有手臂部分，可能需要补全（假设其余锁定关节为0）
            # 这里简单处理，假设传入的就是 reduced_model 对应长度的 q
            logging.warning(f"fk node input q invalid q:{q} nq:{self.reduced_robot.model.nq}")
            return None
        # 1. 创建符合 Pinocchio 要求的全 0 向量 (nq 应该是 14)
        q_pin = pin.neutral(self.reduced_robot.model)

        # 2. 根据映射关系填值
        # 这样无论 Pinocchio 内部顺序如何，都能保证值给到了正确的关节
        # for input_idx, pin_q_idx in enumerate(self.q_index_map):
        #     q_pin[pin_q_idx] = q[input_idx]

        # 2. 计算 FK
        # pin.forwardKinematics 只更新 Joint
        # pin.framesForwardKinematics 更新 Joint 并同步更新所有 Frame (Link)
        pin.framesForwardKinematics(self.reduced_robot.model, self.data, q)
        # self.reduced_robot.framesForwardKinematics(q_pin)
        # 3. 提取所有 Frame 的变换矩阵
        frames_transforms = {}
        for i in range(self.reduced_robot.model.nframes):
            frame = self.reduced_robot.model.frames[i]
            frame_name = frame.name

            # oMf (Object-to-Frame) 是相对于世界坐标系的位姿 SE3
            # .homogeneous() 返回 4x4 numpy 矩阵
            # frames_transforms[frame_name] = self.data.oMf[i].homogeneous
            frames_transforms[frame_name] = self.data.oMf[i]
            # frames_transforms[frame_name] = self.reduced_robot.data.oMf[i]

        return frames_transforms
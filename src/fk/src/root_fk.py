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

class RobotFK:
    def __init__(self, urdf_path=None, model_dir=None, mixed_joints=[], activate_joints=[]):
        np.set_printoptions(precision=5, suppress=True, linewidth=200)

        self.urdf_path = urdf_path
        self.model_dir = model_dir
        import pathlib
        self.cache_path = pathlib.Path(urdf_path).stem + "_fk_model_cache.pkl"
        self.active_joints = activate_joints

        # Try loading cache first
        if os.path.exists(self.cache_path):
            logging.info(f"{pathlib.Path(urdf_path).stem} >>> Loading cached robot model: {self.cache_path}")
            self.robot, self.reduced_robot = self.load_cache()
        else:
            logging.info(f"{pathlib.Path(urdf_path).stem} >>> Loading URDF (slow)...")
            self.robot = pin.RobotWrapper.BuildFromURDF(self.urdf_path, self.model_dir)

            self.reduced_robot = self.robot.buildReducedRobot(
                list_of_joints_to_lock=mixed_joints,
                reference_configuration=np.array([0.0] * self.robot.model.nq),
            )

            # Save cache (only after everything is built)
            if not os.path.exists(self.cache_path):
                self.save_cache()
                logging.info(f">>> Cache saved to {self.cache_path}")


        self.data = self.reduced_robot.model.createData()

        self.joint_name_to_q_idx = {}
        for i in range(1, self.reduced_robot.model.njoints):
            name = self.reduced_robot.model.names[i]
            idx_q = self.reduced_robot.model.joints[i].idx_q
            self.joint_name_to_q_idx[name] = idx_q
            print(f"Joint: {name}, q index: {idx_q}")

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
            frames_transforms[frame_name] = self.data.oMf[i]

        return frames_transforms

    def compute_all_fk(self, q):
        # 1. 创建符合 Pinocchio 要求的全 0 向量 (nq 应该是 14)
        q_full = pin.neutral(self.reduced_robot.model)

        for i in range(len(q)):
            for i, val in enumerate(q):
                joint_name = self.active_joints[i]
                if joint_name in self.joint_name_to_q_idx:
                    q_full[self.joint_name_to_q_idx[joint_name]] = val

        pin.framesForwardKinematics(self.reduced_robot.model, self.data, q_full)

        # self.reduced_robot.framesForwardKinematics(q_pin)
        # 3. 提取所有 Frame 的变换矩阵
        frames_transforms = {}
        for i in range(self.reduced_robot.model.nframes):
            frame = self.reduced_robot.model.frames[i]
            frame_name = frame.name
            frames_transforms[frame_name] = self.data.oMf[i]

        return frames_transforms
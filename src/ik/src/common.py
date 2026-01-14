# unitree
UNITREE_IK_SOL_TOPIC = "/unitree/ik/sol"
UNITREE_LOW_STATE_TOPIC = "/unitree/controller/low_state"
# UNITREE_DEBUG_TOPIC = "/unitree/ik/debug_info"

# fourier
FOURIER_IK_SOL_TOPIC = "/fourier/ik/sol"
FOURIER_LOW_STATE_TOPIC = "/fourier/controller/low_state"

import numpy as np
from scipy.spatial.transform import Rotation as R
from foxglove.Pose_pb2 import Pose
from foxglove.Vector3_pb2 import Vector3
from foxglove.Quaternion_pb2 import Quaternion

def safe_mat_update(default_mat, mat):
    # Return previous matrix and False flag if the new matrix is non-singular (determinant ≠ 0).
    det = np.linalg.det(mat)
    if not np.isfinite(det) or np.isclose(det, 0.0, atol=1e-6):
        return default_mat, False
    return mat, True

def fast_mat_inv(mat):
    ret = np.eye(4)
    ret[:3, :3] = mat[:3, :3].T
    ret[:3, 3] = -mat[:3, :3].T @ mat[:3, 3]
    return ret

def safe_rot_update(prev_rot_array, rot_array):
    dets = np.linalg.det(rot_array)
    if not np.all(np.isfinite(dets)) or np.any(np.isclose(dets, 0.0, atol=1e-6)):
        return prev_rot_array, False
    return rot_array, True

# constants variable
T_TO_UNITREE_HUMANOID_LEFT_ARM = np.array([[1, 0, 0, 0],
                                           [0, 0,-1, 0],
                                           [0, 1, 0, 0],
                                           [0, 0, 0, 1]])

T_TO_UNITREE_HUMANOID_RIGHT_ARM = np.array([[1, 0, 0, 0],
                                            [0, 0, 1, 0],
                                            [0,-1, 0, 0],
                                            [0, 0, 0, 1]])

T_TO_UNITREE_HAND = np.array([[0,  0, 1, 0],
                              [-1, 0, 0, 0],
                              [0, -1, 0, 0],
                              [0,  0, 0, 1]])

T_ROBOT_OPENXR = np.array([[ 0, 0,-1, 0],
                           [-1, 0, 0, 0],
                           [ 0, 1, 0, 0],
                           [ 0, 0, 0, 1]])

T_OPENXR_ROBOT = np.array([[ 0,-1, 0, 0],
                           [ 0, 0, 1, 0],
                           [-1, 0, 0, 0],
                           [ 0, 0, 0, 1]])

R_ROBOT_OPENXR = np.array([[ 0, 0,-1],
                           [-1, 0, 0],
                           [ 0, 1, 0]])

R_OPENXR_ROBOT = np.array([[ 0,-1, 0],
                           [ 0, 0, 1],
                           [-1, 0, 0]])

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


def Matrix2Pose(matrix: np.ndarray) -> Pose:
    if matrix.shape != (4, 4) or np.all(matrix[:3, :3] == 0):
        return Pose()
    # 1. 提取平移向量 (前三行，最后一列)
    position = matrix[:3, 3]
    # 2. 提取旋转矩阵 (左上角 3x3)
    rotation_matrix = matrix[:3, :3]

    # 3. 转换为四元数
    # SciPy 默认返回的顺序是 [x, y, z, w]
    quat = R.from_matrix(rotation_matrix).as_quat()

    pose = Pose()
    pose.position.CopyFrom(Vector3(x=position[0], y=position[1], z=position[2]))
    pose.orientation.CopyFrom(Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3]))
    return pose

def Pose2matrix(pose: Pose) -> np.ndarray:
    matrix = np.eye(4)
    # 提取位置
    matrix[0, 3] = pose.position.x
    matrix[1, 3] = pose.position.y
    matrix[2, 3] = pose.position.z

    # 提取四元数
    quat = [pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w]
    rotation_matrix = R.from_quat(quat).as_matrix()
    matrix[:3, :3] = rotation_matrix
    return matrix

def WebXR2RobotForEEPose(ee_mat: np.ndarray, head_mat: np.ndarray) -> np.ndarray:
    # Controller pose data directly follows the (initial pose) Unitree Humanoid Arm URDF Convention (thus no transform is needed).
    head_pose_from_xrWorld, head_pose_valid = safe_mat_update(CONST_HEAD_POSE, head_mat)
    ee_pose_from_xrWorld, ee_mat_valid = safe_mat_update(CONST_LEFT_ARM_POSE, ee_mat)
    # Change basis convention
    head_pose_from_robotWorld = T_ROBOT_OPENXR @ head_pose_from_xrWorld @ T_OPENXR_ROBOT
    ee_pose_from_robotWorld = T_ROBOT_OPENXR @ ee_pose_from_xrWorld @ T_OPENXR_ROBOT
    # Transfer from WORLD to HEAD coordinate (translation adjustment only)
    ee_pose_from_head_pose_robotWorld = np.eye(4)
    ee_pose_from_head_pose_robotWorld[0:3, 3] = ee_pose_from_robotWorld[0:3, 3] - head_pose_from_robotWorld[0:3, 3]
    ee_pose_from_head_pose_robotWorld[0:3, 0:3] = ee_pose_from_robotWorld[0:3, 0:3]
    # =====coordinate origin offset=====
    # The origin of the coordinate for IK Solve is near the WAIST joint motor. You can use teleop/robot_control/robot_arm_ik.py Unit_Test to check it.
    # The origin of the coordinate of IPunitree_Brobot_head_arm is HEAD.
    # So it is necessary to translate the origin of IPunitree_Brobot_head_arm from HEAD to WAIST.
    return ee_pose_from_head_pose_robotWorld
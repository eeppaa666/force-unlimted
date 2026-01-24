from google.protobuf import descriptor_pb2
from google.protobuf.descriptor import FileDescriptor
from typing import Set
import numpy as np
from scipy.spatial.transform import Rotation as R
from foxglove.Pose_pb2 import Pose
from foxglove.Vector3_pb2 import Vector3
from foxglove.Quaternion_pb2 import Quaternion

HAND_INDEX_MAP = {
    "wrist": 0,
    "thumb-metacarpal": 1,
    "thumb-phalanx-proximal": 2,
    "thumb-phalanx-distal": 3,
    "thumb-tip": 4,
    "index-finger-metacarpal": 5,
    "index-finger-phalanx-proximal": 6,
    "index-finger-phalanx-intermediate": 7,
    "index-finger-phalanx-distal": 8,
    "index-finger-tip": 9,
    "middle-finger-metacarpal": 10,
    "middle-finger-phalanx-proximal": 11,
    "middle-finger-phalanx-intermediate": 12,
    "middle-finger-phalanx-distal": 13,
    "middle-finger-tip": 14,
    "ring-finger-metacarpal": 15,
    "ring-finger-phalanx-proximal": 16,
    "ring-finger-phalanx-intermediate": 17,
    "ring-finger-phalanx-distal": 18,
    "ring-finger-tip": 19,
    "pinky-finger-metacarpal": 20,
    "pinky-finger-phalanx-proximal": 21,
    "pinky-finger-phalanx-intermediate": 22,
    "pinky-finger-phalanx-distal": 23,
    "pinky-finger-tip": 24
}

def collect_schema_with_deps(desc: FileDescriptor) -> descriptor_pb2.FileDescriptorSet:
    """
    加载 proto schema 以及它的所有依赖，返回 FileDescriptorSet
    """
    visited: Set[str] = set()
    fds = descriptor_pb2.FileDescriptorSet()

    def add_deps(fd: FileDescriptor):
        if fd.name in visited:
            return
        visited.add(fd.name)

        # 拷贝当前 proto 描述信息
        proto = descriptor_pb2.FileDescriptorProto()
        fd.CopyToProto(proto)
        fds.file.append(proto)

        # 递归处理依赖
        for dep in fd.dependencies:
            add_deps(dep)

    add_deps(desc)
    return fds

def register_schema(writer, desc) -> int:
    """
    Registers the schema for the given protobuf message type with the MCAP writer.
    Returns the schema ID assigned by the writer.
    """
    fds = collect_schema_with_deps(desc.file)
    schema_bytes = fds.SerializeToString()
    schema_name = desc.full_name

    schema_id = writer.register_schema(
        name=schema_name,
        encoding="protobuf",
        data=schema_bytes,
    )
    return schema_id


from google.protobuf.timestamp_pb2 import Timestamp
def TimeNs2GoogleTs(time_ns: int) -> Timestamp:
    ret = Timestamp()
    ret.seconds = time_ns // 1_000_000_000
    ret.nanos = time_ns % 1_000_000_000
    return ret

def Matrix2Pose(matrix: np.ndarray) -> Pose:
    if len(matrix) == 16:
        matrix = matrix.reshape(4, 4)
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


def ComposeMatrix(position, orientation):
    # 1. 将 9 个数的 orientation 转为 3x3 矩阵
    rot_matrix = np.array(orientation).reshape(3, 3)

    # 2. 创建 4x4 单位阵
    mat = np.eye(4)

    # 3. 填充左上角 3x3
    mat[:3, :3] = rot_matrix

    # 4. 填充第四列的前三行 (Translation)
    mat[:3, 3] = position

    return mat


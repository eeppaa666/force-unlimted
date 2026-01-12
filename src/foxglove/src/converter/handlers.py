from foxglove.src.common.message import OutMessage, InMessage
from typing import Callable
from scipy.spatial.transform import Rotation as R
import numpy as np

from foxglove.FrameTransforms_pb2 import FrameTransforms
from foxglove.FrameTransform_pb2 import FrameTransform
from foxglove.Quaternion_pb2 import Quaternion
from foxglove.Vector3_pb2 import Vector3

def TeleopTrackState(inmsg: InMessage, callback: Callable[[OutMessage], None]):
    from teleop.tele_pose_pb2 import TeleState
    msg = TeleState()
    msg.ParseFromString(inmsg.data)

    out: OutMessage = OutMessage()
    out.channel = inmsg.topic + ":tfs"
    out.timestamp_ns = inmsg.timestamp_ns
    out.data = FrameTransforms()
    # webxr
    tf = FrameTransform()
    tf.timestamp.seconds = inmsg.timestamp_ns // 1_000_000_000
    tf.timestamp.nanos = inmsg.timestamp_ns % 1_000_000_000
    tf.parent_frame_id = "world"
    tf.child_frame_id = "webxr"
    r1 = R.from_euler('z', -90, degrees=True)
    r2 = R.from_euler('x', 90, degrees=True)
    r = r1 * r2
    q = r.as_quat()
    tf.rotation.CopyFrom(Quaternion(x=q[0], y=q[1], z=q[2], w=q[3]))
    out.data.transforms.append(tf)
    # left_ee
    tf.parent_frame_id = "webxr"
    tf.child_frame_id = "left_ee"
    tf.translation.CopyFrom(msg.left_ee_pose.position)
    tf.rotation.CopyFrom(msg.left_ee_pose.orientation)
    out.data.transforms.append(tf)
    # right_ee
    tf.parent_frame_id = "webxr"
    tf.child_frame_id = "right_ee"
    tf.translation.CopyFrom(msg.right_ee_pose.position)
    tf.rotation.CopyFrom(msg.right_ee_pose.orientation)
    out.data.transforms.append(tf)
    callback(out)

    out1 = OutMessage()
    out1.channel = inmsg.topic
    out1.timestamp_ns = inmsg.timestamp_ns
    out1.data = msg
    callback(out1)
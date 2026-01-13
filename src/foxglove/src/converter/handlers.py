from foxglove.src.common.message import OutMessage, InMessage
from typing import Callable
from scipy.spatial.transform import Rotation as R
import numpy as np

from foxglove.FrameTransforms_pb2 import FrameTransforms
from foxglove.FrameTransform_pb2 import FrameTransform
from foxglove.Quaternion_pb2 import Quaternion
from foxglove.Vector3_pb2 import Vector3
from foxglove.RawImage_pb2 import RawImage

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

def UnitreeIKsol(inmsg: InMessage, callback: Callable[[OutMessage], None]):
    from ik.ik_sol_pb2 import UnitTreeIkSol
    out = OutMessage()
    out.channel = inmsg.topic
    out.timestamp_ns = inmsg.timestamp_ns
    out.data = inmsg.data
    out.type = UnitTreeIkSol()
    callback(out)

def UnitreeFKtfs(inmsg: InMessage, callback: Callable[[OutMessage], None]):
    out = OutMessage()
    out.channel = inmsg.topic
    out.timestamp_ns = inmsg.timestamp_ns
    out.data = inmsg.data
    out.type = FrameTransforms()
    callback(out)

def UnitreeLowState(inmsg: InMessage, callback: Callable[[OutMessage], None]):
    from controller.state_pb2 import UnitTreeLowState
    out = OutMessage()
    out.channel = inmsg.topic
    out.timestamp_ns = inmsg.timestamp_ns
    out.data = inmsg.data
    out.type = UnitTreeLowState()
    callback(out)

def ImageFrame(inmsg: InMessage, callback: Callable[[OutMessage], None]):
    from image.image_pb2 import ImageFrame
    image_frame = ImageFrame()
    image_frame.ParseFromString(inmsg.data)

    msg = RawImage()

    # ===== Timestamp =====
    msg.timestamp.seconds = image_frame.timestamp.seconds
    msg.timestamp.nanos   = image_frame.timestamp.nanos

    # ===== Frame ID =====
    msg.frame_id = image_frame.frame_id

    # ===== Width / Height =====
    msg.width  = image_frame.width
    msg.height = image_frame.height

    # ===== Encoding =====
    if image_frame.pixel_format == ImageFrame.RGB8:
        msg.encoding = "rgb8"
    elif image_frame.pixel_format == ImageFrame.BGR8:
        msg.encoding = "bgr8"
    elif image_frame.pixel_format == ImageFrame.GRAY8:
        msg.encoding = "mono8"
    elif image_frame.pixel_format == ImageFrame.RGBA8:
        msg.encoding = "rgba8"
    elif image_frame.pixel_format == ImageFrame.BGRA8:
        msg.encoding = "bgra8"
    elif image_frame.pixel_format == ImageFrame.DEPTH16:
        msg.encoding = "mono16"
    elif image_frame.pixel_format == ImageFrame.FLOAT32:
        msg.encoding = "32FC1"
    else:
        msg.encoding = "unknown"

    # ===== Step =====
    msg.step = image_frame.width * image_frame.channels

    # ===== Data =====
    if isinstance(image_frame.data, bytes):
        msg.data = image_frame.data
    else:
        # numpy ndarray
        msg.data = image_frame.data.tobytes()

    out = OutMessage()
    out.channel = inmsg.topic
    out.timestamp_ns = inmsg.timestamp_ns
    out.data = msg
    callback(out)

    # out_raw = OutMessage()
    # out_raw.channel = inmsg.topic + "_raw"
    # out_raw.timestamp_ns = inmsg.timestamp_ns
    # out_raw.data = inmsg.data
    # out_raw.type = ImageFrame()
    # callback(out_raw)

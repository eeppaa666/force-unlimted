import logging
from foxglove.src.common.message import OutMessage, InMessage
from typing import Callable
from .handlers import *


FUNCITONS = {
    "/teleop/track_state": TeleopTrackState,
    "/unitree/ik_sol": UnitreeIKsol,
    "/unitree/fk/tfs": UnitreeFKtfs,
    "/unitree/low_state": UnitreeLowState,
    "/unitree/head_frame": ImageFrame,
}

class Converter:
    def __init__(self):
        pass

    def Convert(self, data: InMessage, callback=None):
        if data.topic in FUNCITONS:
            FUNCITONS[data.topic](data, callback)
            # print(data.topic)
        else:
            logging.warning(f"cant processor this topic {data.topic}")



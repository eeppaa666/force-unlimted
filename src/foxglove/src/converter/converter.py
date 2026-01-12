import logging
from foxglove.src.common.message import OutMessage, InMessage
from typing import Callable
from .handlers import *


FUNCITONS = {
    "/teleop/track_state": TeleopTrackState,
}

class Converter:
    def __init__(self):
        pass

    def Convert(self, data: InMessage, callback=None):
        if data.topic in FUNCITONS:
            FUNCITONS[data.topic](data, callback)
        else:
            logging.warning(f"cant processor this topic {data.topic}")



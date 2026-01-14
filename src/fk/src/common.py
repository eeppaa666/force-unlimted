UNITREE_FK_TRANSFRAME = '/unitree/fk/tfs'
FOURIER_FK_TRANSFRAME = '/fourier/fk/tfs'

from google.protobuf.timestamp_pb2 import Timestamp

def TimeNs2GoogleTs(time_ns: int) -> Timestamp:
    ret = Timestamp()
    ret.seconds = time_ns // 1_000_000_000
    ret.nanos = time_ns % 1_000_000_000
    return ret
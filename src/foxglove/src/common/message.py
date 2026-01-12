import abc
from dataclasses import dataclass
import google.protobuf.message

# @dataclass
class OutMessage:
    channel: str
    data: bytes | google.protobuf.message.Message
    timestamp_ns: int

# @dataclass
class InMessage:
    topic: str
    data: bytes
    timestamp_ns: int
import sys, os
from mcap.reader import make_reader
from mcap.writer import Writer
import mcap.writer
import argparse
from pathlib import Path
from std_msgs.msg import UInt8MultiArray

this_file = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(os.path.dirname(this_file), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, '../proto/generate'))

from foxglove.src.common.protobuf_mcap_writer import ProtobufWriter
from foxglove.src.converter.converter import Converter
from foxglove.src.common.message import InMessage, OutMessage

import struct

def manual_decode_uint8_array(raw_cdr_data):
    # 假设没有封装头，数据从 layout 开始
    # 如果有封装头，从第 4 字节开始看
    offset = 0
    if raw_cdr_data[0:2] == b'\x00\x01':
        offset = 4
    
    # 1. 跳过 layout.dim 序列 (4字节长度)
    # 2. 跳过 layout.data_offset (4字节)
    # 3. 读取 data 序列的长度 (4字节)
    
    # 关键：CDR 有 4 字节对齐。对于 UInt8MultiArray，
    # data 的长度位通常在第 12 字节（带头）或第 8 字节（不带头）
    try:
        # 尝试定位 data 长度位
        # 这里的 12 是基于：4(头) + 4(empty dim) + 4(data_offset)
        data_len = struct.unpack('<I', raw_cdr_data[offset+8 : offset+12])[0]
        
        # 提取数据
        actual_data = raw_cdr_data[offset+12 : offset+12+data_len]
        return actual_data
    except Exception as e:
        print(f"Manual decode failed: {e}")
        return raw_cdr_data # 实在不行返回原始数据

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Example script with command-line arguments')

    # 添加命令行参数
    parser.add_argument("--file", type=str, help="input mcap file path")
    parser.add_argument('--output', '-o',type=str, help='ouput mcap')

    # # 解析命令行参数
    args = parser.parse_args()

    if args.output == None:
        args.output = Path(args.file).stem + "_proto.mcap"

    one_file = open(args.output, "wb")
    one_mcap = Writer(one_file, compression=mcap.writer.CompressionType.ZSTD)
    one_mcap.start()
    proto_writer = ProtobufWriter(one_mcap)
    # exist_channels = {}
    # exist_schemas = {}
    with open(args.file, "rb") as input_file:
        reader = make_reader(input_file)
        summary = reader.get_summary()
        # cur_schemas = {}
        # cur_channels = {}
        # for id, schema in summary.schemas.items():
        #     cur_schemas[id] = schema.name
        #     # print(id, schema.name, schema.encoding)

        # for id, channel in summary.channels.items():
        #     cur_channels[id] = channel.topic
        #     # print(id, channel.topic, channel.message_encoding)

        for schema, channel, message in reader.iter_messages():
            print(channel.topic, len(message.data), channel.message_encoding, schema.name)
            # continue
            from std_msgs.msg import UInt8MultiArray
            from rclpy.serialization import deserialize_message

            # 假设 serialized_data 是你获得的 bytes 字节流
            # message_type 是你要转换的目标类
            # msg: UInt8MultiArray = deserialize_message(message.data, UInt8MultiArray)

            inmsg = InMessage()
            inmsg.data = manual_decode_uint8_array(message.data)
            inmsg.topic = channel.topic
            inmsg.timestamp_ns = message.log_time

            def callabck(outmsg: OutMessage):
                proto_writer.write_message(outmsg.channel, outmsg.data, outmsg.type,outmsg.timestamp_ns)

            converter = Converter()
            converter.Convert(inmsg, callback=callabck)
        #     if schema.name not in  exist_schemas:
        #         i = one_mcap.register_schema(schema.name, schema.encoding, schema.data)
        #         exist_schemas[schema.name] = i

        #     if channel.topic not in exist_channels:
        #         schema_id = exist_schemas[cur_schemas[channel.schema_id]]
        #         i = one_mcap.register_channel(channel.topic, channel.message_encoding, schema_id, channel.metadata)
        #         exist_channels[channel.topic] = i

        #     channel_id = exist_channels[channel.topic]
        #     one_mcap.add_message(channel_id=channel_id, log_time=message.log_time, data=message.data, publish_time=message.publish_time)

    one_mcap.finish()
    one_file.close()

    # SortSingleFile(args.output, GenSortedFileName(args.output), args.start, args.end)

import asyncio
import logging
import base64

from foxglove_websocket import run_cancellable
from foxglove_websocket.server import FoxgloveServer, FoxgloveServerListener

from .processor_base import FoxgloveProcessor
from .message import OutMessage
from .utils import collect_schema_with_deps

class WebsocketProcessor(FoxgloveProcessor, FoxgloveServerListener):
    """WebSocket数据处理类"""
    def Init(self, args):
        self._server = FoxgloveServer(args.ip, args.port, "Foxglove WebSocket Server")
        self._server.set_listener(self)

        self._server.start()
        self._channels = dict()
        self._channel_ids = dict()
        logging.info(f"websocket processor start {args.ip} {args.port}")

    def Process(self, msg: OutMessage):
        asyncio.create_task(self.process(msg))

    async def process(self, msg: OutMessage):
        if msg.channel not in self._channels:
            fds = collect_schema_with_deps(msg.data.DESCRIPTOR.file)
            id = await self._server.add_channel({
                "topic": msg.channel,
                "encoding": "protobuf",
                "schemaName": msg.data.DESCRIPTOR.full_name,
                "schema": base64.b64encode(fds.SerializeToString()).decode("utf-8"),
            })
            self._channels[msg.channel] = id
            self._channel_ids[id] = msg.channel
            logging.info(f"websocket register new schema {msg.channel} {msg.data.DESCRIPTOR.full_name}")
        else:
            id = self._channels[msg.channel]
        # print(id, msg.data.DESCRIPTOR.full_name)
        await self._server.send_message(id, msg.timestamp_ns, msg.data.SerializeToString())

    def on_subscribe(self, server, channel_id):
        logging.info(f"websocket on sub {self._channel_ids[channel_id]} {channel_id}")
        return super().on_subscribe(server, channel_id)

    def on_unsubscribe(self, server, channel_id):
        logging.info(f"websocket on unsub {self._channel_ids[channel_id]} {channel_id}")
        return super().on_unsubscribe(server, channel_id)
# async def main():
#     parse = argparse.ArgumentParser()
#     args = parse.parse_args()
#     async with FoxgloveServer("0.0.0.0", 8765, "example server") as server:
#         # # Publish the grid message
#         fds = collect_schema_with_deps(Grid.DESCRIPTOR.file)
#         id = await server.add_channel({
#             "topic": "grid",
#             "encoding": "protobuf",
#             "schemaName": "foxglove.Grid",
#             "schema": base64.b64encode(fds.SerializeToString()).decode("utf-8"),
#         })
#         # print(fds.SerializeToString())
#         while True:
#             # 发送消息
#             await server.send_message(id, time.time_ns(), Grid().SerializeToString())
#             # print(grid.SerializeToString())
#             print("sending")
#             # 等待 1 秒
#             await asyncio.sleep(1)

#         # 永久挂起，保持服务器运行
#         await asyncio.Future()
import asyncio
import logging
import base64

from foxglove_websocket import run_cancellable
from foxglove_websocket.server import FoxgloveServer, FoxgloveServerListener
from google.protobuf.message import Message

from .processor_base import FoxgloveProcessor
from .message import OutMessage
from .utils import collect_schema_with_deps
from .async_loop import AsyncManager

class WebsocketProcessor(FoxgloveProcessor, FoxgloveServerListener):
    """WebSocket数据处理类"""
    def Init(self, args):
        self._async_manager = AsyncManager()

        self._async_manager.runAsync(self.runServer(args))

        self._channels = dict()
        self._channel_ids = dict()
        self._async_manager.start()

        logging.info(f"websocket processor start {args.ip} {args.port}")

    def Process(self, msg: OutMessage):
        # asyncio.create_task(self.process(msg))
        self._async_manager.runAsync(self.process(msg))

    async def runServer(self, args):
        self._server = FoxgloveServer(args.ip, args.port, "Foxglove WebSocket Server")
        self._server.set_listener(self)
        self._server.start()

    async def process(self, msg: OutMessage):
        if msg.channel not in self._channels:
            if isinstance(msg.type, Message):
                desc = msg.type.DESCRIPTOR
            else:
                desc = msg.data.DESCRIPTOR
            fds = collect_schema_with_deps(desc.file)
            id = await self._server.add_channel({
                "topic": msg.channel,
                "encoding": "protobuf",
                "schemaName": desc.full_name,
                "schema": base64.b64encode(fds.SerializeToString()).decode("utf-8"),
            })
            self._channels[msg.channel] = id
            self._channel_ids[id] = msg.channel
            logging.info(f"websocket register new schema {msg.channel} {desc.full_name}")
        else:
            id = self._channels[msg.channel]
        # print(id, msg.data.DESCRIPTOR.full_name)
        if isinstance(msg.data, bytes):
            await self._server.send_message(id, msg.timestamp_ns, msg.data)
        else:
            await self._server.send_message(id, msg.timestamp_ns, msg.data.SerializeToString())

    def on_subscribe(self, server, channel_id):
        logging.info(f"websocket on sub {self._channel_ids[channel_id]} {channel_id}")
        return super().on_subscribe(server, channel_id)

    def on_unsubscribe(self, server, channel_id):
        logging.info(f"websocket on unsub {self._channel_ids[channel_id]} {channel_id}")
        return super().on_unsubscribe(server, channel_id)
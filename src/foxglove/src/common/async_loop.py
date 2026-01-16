import asyncio
import threading
import time

class AsyncManager:
    def __init__(self):
        # 1. 创建一个新的 loop 对象
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run_loop, daemon=True)

    def _run_loop(self):
        # 2. 为当前子线程设置事件循环
        asyncio.set_event_loop(self.loop)
        try:
            # 3. 驱动循环，直到手动调用 loop.stop()
            print("Asyncio loop started in background thread.")
            self.loop.run_forever()
        finally:
            # 6. 关闭循环前的清理工作
            self.loop.close()
            print("Asyncio loop closed.")

    def run(self, fn):
        async def FnWrapper(fn):
            fn()
        future = asyncio.run_coroutine_threadsafe(FnWrapper(fn), self.loop)

    def runAsync(self, fn):
        future = asyncio.run_coroutine_threadsafe(fn, self.loop)

    def start(self):
        # 启动线程
        self.thread.start()

    def stop(self):
        # 5. 停止循环（线程会随之结束）
        # stop 可以在任何线程调用，它是线程安全的
        self.loop.call_soon_threadsafe(self.loop.stop)

# async def send_foxglove_msg(data):
#     # 这里是你的异步发送逻辑
#     # await self.foxglove_server.send(data)
#     print(f"Sending: {data}")

# # 在同步代码（如 ROS2 Callback）中调用：
# def ros_callback(msg):
#     # 将协程安全地提交到后台 loop
#     future = asyncio.run_coroutine_threadsafe(send_foxglove_msg(msg.data), manager.loop)
#     # 如果你想获取结果（会阻塞当前线程直到发送完成）：
#     # result = future.result()
from abc import ABC, abstractmethod
from teleop.tele_pose_pb2 import TeleState

class IKProcessor(ABC):
    def AddArgs(args) -> None:
        """添加命令行参数的模板方法（可选重写）"""
        pass

    """数据处理的模板类"""
    @abstractmethod
    def Process(self, tele_state: TeleState):  # 模板方法（不可重写）
        pass

class DemoTest(IKProcessor):
    """具体实现类"""
    def load_data(self):
        print("加载CSV数据")

    def clean_data(self):
        print("清理CSV数据")

    def analyze(self):
        print("分析CSV数据")

# # 使用
# processor = CSVProcessor()
# processor.process()

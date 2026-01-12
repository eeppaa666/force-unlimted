from abc import ABC, abstractmethod

class ControllerInterface(ABC):
    """
    所有控制器的『行为契约』
    """

    def __init__(self, config: dict):
        self.config_          = config
        self.running_         = False
        self.subscriber_list_ = config.get("sub_topic_list", [])
        self.publisher_list_  = config.get("pub_topic_list", [])

    @abstractmethod
    def Start(self):
        """启动控制器（注册回调 / 启线程 / DDS 初始化）"""
        pass

    @abstractmethod
    def Stop(self):
        """优雅退出"""
        pass

    @abstractmethod
    def IsRunnning(self):
        """一次控制循环（可选）"""
        pass

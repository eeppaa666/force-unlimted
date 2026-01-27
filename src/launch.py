import rclpy
from rclpy.executors import SingleThreadedExecutor
import hydra
from omegaconf import DictConfig
import multiprocessing
from teleop.src.teleop_server import TeleopPublisher
from fk.src.fk_node import FkNode
from ik.src.ik_node import IkNode

# 定义一个启动函数
def run_node(node_class, config):
    rclpy.init()
    node = node_class(config)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

@hydra.main(version_base=None, config_path=".", config_name="launch")
def main(cfg: DictConfig):
    # 准备进程列表
    processes = [
        multiprocessing.Process(target=run_node, args=(TeleopPublisher, cfg.teleop.config)),
        multiprocessing.Process(target=run_node, args=(FkNode, cfg.fk.config)),
        multiprocessing.Process(target=run_node, args=(IkNode, cfg.ik.config)),
    ]

    # 启动所有进程
    for p in processes:
        p.start()

    # 等待进程结束
    for p in processes:
        p.join()

if __name__ == '__main__':
    main()
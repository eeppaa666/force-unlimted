import rclpy
from rclpy.executors import SingleThreadedExecutor
import hydra
from omegaconf import DictConfig, OmegaConf
import multiprocessing
from teleop.src.teleop_server import TeleopPublisher
from fk.src.fk_node import FkNode
from ik.src.ik_node import IkNode

DEFAULT_RUN_NODES = ["teleop", "fk", "ik"]

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
    print(OmegaConf.to_yaml(cfg, resolve=True))

    processes = []
    node_list = list(dict.fromkeys(cfg.launch))
    if len(node_list) == 0:
        node_list = DEFAULT_RUN_NODES

    for node_name in node_list:
        if node_name == "teleop":
            print("Launching TeleopPublisher...")
            processes.append(multiprocessing.Process(target=run_node, args=(TeleopPublisher, cfg.teleop.config)))
        elif node_name == "fk":
            print("Launching FkNode...")
            processes.append(multiprocessing.Process(target=run_node, args=(FkNode, cfg.fk.config)))
        elif node_name == "ik":
            print("Launching IkNode...")
            processes.append(multiprocessing.Process(target=run_node, args=(IkNode, cfg.ik.config)))
        else:
            print(f"Unknown node type: {node_name}")

    # 启动所有进程
    for p in processes:
        p.start()

    # 等待进程结束
    for p in processes:
        p.join()

if __name__ == '__main__':
    main()
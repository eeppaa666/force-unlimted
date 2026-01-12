import argparse
import yaml
import sys

def main():
    parser = argparse.ArgumentParser(description="YAML Support Demo")

    # 1. 定义一个用于接收配置文件路径的参数
    parser.add_argument('--config', type=str, help='Path to YAML config file')

    # 2. 定义你实际需要的业务参数
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=32)

    # 第一次解析：只为了获取 --config 的路径
    args, remaining_argv = parser.parse_known_args()

    if args.config:
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f)
            # 将 YAML 中的键值对设为 argparse 的默认值
            parser.set_defaults(**config_dict)

    # 第二次解析：正式解析所有参数（命令行输入的参数会覆盖 YAML 里的默认值）
    print(remaining_argv)
    args = parser.parse_args()

    print(f"Final params: {args}")

if __name__ == "__main__":
    main()
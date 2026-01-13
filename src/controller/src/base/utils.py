# utils.py
import argparse
import signal
import sys
import logging
import os
import cv2
import logging
from datetime import datetime
import numpy as np
from typing import Callable, Optional

def BuildArgParser(extra_args_fn: Optional[Callable[[argparse.ArgumentParser], None]] = None) -> argparse.ArgumentParser:
    """
    构建基础的 argparse Parser

    Args:
        extra_args_fn: 可选函数，用于向 parser 添加额外参数或修改默认值

    Returns:
        argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(description="Python Controller Arguments")

    # 基础参数
    parser.add_argument(
        "--robot-ip",
        type=str,
        default="192.168.123.161",
        help="Unitree robot IP address"
    )
    parser.add_argument(
        "--rate-hz",
        type=int,
        default=100,
        help="Control loop frequency (Hz)"
    )
    parser.add_argument(
        "--pub_topic_list",
        type=str,
        nargs="*",
        default=[],
        help="List of topics to publish, e.g., --pub_topic_list topic_A topic_B"
    )
    parser.add_argument(
        "--sub_topic_list",
        type=str,
        nargs="*",
        default=[],
        help="List of topics to subscribe, e.g., --sub_topic_list topic_X topic_Y"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="",
        help="Path to additional configuration file (YAML/JSON)"
    )

    # 调用子类或外部传入的函数，增加/修改参数
    if extra_args_fn:
        extra_args_fn(parser)

    return parser

def RegisterShutDownHook(stop_callback):
    """
    注册 Ctrl+C / kill 信号处理
    stop_callback: stop_callback 函数
    """

    def _handler(signum, frame):
        logging.info(f"[ShutdownHook] Received signal {signum}, shutting down...")
        try:
            stop_callback()
        except Exception as e:
            logging.error(f"[ShutdownHook] Error during Stop(): {e}")
        sys.exit(0)

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, _handler)

def SaveImage(
    img: np.ndarray,
    save_dir: str = "./images",
    filename: str = None,
    encode: str = "png"
) -> str:
    """
    通用图像保存函数

    Args:
        img: 要保存的 numpy 图像 (BGR)
        save_dir: 保存目录，默认 ./images
        filename: 文件名，不带后缀，默认按时间戳生成
        encode: 保存格式，'png' 或 'jpg'

    Returns:
        保存的完整路径
    """
    if img is None:
        logging.warning("No image provided to save.")
        return ""

    # 创建目录
    os.makedirs(save_dir, exist_ok=True)

    # 自动生成文件名
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{timestamp}"

    # 确定完整路径
    encode = encode.lower()
    if encode not in ["png", "jpg", "jpeg"]:
        logging.warning(f"Unknown encode type '{encode}', default to 'png'")
        encode = "png"

    file_path = os.path.join(save_dir, f"{filename}.{encode}")

    # 保存
    success = cv2.imwrite(file_path, img)
    if success:
        logging.info(f"Saved image to {file_path}")
    else:
        logging.error(f"Failed to save image to {file_path}")

    return file_path
#!/usr/bin/env python3
# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0

"""
Fourier GR1T1 Pick-Place 场景加载脚本
"""

import argparse
from isaaclab.app import AppLauncher

# 命令行参数
parser = argparse.ArgumentParser(description="Fourier GR1T1 Pick-Place 场景加载")
parser.add_argument("--num_envs", type=int, default=1, help="环境数量")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 启动 Omniverse
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from tasks.fourier_gr1t1_tasks.pick_place_cylinder_gr1t1.pickplace_cylinder_fourier_gr1t1_env_cfg import PickPlaceFourierGR1T1BaseFixEnvCfg

def main():
    print("=" * 60)
    print("Fourier GR1T1 Pick-Place 场景加载")
    print("=" * 60)
    
    # 创建环境配置
    try:
        env_cfg = PickPlaceFourierGR1T1BaseFixEnvCfg()
        print(f"[INFO] 环境配置加载成功")
    except Exception as e:
        print(f"[ERROR] 环境配置加载失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 创建环境
    try:
        env = gym.make("Isaac-PickPlace-Cylinder-Fourier-GR1T1-Joint", cfg=env_cfg)
        print(f"[INFO] 环境创建成功")
    except Exception as e:
        print(f"[ERROR] 环境创建失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"\n[SUCCESS] 场景加载完成！")
    print(f"  • 环境数量: {env.num_envs}")
    print(f"  • 观测空间: {env.observation_space}")
    print(f"  • 动作空间: {env.action_space}")
    
    env.close()
    print("[INFO] 环境已关闭")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
        print("[INFO] 应用已关闭")


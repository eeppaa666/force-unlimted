
# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0  

import gymnasium as gym

from . import pickplace_cylinder_fourier_gr1t1_env_cfg


gym.register(
    id="Isaac-PickPlace-Cylinder-Fourier-GR1T1-Joint",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": pickplace_cylinder_fourier_gr1t1_env_cfg.PickPlaceFourierGR1T1BaseFixEnvCfg,
    },
    disable_env_checker=True,
)


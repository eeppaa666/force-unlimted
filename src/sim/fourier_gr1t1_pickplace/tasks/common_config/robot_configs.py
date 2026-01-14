# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0  
"""
Robot configuration for Fourier GR1T1 and support for Unitree robots (if available).
This is a modified version for the Fourier GR1T1 workspace.
"""

from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass
from robots.fourier import GR1T1_CFG, GR1T1_HIGH_PD_CFG
from typing import Optional, Dict, Tuple, Literal

@configclass
class FourierRobotPresets:
    """
    Fourier GR1T1 robot preset configuration collection
    
    include the common robot configuration preset for different scenes
    """
    
    @classmethod
    def fourier_gr1t1_base_fix(cls, init_pos: Tuple[float, float, float] = (-0.15, 0.0, 0.95),
        init_rot: Tuple[float, float, float, float] = (0.7071, 0, 0, 0.7071)) -> ArticulationCfg:
        """pick-place task configuration - Fourier GR1T1"""
        return GR1T1_CFG.replace(
            prim_path="/World/envs/env_.*/Robot",
            init_state=ArticulationCfg.InitialStateCfg(
                pos=init_pos,
                rot=init_rot,
                joint_pos={".*": 0.0},
                joint_vel={".*": 0.0}
            ),
        )
    
    @classmethod
    def fourier_gr1t1_high_pd(cls, init_pos: Tuple[float, float, float] = (-0.15, 0.0, 0.95),
        init_rot: Tuple[float, float, float, float] = (0.7071, 0, 0, 0.7071)) -> ArticulationCfg:
        """pick-place task configuration - Fourier GR1T1 with high PD gains"""
        return GR1T1_HIGH_PD_CFG.replace(
            prim_path="/World/envs/env_.*/Robot",
            init_state=ArticulationCfg.InitialStateCfg(
                pos=init_pos,
                rot=init_rot,
                joint_pos={".*": 0.0},
                joint_vel={".*": 0.0}
            ),
        )
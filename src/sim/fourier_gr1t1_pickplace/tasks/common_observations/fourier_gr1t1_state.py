# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0  
"""
Fourier GR1T1 robot state observations
"""     
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def get_fourier_joint_names() -> list[str]:
    """Get Fourier GR1T1 joint names based on URDF
    
    Returns:
        List of joint names for Fourier GR1T1 robot (matched with gr1t1_fourier_hand_6dof.urdf)
    """
    return [
        # head joints (3) - order: yaw, roll, pitch
        "head_yaw_joint",
        "head_roll_joint",
        "head_pitch_joint",
        
        # trunk/waist joints (3) - order: yaw, pitch, roll
        "waist_yaw_joint",
        "waist_pitch_joint",
        "waist_roll_joint",
        
        # leg joints (12)
        # left leg (6) - order: roll, yaw, pitch, knee, ankle_pitch, ankle_roll
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_hip_pitch_joint",
        "left_knee_pitch_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        
        # right leg (6)
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_hip_pitch_joint",
        "right_knee_pitch_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        
        # arm joints (12) - 6 DOF per arm (wrist has 3 DOF: yaw, roll, pitch)
        # left arm (6)
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_pitch_joint",
        "left_wrist_yaw_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        
        # right arm (6)
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_pitch_joint",
        "right_wrist_yaw_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
    ]


def get_robot_boy_joint_states(
    env: ManagerBasedRLEnv,
    enable_dds: bool = False,
) -> torch.Tensor:
    """Get the robot body joint states (positions, velocities, torques)
    
    Args:
        env: ManagerBasedRLEnv - reinforcement learning environment instance
        enable_dds: bool - whether to enable DDS publish (not used for Fourier yet)
    
    Returns:
        torch.Tensor: Concatenated [positions, velocities, torques]
        Shape: (batch, num_joints * 3)
    """
    # Get all joint states from the robot
    joint_pos = env.scene["robot"].data.joint_pos
    joint_vel = env.scene["robot"].data.joint_vel
    joint_torque = env.scene["robot"].data.applied_torque
    
    # Get the joint names from the robot
    all_joint_names = env.scene["robot"].data.joint_names
    
    # Get Fourier-specific joint names
    target_joint_names = get_fourier_joint_names()
    
    # Find indices of target joints
    joint_indices = []
    for joint_name in target_joint_names:
        try:
            idx = all_joint_names.index(joint_name)
            joint_indices.append(idx)
        except ValueError:
            print(f"Warning: Joint '{joint_name}' not found in robot. Available joints: {all_joint_names}")
    
    if not joint_indices:
        print("ERROR: No matching joints found! Using all joints as fallback.")
        # Fallback: use all joints
        return torch.cat([joint_pos, joint_vel, joint_torque], dim=-1)
    
    # Convert to tensor
    device = joint_pos.device
    indices_tensor = torch.tensor(joint_indices, dtype=torch.long, device=device)
    
    # Extract target joints
    selected_pos = torch.index_select(joint_pos, 1, indices_tensor)
    selected_vel = torch.index_select(joint_vel, 1, indices_tensor)
    selected_torque = torch.index_select(joint_torque, 1, indices_tensor)
    
    # Concatenate [pos, vel, torque]
    return torch.cat([selected_pos, selected_vel, selected_torque], dim=-1)


def get_robot_gripper_joint_states(
    env: ManagerBasedRLEnv,
    enable_dds: bool = False,
) -> torch.Tensor:
    """Get robot gripper/hand joint states
    
    Fourier GR1T1 has 6-DOF dexterous hands (11 actuated joints per hand):
    - Thumb: yaw, pitch, distal (3 joints)
    - Index: proximal, intermediate (2 joints)
    - Middle: proximal, intermediate (2 joints)
    - Ring: proximal, intermediate (2 joints)
    - Pinky: proximal, intermediate (2 joints)
    
    Args:
        env: ManagerBasedRLEnv - environment instance
        enable_dds: bool - whether to enable DDS publish
    
    Returns:
        torch.Tensor: Hand joint states [positions, velocities]
        Shape: (batch, num_hand_joints * 2)
    """
    joint_pos = env.scene["robot"].data.joint_pos
    joint_vel = env.scene["robot"].data.joint_vel
    all_joint_names = env.scene["robot"].data.joint_names
    
    # Define hand joint names based on URDF
    hand_joint_names = [
        # Left hand (11 actuated joints)
        "L_thumb_proximal_yaw_joint",
        "L_thumb_proximal_pitch_joint",
        "L_thumb_distal_joint",
        "L_index_proximal_joint",
        "L_index_intermediate_joint",
        "L_middle_proximal_joint",
        "L_middle_intermediate_joint",
        "L_ring_proximal_joint",
        "L_ring_intermediate_joint",
        "L_pinky_proximal_joint",
        "L_pinky_intermediate_joint",
        
        # Right hand (11 actuated joints)
        "R_thumb_proximal_yaw_joint",
        "R_thumb_proximal_pitch_joint",
        "R_thumb_distal_joint",
        "R_index_proximal_joint",
        "R_index_intermediate_joint",
        "R_middle_proximal_joint",
        "R_middle_intermediate_joint",
        "R_ring_proximal_joint",
        "R_ring_intermediate_joint",
        "R_pinky_proximal_joint",
        "R_pinky_intermediate_joint",
    ]
    
    # Find indices of hand joints
    hand_indices = []
    for joint_name in hand_joint_names:
        try:
            idx = all_joint_names.index(joint_name)
            hand_indices.append(idx)
        except ValueError:
            # Joint not found, skip it
            pass
    
    if not hand_indices:
        # No hand joints found, return empty tensor
        batch_size = joint_pos.shape[0]
        return torch.zeros((batch_size, 0), device=joint_pos.device)
    
    # Extract hand joints
    device = joint_pos.device
    indices_tensor = torch.tensor(hand_indices, dtype=torch.long, device=device)
    
    selected_pos = torch.index_select(joint_pos, 1, indices_tensor)
    selected_vel = torch.index_select(joint_vel, 1, indices_tensor)
    
    return torch.cat([selected_pos, selected_vel], dim=-1)


__all__ = [
    "get_robot_boy_joint_states",
    "get_robot_gripper_joint_states",
    "get_fourier_joint_names",
]

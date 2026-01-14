import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sim.converters import UrdfConverterCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

#指定一下urdf路径
URDF_PATH = "/home/infra/workspace/hsx/embodi-vis/urdf/gr1t1_fourier_hand_6dof.urdf"

##
# Configuration
##
GR1T1_CFG = ArticulationCfg(
    # Load URDF file
    spawn=sim_utils.UrdfFileCfg(
        asset_path=URDF_PATH,
        fix_base=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=100.0, damping=10.0)
        ),
    ),
    
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.95),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    
    actuators={
        "head": ImplicitActuatorCfg(
            joint_names_expr=[
                "head_.*",
            ],
            effort_limit=None,
            velocity_limit=None,
            stiffness=None,
            damping=None,
        ),
        "trunk": ImplicitActuatorCfg(
            joint_names_expr=[
                "waist_.*",
            ],
            effort_limit=None,
            velocity_limit=None,
            stiffness=None,
            damping=None,
        ),
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_.*",
                ".*_knee_.*",
                ".*_ankle_.*",
            ],
            effort_limit=None,
            velocity_limit=None,
            stiffness=None,
            damping=None,
        ),
        "right-arm": ImplicitActuatorCfg(
            joint_names_expr=[
                "right_shoulder_.*",
                "right_elbow_.*",
                "right_wrist_.*",
            ],
            effort_limit=torch.inf,
            velocity_limit=torch.inf,
            stiffness=None,
            damping=None,
            armature=0.0,
        ),
        "left-arm": ImplicitActuatorCfg(
            joint_names_expr=[
                "left_shoulder_.*",
                "left_elbow_.*",
                "left_wrist_.*",
            ],
            effort_limit=torch.inf,
            velocity_limit=torch.inf,
            stiffness=None,
            damping=None,
            armature=0.0,
        ),
        "right-hand": ImplicitActuatorCfg(
            joint_names_expr=[
                "R_.*",
            ],
            effort_limit=None,
            velocity_limit=None,
            stiffness=None,
            damping=None,
        ),
        "left-hand": ImplicitActuatorCfg(
            joint_names_expr=[
                "L_.*",
            ],
            effort_limit=None,
            velocity_limit=None,
            stiffness=None,
            damping=None,
        ),
    },
)

GR1T1_HIGH_PD_CFG = GR1T1_CFG.replace(
    actuators={
        "trunk": ImplicitActuatorCfg(
            joint_names_expr=["waist_.*"],
            effort_limit=None,
            velocity_limit=None,
            stiffness=4400,
            damping=40.0,
            armature=0.01,
        ),
        "right-arm": ImplicitActuatorCfg(
            joint_names_expr=["right_shoulder_.*", "right_elbow_.*", "right_wrist_.*"],
            stiffness=4400.0,
            damping=40.0,
            armature=0.01,
        ),
        "left-arm": ImplicitActuatorCfg(
            joint_names_expr=["left_shoulder_.*", "left_elbow_.*", "left_wrist_.*"],
            stiffness=4400.0,
            damping=40.0,
            armature=0.01,
        ),
        "right-hand": ImplicitActuatorCfg(
            joint_names_expr=["R_.*"],
            stiffness=None,
            damping=None,
        ),
        "left-hand": ImplicitActuatorCfg(
            joint_names_expr=["L_.*"],
            stiffness=None,
            damping=None,
        ),
    },
)

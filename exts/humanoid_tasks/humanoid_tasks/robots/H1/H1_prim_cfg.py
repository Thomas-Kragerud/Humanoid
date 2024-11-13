# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ActuatorNetMLPCfg, DCMotorCfg, ImplicitActuatorCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg
from omni.isaac.lab.utils.assets import ISAACLAB_NUCLEUS_DIR

TK_LOCAL_ASSET_PATH = "/home/thomas/Assets"
# old_usd_path = f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/H1/h1.usd",
##
# Configuration
##
# noinspection DuplicatedCode
H1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{TK_LOCAL_ASSET_PATH}/H1/modified_feet_isaac_copy/h1.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=4
        ),
    ),

    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.05),
        joint_pos={
            ".*_hip_yaw": 0.0,
            ".*_hip_roll": 0.0,
            ".*_hip_pitch": -0.28,  # -16 degrees
            ".*_knee": 0.79,  # 45 degrees
            ".*_ankle": -0.52,  # -30 degrees
            "torso": 0.0,
            ".*_shoulder_pitch": 0.28,
            ".*_shoulder_roll": 0.0,
            ".*_shoulder_yaw": 0.0,
            ".*_elbow": 0.52,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_yaw", ".*_hip_roll", ".*_hip_pitch", ".*_knee", "torso"],
            effort_limit=300,
            velocity_limit=100.0,
            stiffness={
                ".*_hip_yaw": 150.0,
                ".*_hip_roll": 150.0,
                ".*_hip_pitch": 200.0,
                ".*_knee": 200.0,
                "torso": 200.0,
            },
            damping={
                ".*_hip_yaw": 5.0,
                ".*_hip_roll": 5.0,
                ".*_hip_pitch": 5.0,
                ".*_knee": 5.0,
                "torso": 5.0,
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle"],
            effort_limit=100,
            velocity_limit=100.0,
            stiffness={".*_ankle": 20.0},
            damping={".*_ankle": 4.0},
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[".*_shoulder_pitch", ".*_shoulder_roll", ".*_shoulder_yaw", ".*_elbow"],
            effort_limit=300,
            velocity_limit=100.0,
            stiffness={
                ".*_shoulder_pitch": 40.0,
                ".*_shoulder_roll": 40.0,
                ".*_shoulder_yaw": 40.0,
                ".*_elbow": 40.0,
            },
            damping={
                ".*_shoulder_pitch": 10.0,
                ".*_shoulder_roll": 10.0,
                ".*_shoulder_yaw": 10.0,
                ".*_elbow": 10.0,
            },
        ),
    },
)
"""Configuration for the Unitree H1 Humanoid robot."""

#print(f"{H1_CFG.spawn.usd_path}")
TKH1 = H1_CFG.copy()
#TKH1.spawn.usd_path = f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/H1/h1_minimal.usd"
TKH1.spawn.usd_path = f"{TK_LOCAL_ASSET_PATH}/H1/modified_feet_isaac_copy/h1_minimal.usd"


H1_slider = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        activate_contact_sensors=True,
        #usd_path=f"{TK_LOCAL_ASSET_PATH}/H1/with_box/h1_minimal_w_box_lifted_distance_joint.usd",
        usd_path=f"{TK_LOCAL_ASSET_PATH}/H1/with_box/h1_wrong_root.usd",
        #rigid_props
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=4,
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=2.0,
            angular_damping=2.0,
            max_linear_velocity=500.0,
            max_angular_velocity=500.0,
            max_depenetration_velocity=2.0,
        ),

    ),
    init_state=ArticulationCfg.InitialStateCfg(
        #pos=(0, 0, 1.05), # old perfect pos when root was pelvis
        pos=(0, 0, 2.37709), # with articulation root at box
    #     joint_pos={
    #         ".*_hip_yaw": 0.0,
    #         ".*_hip_roll": 0.0,
    #         ".*_hip_pitch": -0.28,  # -16 degrees
    #         ".*_knee": 0.79,  # 45 degrees
    #         ".*_ankle": -0.52,  # -30 degrees
    #         "torso": 0.0,
    #         ".*_shoulder_pitch": 0.28,
    #         ".*_shoulder_roll": 0.0,
    #         ".*_shoulder_yaw": 0.0,
    #         ".*_elbow": 0.52,
    #         #"box_rot_rod_joint:0": 0.0,
    #         #"box_rot_rod_joint:1": 0.0,
    #         # "rot_y_cap_box": 0.0,
    #         # "right_elbow_box": 0.0,
    #         # "x_slide_joint": 0.0,
    #     },
    #     joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators = {
        "x_actuator": ImplicitActuatorCfg(
            #joint_names_expr = [".*"],
            joint_names_expr=["x_slide_joint"],
            effort_limit=2000.0,
            velocity_limit=100.0,
            stiffness=100.0,
            damping=100.0,
        ),


        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_yaw", ".*_hip_roll", ".*_hip_pitch", ".*_knee", "torso"],
            effort_limit=300,
            velocity_limit=100.0,
            stiffness={
                ".*_hip_yaw": 150.0,
                ".*_hip_roll": 150.0,
                ".*_hip_pitch": 200.0,
                ".*_knee": 200.0,
                "torso": 200.0,
            },
            damping={
                ".*_hip_yaw": 5.0,
                ".*_hip_roll": 5.0,
                ".*_hip_pitch": 5.0,
                ".*_knee": 5.0,
                "torso": 5.0,
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle"],
            effort_limit=100,
            velocity_limit=100.0,
            stiffness={".*_ankle": 20.0},
            damping={".*_ankle": 4.0},
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[".*_shoulder_pitch", ".*_shoulder_roll", ".*_shoulder_yaw", ".*_elbow"],
            effort_limit=300,
            velocity_limit=100.0,
            stiffness={
                ".*_shoulder_pitch": 40.0,
                ".*_shoulder_roll": 40.0,
                ".*_shoulder_yaw": 40.0,
                ".*_elbow": 40.0,
            },
            damping={
                ".*_shoulder_pitch": 10.0,
                ".*_shoulder_roll": 10.0,
                ".*_shoulder_yaw": 10.0,
                ".*_elbow": 10.0,
            },
        ),
    },
)

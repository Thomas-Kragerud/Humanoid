

########
# LAUNCH THE SIMULATOR FIRST
########
import argparse
from omni.isaac.lab.app import AppLauncher

# create argparse
parser = argparse.ArgumentParser(description="Creating empty scene")
parser.add_argument("--num_envs", type=int, default=4, help="Number of envs to spawn")
AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


from omni.isaac.lab.sim import SimulationCfg, SimulationContext, DistantLightCfg, GroundPlaneCfg, ConeCfg
import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.assets import RigidObjectCfg, RigidObject
from pxr import Gf, Sdf, Usd
import omni.isaac.lab.utils.math as math_utils
import torch
from omni.isaac.lab.scene import InteractiveSceneCfg, InteractiveScene
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.assets import AssetBaseCfg, ArticulationCfg, Articulation
from omni.isaac.lab.sim import SimulationContext
from omni.isaac.lab.actuators import ImplicitActuatorCfg

from humanoid_tasks.robots.H1.H1_prim_cfg import H1_slider
TK_LOCAL_ASSET_PATH = "/home/thomas/Assets"
#
# main_prim = ArticulationCfg(
#     spawn=sim_utils.UsdFileCfg(
#         usd_path=f"{TK_LOCAL_ASSET_PATH}/H1/with_box/h1_minimal_w_box_lifted_distance_joint.usd",
#         #rigid_props
#         articulation_props=sim_utils.ArticulationRootPropertiesCfg(
#             enabled_self_collisions=False,
#             solver_position_iteration_count=4,
#             solver_velocity_iteration_count=4,
#         ),
#         rigid_props=sim_utils.RigidBodyPropertiesCfg(
#             disable_gravity=False,
#             retain_accelerations=False,
#             linear_damping=2.0,
#             angular_damping=2.0,
#             max_linear_velocity=500.0,
#             max_angular_velocity=500.0,
#             max_depenetration_velocity=2.0,
#         ),
#
#     ),
#     init_state=ArticulationCfg.InitialStateCfg(
#         pos=(0, 0, 1.05),
#     #     joint_pos={
#     #         ".*_hip_yaw": 0.0,
#     #         ".*_hip_roll": 0.0,
#     #         ".*_hip_pitch": -0.28,  # -16 degrees
#     #         ".*_knee": 0.79,  # 45 degrees
#     #         ".*_ankle": -0.52,  # -30 degrees
#     #         "torso": 0.0,
#     #         ".*_shoulder_pitch": 0.28,
#     #         ".*_shoulder_roll": 0.0,
#     #         ".*_shoulder_yaw": 0.0,
#     #         ".*_elbow": 0.52,
#     #         #"box_rot_rod_joint:0": 0.0,
#     #         #"box_rot_rod_joint:1": 0.0,
#     #         # "rot_y_cap_box": 0.0,
#     #         # "right_elbow_box": 0.0,
#     #         # "x_slide_joint": 0.0,
#     #     },
#     #     joint_vel={".*": 0.0},
#     ),
#     soft_joint_pos_limit_factor=0.9,
#     actuators = {
#         "x_actuator": ImplicitActuatorCfg(
#             #joint_names_expr = [".*"],
#             joint_names_expr=["x_slide_joint"],
#             effort_limit=2000.0,
#             velocity_limit=100.0,
#             stiffness=100.0,
#             damping=100.0,
#         ),
#
#
#         "legs": ImplicitActuatorCfg(
#             joint_names_expr=[".*_hip_yaw", ".*_hip_roll", ".*_hip_pitch", ".*_knee", "torso"],
#             effort_limit=300,
#             velocity_limit=100.0,
#             stiffness={
#                 ".*_hip_yaw": 150.0,
#                 ".*_hip_roll": 150.0,
#                 ".*_hip_pitch": 200.0,
#                 ".*_knee": 200.0,
#                 "torso": 200.0,
#             },
#             damping={
#                 ".*_hip_yaw": 5.0,
#                 ".*_hip_roll": 5.0,
#                 ".*_hip_pitch": 5.0,
#                 ".*_knee": 5.0,
#                 "torso": 5.0,
#             },
#         ),
#         "feet": ImplicitActuatorCfg(
#             joint_names_expr=[".*_ankle"],
#             effort_limit=100,
#             velocity_limit=100.0,
#             stiffness={".*_ankle": 20.0},
#             damping={".*_ankle": 4.0},
#         ),
#         "arms": ImplicitActuatorCfg(
#             joint_names_expr=[".*_shoulder_pitch", ".*_shoulder_roll", ".*_shoulder_yaw", ".*_elbow"],
#             effort_limit=300,
#             velocity_limit=100.0,
#             stiffness={
#                 ".*_shoulder_pitch": 40.0,
#                 ".*_shoulder_roll": 40.0,
#                 ".*_shoulder_yaw": 40.0,
#                 ".*_elbow": 40.0,
#             },
#             damping={
#                 ".*_shoulder_pitch": 10.0,
#                 ".*_shoulder_roll": 10.0,
#                 ".*_shoulder_yaw": 10.0,
#                 ".*_elbow": 10.0,
#             },
#         ),
#     },
# )


@configclass
class MySceneCfg(InteractiveSceneCfg):

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(
            intensity=3000.0,
            color=(0.75, 0.75, 0.75),
        ),
    )

    robot: ArticulationCfg = H1_slider.replace(prim_path="{ENV_REGEX_NS}/robot")


def run(
        sim: sim_utils.SimulationContext,
        scene: InteractiveScene,
) -> None:
    robot: Articulation = scene["robot"]

    sim_dt = sim.get_physics_dt()
    count = 0
    while simulation_app.is_running():
        if count % 500 == 0:
            count = 0
            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            robot.write_root_state_to_sim(root_state)

            joint_pos = robot.data.default_joint_pos.clone()
            joint_vel = robot.data.default_joint_vel.clone()

            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            # clear scene buffers
            scene.reset()


        #efforts = torch.randn_like(robot.data.joint_pos) * 1.0
       # robot.set_joint_effort_target(efforts)
        scene.write_data_to_sim()
        #scene.write_data_to_sim()
        sim.step()
        count += 1
        scene.update(sim_dt)


def main() -> None:
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view((2.5, 0.0, 4.0), (0.0, 0.0, 2.0))

    scene_cfg = MySceneCfg(
        num_envs=args_cli.num_envs,
        env_spacing=2.5,
    )

    scene = InteractiveScene(scene_cfg)

    sim.reset()
    print(f"[INFO]: Setup complete...")
    run(sim, scene)

if __name__ == "__main__":
    main()
    simulation_app.close()
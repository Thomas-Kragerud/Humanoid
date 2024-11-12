########
# LAUNCH THE SIMULATOR FIRST
########
import argparse
from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser(description="Demo of interactive scene")
# Number of envs to be created:
parser.add_argument("--num_envs", type=int, default=4, help="Number of envs to spawn")
AppLauncher.add_app_launcher_args(parser) # Add Issac sim specific arguments
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


"""
After this point, we can import Isaac Sim specific modules because the simulator is now running
"""

from omni.isaac.lab.assets import (
    AssetBase,
    ArticulationCfg,
    Articulation,
    RigidObjectCfg, AssetBaseCfg
)
from omni.isaac.lab.scene import (
    InteractiveSceneCfg,
    InteractiveScene
)

from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.utils import configclass

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.sim import SimulationContext

import torch

THOMAS_ASSETS:str = "/home/thomas/Assets"

XYZ_BOX = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{THOMAS_ASSETS}/MOVING/test_xyz.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=10.0,
            max_angular_velocity=10.0,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,  # IDK
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0, 0, 0), # position in world frame
        # joint_pos={"joint_name_1": 0.0, "joint_name_2": 0.0},
        # joint_vel={"joint_name_1": 0.0, "joint_name_2": 0.0},    )
    ),
    actuators={
        "x_actuator": ImplicitActuatorCfg(
            joint_names_expr=["x_drive_joint"],
            effort_limit=400.0,
            velocity_limit=100.0,
            stiffness=0.0,
            damping=10.0,
        ),
        "y_actuator": ImplicitActuatorCfg(
            joint_names_expr=["y_drive_joint"],
            effort_limit=400.0,
            velocity_limit=100.0,
            stiffness=0.0,
            damping=10.0,
        )
    }
)


@configclass
class TestSceneCfg(InteractiveSceneCfg):
    """ Config class for the scene  """
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(
            intensity=3000.0,
            color=(0.75, 0.75, 0.75),
        ),
    )

    xyz_box: ArticulationCfg = XYZ_BOX.replace(prim_path="{ENV_REGEX_NS}/xyz_box")

def run_sim(sim: sim_utils.SimulationContext, scene: InteractiveScene) -> None:
    """
    Main sim loop
    """
    xyz_box: Articulation = scene["xyz_box"]

    sim_dt = sim.get_physics_dt()

    count = 0
    while simulation_app.is_running():
        if count % 500 == 0:
            count = 0
            root_state = xyz_box.data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins # Offset env origins ???
            xyz_box.write_root_state_to_sim(root_state)

            joint_pos = xyz_box.data.default_joint_pos.clone()
            joint_vel = xyz_box.data.default_joint_vel.clone()

            # add some noice
            joint_pos += torch.rand_like(joint_pos) * 0.1

            xyz_box.write_joint_state_to_sim(joint_pos, joint_vel)

            # clear scene buffers
            scene.reset()
            print(f"[INFO]: Reset scene buffers")
            print(f"Joint pos tensor:\n{xyz_box.data.joint_pos}")
            print(f"Joint pos shape: {xyz_box.data.joint_pos.shape}\n\n")
            print()
            print(f"Joint target tensor:\n{xyz_box.data.joint_vel_target}")
            print(f"Joint target shape: {xyz_box.data.joint_vel_target.shape}")

        if count % 100 == 0:
            all_joint_vel_target = torch.zeros_like(xyz_box.data.joint_vel_target)
            env_1_joint_vel_target = torch.tensor([0, 0.5, 0.5], device=sim.device)

            # set target for first box
            all_joint_vel_target[0, :] = env_1_joint_vel_target
            print("Find:")
            print(f"x_drive index: {xyz_box.find_joints('x_drive_joint')}")
            print(f"y_drive index: {xyz_box.find_joints('y_drive_joint')}\n")

            print(f"All:")
            print(f"Body names {xyz_box.data.body_names}")
            print(f"Joint names {xyz_box.data.joint_names}")

            # set with full state tensor
            # xyz_box.set_joint_position_target(
            #     env_1_joint_vel_target,
            #     joint_ids=None,
            #     env_ids=None,
            # )

            # set with specific env id
            # all_joint_indices = torch.arange(0, xyz_box., dtype=torch.int32, device=sim.device)
            xyz_box.set_joint_velocity_target(
                env_1_joint_vel_target,
                env_ids=[0],
            )

            # set with specific joint id and env_id

            xyz_box.set_joint_velocity_target(
                target=torch.ones((4, 1), device=sim.device) * 0.5,
                joint_ids=[1], # x?
                #env_ids=[1, 2],
            )

        # set velocity targets
        #xyz_box.set_joint_velocity_target()
        scene.write_data_to_sim()
        sim.step()
        count += 1
        scene.update(sim_dt)


def main() -> None:
    """
    Main function sets up and run sim
    """
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)

    sim.set_camera_view((2.5, 0.0, 4.0), (0.0, 0.0, 2.0))

    # Create scene config
    scene_cfg = TestSceneCfg(
        num_envs=args_cli.num_envs,
        env_spacing=2.5,
    )

    scene = InteractiveScene(scene_cfg)

    # Reset the sim to init state
    sim.reset()
    print(f"[INFO]: Setup complete...")

    run_sim(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
########
# LAUNCH THE SIMULATOR FIRST
########
import argparse
from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser(description="Demo of interactive scene")
# Number of envs to be created:
parser.add_argument("--num_envs", type=int, default=2, help="Number of envs to spawn")
AppLauncher.add_app_launcher_args(parser) # Add Issac sim specific arguments
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


"""
After this point, we can import Isaac Sim specific modules because the simulator is now running
"""

from omni.isaac.lab.scene import InteractiveSceneCfg, InteractiveScene
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.assets import AssetBaseCfg, ArticulationCfg, Articulation, RigidObjectCfg
from omni.isaac.lab_assets import CARTPOLE_CFG # Pre-defined cart-pole config
from omni.isaac.lab.sim import SimulationContext
from omni.isaac.lab.actuators import ImplicitActuatorCfg
import omni.isaac.lab.envs.mdp as mdp

from omni.isaac.lab.envs import ManagerBasedEnvCfg, ManagerBasedEnv
from omni.replicator.core.scripts.physics import physics_material
from omni.isaac.lab.managers import (
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    SceneEntityCfg,
)

import omni.isaac.lab.sim as sim_utils
import omni.isaac.core.utils.prims as prim_utils

import torch
THOMAS_ASSETS:str = "/home/thomas/Assets"

XY_Z_BOX = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{THOMAS_ASSETS}/rebuild_cart_pole/simple.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=100.0, # WHAT IS THIS?
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4, # WHAT IS THIS
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001, # IDK
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0, 0, 0), # position of root in sim world frame
        # joint_pos={"joint_name_1": 0.0, "joint_name_2": 0.0},
        # joint_vel={"joint_name_1": 0.0, "joint_name_2": 0.0},
    ),
    actuators={
        "motor": ImplicitActuatorCfg(
            joint_names_expr=["cart_to_pole"],
            effort_limit=400.0,
            velocity_limit=100.0,
            stiffness=0.0,
            damping=10.0,
        ),
    },
)



@configclass
class SimpleSceneCfg(InteractiveSceneCfg):
    """
    Defines all the elements that will be in the scene
    """
    ## Define ground
    # since AssetBaseCfg its non interactive
    # ground = AssetBaseCfg(
    #     prim_path="/World/defaultGroundPlane",
    #     spawn=sim_utils.GroundPlaneCfg(),
    # )

    ## Define light
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(
            intensity=3000.0,
            color=(0.75, 0.75, 0.75),
        )
    )

    ## Define box
    # box = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/Box",
    #     spawn=sim_utils.CuboidCfg(
    #         size=(1.0, 1.0, 1.0),
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(),
    #         mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
    #         #physics_material=sim_utils.PhysicsMaterialCfg(),
    #         collision_props=sim_utils.CollisionPropertiesCfg(),
    #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)),
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=(1.0, -0.2, 0.05)),
    # )

    simple_cart_pole: ArticulationCfg = XY_Z_BOX.replace(prim_path="{ENV_REGEX_NS}/simple_cart_pole")


@configclass
class ActionsCfg:
    """
    Defines the actions space and control schemes for the environment
    """
    # Define joint effort
    joint_effort = mdp.JointEffortActionCfg(
        asset_name="simple_cart_pole",
        joint_names=["cart_to_pole"],
        scale=5.0,
    )

@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        joint_pos_rel = ObservationTermCfg(
            func=mdp.joint_vel_rel,
            params={"asset_cfg":SceneEntityCfg("simple_cart_pole")}
        )

        def __post_init__(self) -> None:
            """ Post-initialization configuration"""
            self.enable_corruption = False # Disable observation noice
            self.concatenate_terms = True  # Combine all terms into single tensor

    policy: PolicyCfg = PolicyCfg()
     

@configclass
class SimpleEnvCfg(ManagerBasedEnvCfg):
    """
    Main environment config that combines all components
    """
    # Scene configuration
    scene = SimpleSceneCfg(num_envs=8, env_spacing=2.5)

    observations = ObservationsCfg()
    actions = ActionsCfg()

    def __post_init__(self) -> None:
        """ Post-initialization configuration """
        self.viewer.eye = (4.5, 0.0, 4.0)
        self.viewer.lookat = (0.0, 0.0, 2.0)

        self.decimation = 4 # env step every 4 sim steps: 200 Hz / 4 = 50Hz
        self.sim.dt = 0.005 # sim step every 5ms: 200 Hz
        self.rendering_interval = 4



def main() -> None:

    env_cfg = SimpleEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs

    # Initialize the env, triggers the following sequence
    # 1. Scene creation and asset spawning
    # 2. Action manager initialization ( Action, Observation, Events)
    # 3. Startup events execution

    env = ManagerBasedEnv(env_cfg)

    ### Run the simulation
    count = 0
    while simulation_app.is_running():
        # Using inference mode to disable PyTorch gradients
        
        with torch.inference_mode():
            # Periodic reset
            if count % 500 == 0:
                count = 0
                env.reset() # Resets scene, event execution, initial observation computation
                print(f"[INFO]: Reset complete...")
            if count % 100 == 0:
                cart_articulation: Articulation = env.scene["simple_cart_pole"]
                #print(f"cart_articulation {cart_articulation}, type {type(cart_articulation)}")

                # operations
                #cart_articulation.set_joint_position_target(position) # what shape does position take here ?

                # write external wrench and joint commands to the sim
                # if any explicit actuators are present, actuator models are used to compute joint commands, otherwise directly set in sim
                cart_articulation.write_data_to_sim() # propagate the actuator models and apply the computed command in sim

                # dictionary of actuator instances for the articulation, keys are names, values instances
                cart_actuators = cart_articulation.actuators

                cart_num_instances = cart_articulation.num_instances
                cart_num_joints = cart_articulation.num_joints
                cart_num_bodies = cart_articulation.num_bodies
                cart_body_names = cart_articulation.body_names
                cart_joint_names = cart_articulation.joint_names

                # articulation view for the asset (PhysX) - use with caution. requires handling of tensors in specific way
                cart_articulation_view_physx = cart_articulation.root_physx_view
                print(f"test: {cart_articulation_view_physx.get_body_names}")


                cart_articulation_device = cart_articulation.device
                cart_is_initialized = cart_articulation.is_initialized

                # find bodies in articulation based on the name keys (name_keys: str | Sequence[str] - tuple[list[int], list[str]]
                # returns a tuple of lists containing the body indices and names
                cart_body = cart_articulation.find_bodies(["cart"])
                print(f"cart_body: {cart_body}")

                # same_but for joints
                cart_find_joint = cart_articulation.find_joints(["cart_to_pole"])
                print(f"cart_joints {cart_find_joint}")

                # Root state comprises of the cartesian positions, quaternion, linear and angular velocity. All in simulation frame
                # cart_articulation.write_root_state_to_sim(root_state[, env_ids]) (root_state: torch.Tensor, env_ids:Sequence[int])

                # root pose: cartesian position, quaternion orientation (w, x, y, z)
                # cart_articulation.write_root_pose_to_sim(root_pose[,env_ids])

                # Root velocity: root_velocity shape ( len(env_ids), 6), env_ids
                #cart_articulation.write_root_velocity_to_sim(root_velocity[, ...])

                # Joint State: position: ( len(env_ids), len(joint_ids)), same for vel, joint_ids: joint:indices
                # cart_articulation.write_joint_state_to_sim((position: torch.Tensor, velocity: torch.Tensor, joint_ids: Sequence[int], env_ids: Sequence[int])

                #cart_articulation.write_joint_stiffness_to_sim(stiffness[, ...])
                # cart_articulation.write_joint_damping_to_sim(damping[, ...])
                # cart_articulation.write_joint_friction_to_sim(joint_friction)

                ### SET External force
                # forces: external in body frame shape: ( len(env_ids), len(body_ids), 3)
                # body_ids, body indices to apply external wrench to
                # set external force and toques to apply on the assets bodies in their local frame
                # If function called with empty forces, function disables application of external wrench to the sim
                # asset.set_external_force_and_torque(forces=torch.zeros(0, 3), torques=torch.zeros(0, 3))
                #cart_articulation.set_external_force_and_torque(forces, torques)

                ### SET JOINT POS TARGET
                # target shape: ( len(env_ids), len(joint_ids))
                # joint_ids defaults to none for all joints
                #cart_articulation.set_joint_position_target(target: torch.Tensor, joint_ids: Sequence[int] , env_ids: Sequence[int])
                # cart_articulation.set_joint_velocity_target(tartet[,...])
                #cart_articulation.set_joint_effort_target(target[, joint_ids, ...])
                #cart_articulation.set_debug_vis((debug_vis))



            null_efforts = torch.rand_like(env.action_manager.action)

            env.step(null_efforts)
            count += 1

    # clean up env
    env.close()
    

if __name__ == "__main__":
    main()
    simulation_app.close()








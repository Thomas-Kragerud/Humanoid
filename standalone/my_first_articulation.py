#########
# LAUNCH THE SIMULATOR FIRST
#########

import argparse
from omni.isaac.lab.app import AppLauncher

# Parse arguments and launch simulator
parser = argparse.ArgumentParser(description="Demo application spawning and interacting w an articulation")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

################
# Other imports
################
import omni.isaac.lab.sim as sim_utils
import omni.isaac.core.utils.prims as prim_utils

from omni.isaac.lab_assets import CARTPOLE_CFG # Pre-defined cart-pole config
from omni.isaac.lab.sim import SimulationContext, SimulationCfg
from omni.isaac.lab.assets import Articulation, ArticulationCfg
import torch

"""
This script demonstrates how to work with articulated objects like robots in isaac sim.
An articulation is different from a rigid object because it has: 
1. Multiple connected bodies (links)
2. Joints between the bodies 
3. Actuators that can move the joints 
4. A hierarchical structure (parent-child relationship)

In this example, we use a cart-pole system which has: 
- A cart that moves along x-axis
- A pole attached to the cart that can rotate

------------
omni.isaac.lab.assets 
Framework provides structured way to handle different types of physical objects (assets) in sim environment. 
Its built around three main types of assets:
1. Rigid Objects: Single solid bodies (boxes, spheres, etc..)
2. Articulated Objects: Connected bodies with joints 
3. Deformable Objects: Soft bodies that can change shape (cloth, soft materials)

All assets inherit from AssetBase class
Each asset type has three key components:
1. Main class (eg. RigidObject)
2. Data Container (eg. RigidObjectData)
3. Configuration class (e.g RigidObjectCfg)

# Common pattern for all assets:
my_asset = AssetType(cfg)       # Initialization 
my_asset.set_something()        # Set state/parameters
my_asset.write_data_to_sim()    # Apply to simulation 
sim_context.step()              # Step simulation 
my_asset.update(dt)             # Update internal state 

>Key pattern is the separation between setting values (which only updates internal buffers) 

>All assets support multiple instances in different environments
asset.reset(env_ids=[0, 1, 4])
asset.write_root_pose_to_sim(poses, env_id=[0,1,7])

>Initial state of an asset is defined w.r.t its local environment frame. 
 This then needs to be transformed into the global simulation frame when resetting the assets state.
"""

# Q Kinematic chain?

def design_scene() -> tuple[dict, list[list[float]]]:
    """ Creates the sim scene w ground, lights and robots"""

    # Create ground and lights
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)

    light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    # Create spawn points for robots
    origins = [
        [0.0, 0.0, 0.0],
        [-3.0, 0.0, 0.0],
    ]

    # Create Xform prims for organisation
    for i, origin in enumerate(origins):
        prim_utils.create_prim(f"/World/Origin{i}", "Xform", translation=origin)

    # Create the cart-pole robots
    # 1. Copy the pre-defined configuration
    cartpole_cfg: ArticulationCfg = CARTPOLE_CFG.copy()
    # 2. Set the path where robots will be spawned (uses regex for multiple instances)
    cartpole_cfg.prim_path = "/World/Origin.*/CartPole"
    # 3. Create the articulation object
    cartpole_asset = Articulation(cfg=cartpole_cfg)

    return {"cartpole": cartpole_asset}, origins


def run_simulator(
        sim: SimulationContext,
        entities: dict[str, Articulation],
        origins: torch.Tensor,
) -> None:
    """
    Runs the simulation loop with robot control

    """
    # Get reference to robot
    robot = entities["cartpole"]
    sim_dt = sim.get_physics_dt()
    count = 0

    while simulation_app.is_running():
        # Reset robots every 500 steps
        if count % 500 == 0:
            count = 0

            # 1. Reset root state (position/orientation of base link)
            root_state = robot.data.default_root_state.clone()
            #  Add offset to place robot at their origins
            root_state[:, :3] += origins
            robot.write_root_state_to_sim(root_state)

            # 2. Reset joint state
            joint_pos = robot.data.default_joint_pos.clone()
            joint_vel = robot.data.default_joint_vel.clone()
            #  Add some random noice to joint positions
            joint_pos += torch.rand_like(joint_pos) * 0.1
            robot.write_joint_state_to_sim(joint_pos, joint_vel)

            # 3. Clear internal buffers
            robot.reset()
            print("[Info]: Resetting robot state...")

        # Generate and apply random control commands
        # 1. Create random joint efforts (forces/torques)
        efforts = torch.rand_like(robot.data.joint_pos) * 5.0
        # 2. Set the effort targets
        robot.set_joint_effort_target(efforts)
        # 3. Write commands to simulation
        robot.write_data_to_sim()

        # Step physics simulation
        sim.step()
        count += 1

def main() -> None:
    """ Main function to set up and run simulation """

    # Initialize simulation
    sim_cfg = SimulationCfg(dt=0.01, device=args_cli.device)
    sim = SimulationContext(sim_cfg) # create physics engine and sets up the sim environment
    sim.set_camera_view((2.5, 0.0, 4.0), (0.0, 0.0, 2.0))

    # Create scene
    scene_entities, scene_origins = design_scene()
    # Convert origins to tensor on same device as simulation
    scene_origins = torch.tensor(scene_origins, device=sim.device)

    # Start simulation
    sim.reset()
    print("[Info]: Simulation started...")

    # Run main loop
    run_simulator(sim, scene_entities, scene_origins)

if __name__ == "__main__":
    # Run the simulation
    main()
    simulation_app.close()

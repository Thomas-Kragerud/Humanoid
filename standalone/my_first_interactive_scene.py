"""
#####
# INTERACTIVE SCENE
#####
This demonstrates how to use the InteractiveScene in Isaac Sim
By creating multiple instances of Cartpole

Key Concepts:
1. INTERACTIVE SCENE PURPOSE
    - instead of manually creating and managing individual objects in a sim
      provide unified way to manage multiple sim elements
    - Allows easy cloning of objects for multiple envs

2. MAIN COMPONENTS:
    - Non-interactive prims (ground plane, light)
    - Interactive prims (robots, articulated objects)
    - Sensors (cameras, lidars)

3. BENEFITS
    - Automatic asset spawning
    - Easy cloning for multiple envs
    - Central management of all scene elements

- InteractiveSceneCfg: Configuration class that defines what should be in the scene
- InteractiveScene: Runtime class that creates and manages the scene

> When initialising the Cfg these are the attributes:
    - num_envs
    - env_spacing
    - lazy_sensor_update: whether to update sensor only when they are accessed
    - replicate_physics: Enable/disable replication of physics schemas when using the Cloner APIs

"""


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
from omni.isaac.lab.assets import AssetBaseCfg, ArticulationCfg, Articulation
from omni.isaac.lab_assets import CARTPOLE_CFG # Pre-defined cart-pole config
from omni.isaac.lab.sim import SimulationContext

import omni.isaac.lab.sim as sim_utils
import omni.isaac.core.utils.prims as prim_utils
import torch

@configclass
class CartpoleSceneCfg(InteractiveSceneCfg):
    """
    Config class for the cartpole scene
    Defines all the elements that will be in our simulation
    """

    ## Define Ground
    # - Uses AssetBaseCfg since its non-interactive
    # - Absolute path in the scene graph
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    ## Define Lighting
    # - Also non-interactive, uses AssetBaseCfg
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(
            intensity=3000.0,
            color=(0.75, 0.75, 0.75),
        ),
    )

    ## Define CartPole Articulation
    # - Uses ArticulationCfg since it's an interactive object
    # - Uses {ENV_REGEX_NS} for automatic environment multiplication
    cartpole: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="{ENV_REGEX_NS}/CartPole")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene) -> None:
    """
    Main simulation loop that runs the physics and handles resets

    :param sim: Simulation context for physics
    :param scene: Interactive scene containing all the objects
    :return:
    """
    # Get the cartpole object from the scene
    cartpole: Articulation = scene["cartpole"]

    # Get physics timestep
    sim_dt = sim.get_physics_dt()
    count = 0

    # Main simulation loop
    while simulation_app.is_running():
        # Reset every 500 steps
        if count % 500 == 0:
            count = 0

            ## Reset robot state
            # 1. Handle root state position/orientation
            root_state = cartpole.data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins # Offset env origins to place robot correctly
            cartpole.write_root_state_to_sim(root_state)

            # 2. Reset joint states w some noice
            joint_pos = cartpole.data.default_joint_pos.clone()
            joint_vel = cartpole.data.default_joint_vel.clone()
            # add small random perturbations
            joint_pos += torch.rand_like(joint_pos) * 0.1
            cartpole.write_joint_state_to_sim(joint_pos, joint_vel)

            # Clear scene buffers
            scene.reset()

        ## Apply random control action
        #  Generate random joint efforts (forces/torques)
        efforts = torch.randn_like(cartpole.data.joint_pos) * 5.0
        cartpole.set_joint_effort_target(efforts)

        # Write all scene data to simulator
        scene.write_data_to_sim()

        # Step the physics simulation
        sim.step()

        # Update the counter
        count += 1

        # Update scene buffers with new state
        scene.update(sim_dt)


def main() -> None:
    """
    Main function to set up and run simulation
    """
    # Initialise simulation context
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)

    sim.set_camera_view((2.5, 0.0, 4.0), (0.0, 0.0, 2.0))

    # Create the scene configuration
    scene_cfg = CartpoleSceneCfg(
        num_envs=args_cli.num_envs,
        env_spacing=2.0, # Space between envs
    )

    # Create the interactive scene
    scene = InteractiveScene(scene_cfg)

    # Reset the simulator to init state
    sim.reset()
    print(f"[INFO]: Setup complete...")

    # Run the main simulation loop
    run_simulator(sim, scene)

if __name__ == "__main__":
    main()
    simulation_app.close()



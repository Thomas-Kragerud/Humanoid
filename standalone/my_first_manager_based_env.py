"""
#####
# MANAGER BASED Environments
#####

The idea is to wrap a class around many intricacies of the simulation interaction and
provide a simple interface for the user.

Core components:
- scene.InteractiveScene      - the scene that is used for simulation
- managers.ActionManagers     - the manager that handles actions
- managers.ObservationManager - tha manager that handles observations
- managers.EventManager       - the manager that schedules operations like domain randomization
"""

import argparse

from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser(description="Demo of Manager Based Environments")
parser.add_argument("--num_envs", type=int, default=128, help="Number of envs to spawn")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

################
# Other imports
################
import omni.isaac.lab.sim as sim_utils
import omni.isaac.core.utils.prims as prim_utils

from omni.isaac.lab_assets import CARTPOLE_CFG
from omni.isaac.lab.sim import SimulationContext

from colored import fg, attr
red_color = fg('red')
blue_color = fg('blue')
green_color = fg('green')
reset_color = attr('reset')

import omni.isaac.lab.envs.mdp as mdp
from omni.isaac.lab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from omni.isaac.lab.managers import (
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    SceneEntityCfg,
)

from omni.isaac.lab.utils import configclass
from omni.isaac.lab.scene import InteractiveSceneCfg, InteractiveScene
from omni.isaac.lab.assets import AssetBaseCfg, ArticulationCfg, Articulation
# from omni.isaac.lab_tasks.manager_based.classic.cartpole.cartpole_env_cfg import CartpoleSceneCfg
# my_first_interactive_scene import CartpoleSceneCfg
import torch
import math



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
    CartPole: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="{ENV_REGEX_NS}/CartPole")





# 1. Action configuration
@configclass
class ActionsCfg:
    """
    Defines the action space and control schemes for the environment
    mpd.JointEffortActionCfg -> JointEffortAction -> _asset.set_joint_effort_target(processed_actions, joint_ids)
    """
    # Define joint effort control for the cart
    joint_effots = mdp.JointEffortActionCfg(
        asset_name="CartPole",          # Name of the asset to control
        joint_names=["slider_to_cart"], # Which joints to control
        scale=5.0,                      # scaling factor for control inputs
    )

# 2. Observation configuration
@configclass
class ObservationsCfg:
    """
    Defines the observation space and processing, which are further grouped into observation group.
    Ex: for hierarchical control, we may want to define two observation groups:
        - one for low level control
        - another for the high level control

    Its assumed that all observation terms in a group have the same dimension.
    Observations are computed by managers.ObservationManager
    """

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        """
        Policy-specific observation group.
        Necessary requirement for various wrappers in Isaac Lab.

        Individual terms are define by instantiating ObservationTermCfg.
            takes inn managers.ObservationTermCfg.func: specify a functon (callable) that computes the observation term
            Includes other params for defining noise model, clipping, scaling, etc
        """

        # Individual observation terms
        # Order is preserved in the final observation vector
        # if you use name robot can simply call
        # joint_pos_rel = ObservationTermCfg(func=mdp.joint_pos_rel)  # relative joint positions
        # joint_vel_rel = ObservationTermCfg(func=mdp.joint_vel_rel) # relative joint velocities

        # However since we used name CartPole we need to specify the scne entity
        joint_pos_rel = ObservationTermCfg(
            func=mdp.joint_vel_rel,
            params = {"asset_cfg":SceneEntityCfg("CartPole")},
        )

        joint_vel_rel = ObservationTermCfg(
            func=mdp.joint_vel_rel,
            params = {"asset_cfg":SceneEntityCfg("CartPole")},
        )


        def __post_init__(self) -> None:
            """ Post-initialization configuration"""
            self.enable_corruption = False # Disable observation noice
            self.concatenate_terms = True  # Combine all terms into single tensor

    # Define the observation groups
    # Currently only one but could have multiple
    policy: PolicyCfg = PolicyCfg()


# 3. Event Configuration
@configclass
class EventsCfg:
    """
    Defines the simulation events and their handlers
    """

    ### Startup Event - Runs Once when environment is created
    add_pole_mass = EventTermCfg(
        func=mdp.randomize_rigid_body_mass,
        mode="startup", # Only runs at startup
        params={
            "asset_cfg":SceneEntityCfg("CartPole", body_names=["pole"]),
            "mass_distribution_params": (0.1, 0.5),
            "operation": "add",
        },
    )

    ### Reset Events - Runs whenever environment is reset
    # reset and randomize slider position
    reset_cart_position = EventTermCfg(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("CartPole", joint_names=["slider_to_cart"]),
            "position_range": (-1.0, 1.0),
            "velocity_range": (-0.1, 0.1),
        },
    )

    # reset and randomize pole position
    reset_pole_position = EventTermCfg(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("CartPole", joint_names=["cart_to_pole"]),
            "position_range": (-0.125 * math.pi, 0.125 * math.pi),
            "velocity_range": (-0.01 * math.pi, 0.01 * math.pi),
        },
    )


# 4. Environment Configuration
@configclass
class CartPoleEnvCfg(ManagerBasedEnvCfg):
    """
    Main environment configuration that combines all the components

    In addition to setting up:
    - scene
    - observations
    - actions
    - events
    It also handles the SimulationContext and sets up an InteractiveScene
    so if you need to set sim specific params like gravity, timestep etc. set them in __post_init__

    Design Pattern: Builder Pattern
    - construct complex envs from simpler components
    - each component can be configured independently
    """
    # Scene configuration
    scene = CartpoleSceneCfg(num_envs=128, env_spacing=2.5)

    # Component configuration
    observations = ObservationsCfg()
    actions = ActionsCfg()
    events = EventsCfg()

    def __post_init__(self) -> None:
        """ Post-initialization settings"""
        # Camera view settings
        self.viewer.eye = (4.5, 0.0, 6.0)
        self.viewer.look_at = (0.0, 0.0, 2.0)

        # simulation settings
        self.decimation = 4 # Num of control action updates @ sim det per policy dt
        # -> if sim dt is 0.01 and policy dt is 0.1, then decimation = 10, control action updated every 10 sim steps
        self.sim.dt = 0.005 # Simulation timestep (200Hz)

        # rendering settings
        self.rendering_interval = 4 # number of physics sim steps per rendering (default 1)







"""
Main execution code for the Manager-Based Environment system 
Shows initialization and running of the environment
"""

def main() -> None:
    """
    Main execution function
    """

    # Create and configure the environment
    env_cfg = CartPoleEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs

    # Initialize the environment
    # This triggers the following sequence:
    # 1. Scene creation and asset spawning
    # 2. Manager initialization (Action, Observation, Events)
    # 3. Startup events execution

    env = ManagerBasedEnv(env_cfg)

    # Run simulation loop
    count = 0
    while simulation_app.is_running():
        # Use inference mode to disable PyTorch gradients
        with torch.inference_mode():
            # Periodic resets every 300 steps
            if count % 300 == 0:
                count = 0
                # Reset triggers:
                # 1. Scene reset
                # 2. Reset events execution
                # 3. Initial observation computation
                env.reset()

            # Generate random actions
            # In practice this would come from a policy
            joint_efforts = torch.randn_like(env.action_manager.action)

            # Step the environment
            # This triggers:
            # 1. Action application
            # 2. Physics simulation
            # 3. Observation update
            # 4. Periodic events (if any)
            obs, _ = env.step(joint_efforts)

            # Monitor pole angle from first env
            print(f"{blue_color}[Env 0]: Pole joint: {obs['policy'][0][1].item()}")
            count += 1

    # Clean up env
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
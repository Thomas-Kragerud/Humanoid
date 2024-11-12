

########
# LAUNCH THE SIMULATOR FIRST
########
import argparse
from omni.isaac.lab.app import AppLauncher
# create argparse
parser = argparse.ArgumentParser(description="Creating empty scene")
parser.add_argument("--num_envs", type=int, default=2, help="Number of envs to spawn")
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


TK_LOCAL_ASSET_PATH = "/home/thomas/Assets"

main_prim = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{TK_LOCAL_ASSET_PATH}/H1/with_box/h1_minimal_w_box.usd",
        #rigid_props
        #articulation_props

    ),
    #init_state=ArticulationCfg.InitialStateCfg(pos=(), joint_pos={}),
    actuators = {
        "x_actuator": ImplicitActuatorCfg(
            joint_names_expr=["x_drive_joint"],
            effort_limit=2000.0,
            velocity_limit=5.0,
            stiffness=0.0,
            damping=100.0,
        ),
    }
)


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

    robot: ArticulationCfg = main_prim.replace(prim_path="{ENV_REGEX_NS}/main_prim")

def run(
        sim: sim_utils.SimulationContext,
        scene: InteractiveScene,
) -> None:
    robot: ArticulationCfg = scene["robot"]

    sim_dt = sim.get_physics_dt()
    count = 0
    while simulation_app.is_running():
        if count % 500 == 0:
            count = 0
            scene.reset()

        scene.write_data_to_sim()
        sim.step()
        count += 1
        scene.update(sim_dt)


def main() -> None:
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)

    scene_cfg = MySceneCfg(
        num_envs=args_cli.num_envs,
        env_spacing=2.5,
    )

    scene = InteractiveScene(scene_cfg)

    sim.reset()
    run(sim, scene)

if __name__ == "__main__":
    main()
    simulation_app.close()
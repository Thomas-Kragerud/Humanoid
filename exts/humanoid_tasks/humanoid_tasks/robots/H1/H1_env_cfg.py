
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.utils import configclass

from humanoid_tasks import mdp
from humanoid_tasks.cfg.humanoid_env_cfg import HumanoidEnvCfg
from omni.isaac.lab_assets import H1_MINIMAL_CFG  # isort: skip
from humanoid_tasks.cfg.rough_terrains_cfg import ROUGH_TERRAINS_CFG

from .H1_prim_cfg import TKH1, H1_slider

@configclass
class H1BaseEnvCfg(HumanoidEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Set robot in scene
        #self.scene.robot = H1_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot = TKH1.replace(prim_path="{ENV_REGEX_NS}/Robot")


        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None

        # Randomization
        self.events.push_robot = None
        self.events.add_base_mass = None

        # Rewards
        #self.rewards.pen_undesired_contacts = None




class H1BaseEnvCfg_PLAY(H1BaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0

        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None

        # commands
        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0) # TODO: Why set headings



@configclass
class H1FlatEnvCfg(H1BaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.curriculum.terrain_levels = None


        # larger reward for feet in the air and higher threshold
        self.rewards.feet_air_time.weight = 1.0
        self.rewards.feet_air_time.params["threshold"] = 0.6


class H1FlatEnvCfg_PLAY(H1BaseEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()

        self.curriculum.terrain_levels = None



@configclass
class H1RoughEnvCfg(H1BaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = ROUGH_TERRAINS_CFG


@configclass
class H1RoughEnvCfg_PLAY(H1BaseEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()

        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

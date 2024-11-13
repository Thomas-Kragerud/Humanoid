
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.utils import configclass

from humanoid_tasks import mdp
from humanoid_tasks.cfg.humanoid_env_cfg import HumanoidEnvCfg
from omni.isaac.lab_assets import H1_MINIMAL_CFG  # isort: skip
from humanoid_tasks.cfg.rough_terrains_cfg import ROUGH_TERRAINS_CFG
from omni.isaac.lab.sensors import ContactSensorCfg, RayCasterCfg, patterns

from .H1_prim_cfg import TKH1, H1_slider

from humanoid_tasks.cfg.humanoid_env_cfg import *

# @configclass
# class BoxTerminationsCfg(TerminationsCfg):
#     base_contact = None

@configclass
class BoxHumanoidSceneCfg(HumanoidSceneCfg):
    contact_forces = ContactSensorCfg(
        #prim_path="{ENV_REGEX_NS}/Robot/h1/.*",
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
        # update_period=0.0 # update period of sensor buffer in sec
    )

@configclass
class BoxActionsCfg(ActionsCfg):
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "left_hip_yaw",
            "left_hip_roll",
            "left_hip_pitch",
            "left_knee",
            "left_ankle",
            "right_hip_yaw",
            "right_hip_roll",
            "right_hip_pitch",
            "right_knee",
            "torso",
            "left_shoulder_pitch",
            "left_shoulder_roll",
            "left_shoulder_yaw",
            "left_elbow",
            "right_shoulder_pitch",
            "right_shoulder_roll",
            "right_shoulder_yaw",
            "right_elbow",
        ],
        scale=0.5,
        use_default_offset=True  # TODO: Whut
    )

@configclass
class H1BoxBaseEnvCfg(HumanoidEnvCfg):
    scene: BoxHumanoidSceneCfg = BoxHumanoidSceneCfg(
        num_envs=4096,
        env_spacing=2.5,
    )
    actions: BoxActionsCfg = BoxActionsCfg()
    #terminations: BoxTerminationsCfg = BoxTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()

        # Set robot in scene
        #self.scene.robot = H1_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # self.scene.robot = TKH1.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot = H1_slider.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Bc contact forces fucks up
        self.scene.contact_forces = None
        self.rewards.feet_air_time = None
        self.rewards.pen_feet_air_time = None
        self.rewards.pen_feet_slide = None
        self.terminations.base_contact = None

        self.events.physics_material = None
        self.events.reset_base

        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None

        # Randomization
        self.events.push_robot = None
        self.events.add_base_mass = None

        # Rewards
        #self.rewards.pen_undesired_contacts = None




class H1BoxBaseEnvCfg_PLAY(H1BoxBaseEnvCfg):
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
class H1BoxFlatEnvCfg(H1BoxBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.curriculum.terrain_levels = None
        print(f"\n\nActions: {self.actions}")


        # larger reward for feet in the air and higher threshold
        #self.rewards.feet_air_time.weight = 1.0
        #self.rewards.feet_air_time.params["threshold"] = 0.6


class H1BoxFlatEnvCfg_PLAY(H1BoxBaseEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()

        self.curriculum.terrain_levels = None

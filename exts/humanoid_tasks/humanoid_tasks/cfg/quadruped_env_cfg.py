
import math
from dataclasses import MISSING # allowing dataclass to check if field has been explicitly provided or not


from humanoid_tasks import mdp

from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.envs.mdp import time_out
from omni.isaac.lab.managers import (
    ObservationGroupCfg,
    EventTermCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
    CurriculumTermCfg,
    SceneEntityCfg,
)
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import ContactSensorCfg
from omni.isaac.lab.sim import (
    RigidBodyMaterialCfg,
    MdlFileCfg,
    DomeLightCfg
)
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils.noise import AdditiveGaussianNoiseCfg as GaussianNoise
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from omni.isaac.lab.utils import configclass
from warp.tests.test_hash_grid import scale


##################
# Scene Definition
##################

@configclass
class QuadrupedSceneCfg(InteractiveSceneCfg):
    """ Config class for the quadruped scene """
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        terrain_generator=None,
        max_init_terrain_level=0, # specified by num of rows in grid arrangement of sub-terrains
        collision_group=-1,
        physics_material=RigidBodyMaterialCfg(
            friction_combine_mode="multiply", # determines the way friction will be combined during collisions [average, min, multiply, max]
            restitution_combine_mode="multiply", # determines the way restitution coefficient will be combined
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        # Configure MDL - Material Definition Language
        visual_material=MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True, # determines how UVW texture coordinates are applied to material, UVW is a map that tells sim how to wrap a 2D texture to 3D surface object
            texture_scale=(0.25, 0.25)  # the scale factor in two dimensions (horizontal and vertical scaling)
        ),
        debug_vis=False, # visualization of terrain origins
    )

    ### Skylight
    light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=DomeLightCfg(
            intensity=750.0,
            color=(0.9, 0.9, 0.9),
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    ### Quadroped robot
    robot: ArticulationCfg = MISSING

    ### Contact sensor
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=4, # num past frames to store in sensor buffers
        track_air_time=True, # track time between contacts
        update_period=0.0, # update period of sensor buffer in sec
    )

##############
# MDP settings
##############

@configclass
class CommandsCfg:
    """ Command terms for the MDP """

    # config for uniform velocity command generator
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        heading_command=False, # heading cmd or angular velocity cmd, if True ang vel cmd computed from the heading error, else ang vel cmd sampled uniformly
        rel_standing_envs=0.02, # sampled prob of envs that should be standing still
        rel_heading_envs=0.0, # sampled prob of envs robot follow heading base ang vel - others follow ang velocity cmd
        debug_vis=True,
        resampling_time_range=(10.0, 10.0), # time before commands are changed in sec
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.5, 1.5),
            lin_vel_y=(1.0, 1.0),
            ang_vel_z=(-math.pi / 2, math.pi / 2),
        ),
    )


@configclass
class ActionsCfg:
    """ Action specification term for the MDP """
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.2,
        use_default_offset=True,
    )

@configclass
class ObservationsCfg:
    """ Observation specification term for the MDP """
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        """ Observation for policy group """
        # velocity command
        vel_command = ObservationTermCfg(func=mdp.base_lin_vel, noise=GaussianNoise(mean=0.0, std=0.05))

        # robot base measurements
        base_lin_vel = ObservationTermCfg(func=mdp.base_lin_vel, noise=GaussianNoise(mean=0.0, std=0.05))
        base_ang_vel = ObservationTermCfg(func=mdp.base_ang_vel, noise=GaussianNoise(mean=0.0, std=0.05))
        proj_gravity = ObservationTermCfg(func=mdp.projected_gravity, noise=GaussianNoise(mean=0.0, std=0.025))

        # robot joint measurements
        joint_pos = ObservationTermCfg(func=mdp.joint_pos_rel, noise=GaussianNoise(mean=0.0, std=0.01))

        # last action
        last_action = ObservationTermCfg(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True # terms in the group are corrupted by adding noise
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

@configclass
class EventsCfg:
    """ Configuration for events """

    # startup
    add_base_mass = EventTermCfg(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-1.0, 2.0),
            "operation":"add",
        },
    )

    # reset
    reset_robot_base = EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "yaw": (-3.14, 3.14)
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTermCfg(
        func=mdp.reset_joints_by_scale, # samples random values from the given ranges and scales the default joint positions
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
        }
    )

    # interval
    push_robot = EventTermCfg(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "yaw": (-math.pi / 6, math.pi / 6),
            },
        },
    )

@configclass
class RewardsCfg:
    """ Configuration for rewards """

    # rewards
    rew_lin_vel_xy = RewardTermCfg(
        func=mdp.track_lin_vel_xy_exp,
        weight=2.0,
        params={
            "command_name":"base_velocity",
            "std": math.sqrt(0.25),
        },
    )

    rew_ang_vel_xy = RewardTermCfg(
        func=mdp.track_ang_vel_z_exp,
        weight=1.5,
        params={
            "command_name":"base_velocity",
            "std": math.sqrt(0.25),
        }
    )

    # penalization
    pen_joint_deviation = RewardTermCfg(
        func=mdp.joint_deviation_l1,
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]), # specify name of scene entity that is queried from InteractiveScene
        },
    )

    pen_feet_slide = RewardTermCfg(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
        },
    )

    pen_undesired_contacts = RewardTermCfg(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*THIGH"), # TODO: What is THIGH
            "threshold": 1.0,
        }
    )

    pen_lin_vel_z = RewardTermCfg(func=mdp.lin_vel_z_l2, weight=-1.0)
    pen_ang_vel_xy = RewardTermCfg(func=mdp.ang_vel_xy_l2, weight=-0.05)
    pen_action_rate = RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01)
    pen_joint_accel = RewardTermCfg(func=mdp.joint_acc_l2, weight=-1.0e-6)
    pen_joint_powers = RewardTermCfg(func=mdp.joint_powers_l1, weight=-5.0e-4)
    pen_flat_orientation = RewardTermCfg(func=mdp.flat_orientation_l2, weight=-2.5)


@configclass
class TerminationsCfg:
    """ Termination for the MDP """
    time_out = TerminationTermCfg(
        func=mdp.time_out,
        time_out=True,
    )
    base_contact = TerminationTermCfg(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"),
            "threshold": 1.0,
        }
    )

@configclass
class CurriculumCfg:
    """ Curriculum terms for the MDP """
    terrain_levels = CurriculumTermCfg(func=mdp.terrain_levels_vel) # curriculum based on distance walked


@configclass
class QuadrupedEnvCfg(ManagerBasedRLEnvCfg):
    """ Configuration for the quadruped environment """

    # Scene settings
    scene: QuadrupedSceneCfg = QuadrupedSceneCfg(
        num_envs=1024,
        env_spacing=2.0,
    )

    # basic
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()

    # MDP
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 8 # control cation updated every 8 sim step
        self.episode_length_s = 20.0
        self.sim.render_interval = 2 * self.decimation
        # simulation settings
        self.sim.dt = 1 / 400.0
        self.seed = 42







import gymnasium as gym

from humanoid_tasks.agents.rsl_rl_cfg import QuadrupedPPORunnerCfg

from . import go2_env_cfg

### Create PPO Runner for RSL-RL
# manages training loop and interaction between agent and env

go2_test_runner = QuadrupedPPORunnerCfg()
go2_test_runner.experiment_name = "go2_test"

go2_blind_flat_runner_cfg = QuadrupedPPORunnerCfg()
go2_blind_flat_runner_cfg.experiment_name = "go2_blind_flat"

go2_blind_rough_runner_cfg = QuadrupedPPORunnerCfg()
go2_blind_rough_runner_cfg.experiment_name = "go2_blind_rough"

go2_blind_stairs_runner_cfg = QuadrupedPPORunnerCfg()
go2_blind_stairs_runner_cfg.experiment_name = "go2_blind_stairs"


### Register gym evn

############################
# Go2 Test Environment
############################
gym.register(
    id="testGo2",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": go2_env_cfg.Go2BlindRoughEnvCfg,
        "rsl_rl_cfg_entry_point": go2_test_runner,
    },
)

gym.register(
    id="testGo2Play",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": go2_env_cfg.Go2BlindRoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": go2_test_runner,
    },
)

############################
# Go2 Blind Flat Environment
############################

gym.register(
    id="Isaac-Quadruped-Go2-Blind-Flat-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": go2_env_cfg.Go2BlindFlatEnvCfg,
        "rsl_rl_cfg_entry_point": go2_blind_flat_runner_cfg,
    }
)

gym.register(
    id="Isaac-Quadruped-Go2-Blind-Flat-Play-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedPlayEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": go2_env_cfg.Go2BlindFlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": go2_blind_flat_runner_cfg,
    },
)


#############################
# Go2 Blind Rough Environment
#############################

gym.register(
    id="Isaac-Quadruped-Go2-Blind-Rough-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": go2_env_cfg.Go2BlindRoughEnvCfg,
        "rsl_rl_cfg_entry_point": go2_blind_rough_runner_cfg,
    },
)

gym.register(
    id="Isaac-Quadruped-Go2-Blind-Rough-Play-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": go2_env_cfg.Go2BlindRoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": go2_blind_rough_runner_cfg,
    },
)

##############################
# Go2 Blind Stairs Environment
##############################

gym.register(
    id="Isaac-Quadruped-Go2-Blind-Stairs-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": go2_env_cfg.Go2BlindStairsEnvCfg,
        "rsl_rl_cfg_entry_point": go2_blind_stairs_runner_cfg,
    },
)

gym.register(
    id="Isaac-Quadruped-Go2-Blind-Stairs-Play-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": go2_env_cfg.Go2BlindStairsEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": go2_blind_stairs_runner_cfg,
    },
)

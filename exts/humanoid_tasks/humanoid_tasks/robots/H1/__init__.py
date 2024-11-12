# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import H1_env_cfg
from humanoid_tasks.agents.rsl_rl_cfg import H1FlatPPORunnerCfg, H1RoughPPORunnerCfg

##
# Register Gym environments.
##

h1_rough_runner_cfg = H1RoughPPORunnerCfg()
h1_rough_runner_cfg.experiment_name = "h1_velocity_rough"

h1_flat_runner_cfg = H1RoughPPORunnerCfg()
h1_flat_runner_cfg.experiment_name="h1_velocity_flat"


gym.register(
    id="h1-velocity-rough",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": H1_env_cfg.H1RoughEnvCfg,
        "rsl_rl_cfg_entry_point": h1_rough_runner_cfg,
        #"skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
    },
)

gym.register(
    id="h1-velocity-rough-play",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": H1_env_cfg.H1RoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": h1_rough_runner_cfg,
    },
)

gym.register(
    id="h1-velocity-flat",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": H1_env_cfg.H1FlatEnvCfg,
        "rsl_rl_cfg_entry_point": h1_flat_runner_cfg,
    },
)

gym.register(
    id="h1-velocity-flat-play",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": H1_env_cfg.H1FlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": h1_flat_runner_cfg,
    }
)

# gym.register(
#     id="Isaac-Velocity-Rough-H1-Play-v0",
#     entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": H1_env_cfg.H1RoughEnvCfg_PLAY,
#         "rsl_rl_cfg_entry_point": h1_rough_runner_cfg,
#         #"skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
#     },
# )


#
# gym.register(
#     id="Isaac-Velocity-Flat-H1-v0",
#     entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": H1_env_cfg.H1FlatEnvCfg,
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:H1FlatPPORunnerCfg",
#         "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
#     },
# )
#
#
# gym.register(
#     id="Isaac-Velocity-Flat-H1-Play-v0",
#     entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": H1_env_cfg.H1FlatEnvCfg_PLAY,
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:H1FlatPPORunnerCfg",
#         "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
#     },
# )

"""
License: MIT License
Copyright (c) 2024, Felipe Mohr Santos

Sub module with implementations of manager terms
Functions can be provided to different managers that are responsible for the different aspects of the MDP
- Observation
- Reward
- Termination
- Actions
- Events
- Curriculum

The terms are defined under the envs module bc they are used to define the environment. However they are not part of
the environment directly, but used to define the environment through their managers.

"""

from omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp import *  # noqa: F401, F403

from .curriculums import *  # noqa: F401, F403
from .observations import *  # noqa: F401, F403
from .rewards import *  # noqa: F401, F403
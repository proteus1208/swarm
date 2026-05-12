"""Env factory for SB3 workers.

Imported by SubprocVecEnv child processes on Windows (spawn). Must not import
stable_baselines3 or torch — only gymnasium + swarm sim stack.
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from gymnasium.wrappers import RecordEpisodeStatistics

from swarm.constants import SIM_DT
from swarm.utils.env_factory import make_env
from swarm.validator.task_gen import random_task


def make_training_env(
    seed: int,
    *,
    gui: bool = False,
    path_monitor: bool = False,
):
    def _init():
        task = random_task(sim_dt=SIM_DT, seed=seed)
        env = make_env(task, gui=gui, path_monitor=path_monitor)
        env = RecordEpisodeStatistics(env)
        return env

    return _init

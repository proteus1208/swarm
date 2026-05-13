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
        last_error = None

        for attempt in range(200):
            task_seed = seed + attempt * 10007

            try:
                task = random_task(sim_dt=SIM_DT, seed=task_seed)

                env = make_env(
                    task,
                    gui=False,
                    path_monitor=path_monitor,
                )

                env = RecordEpisodeStatistics(env)
                return env

            except RuntimeError as e:
                msg = str(e)

                if "unable to find collision-free platform position" in msg:
                    last_error = e
                    print(
                        f"[env retry] base_seed={seed} "
                        f"attempt={attempt + 1} "
                        f"task_seed={task_seed} "
                        f"error={e}"
                    )
                    continue

                raise

        raise RuntimeError(
            f"Failed to create environment after 200 retries. "
            f"base_seed={seed}. Last error: {last_error}"
        )

    return _init
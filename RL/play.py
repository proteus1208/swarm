#!/usr/bin/env python3

import argparse
from pathlib import Path
import sys
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import time

# Ensure we import the local `swarm/` package (repo root),
# not an unrelated `swarm` installed in site-packages.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from swarm.utils.env_factory import make_env
from swarm.validator.task_gen import random_task
from swarm.constants import SIM_DT, SPEED_LIMIT
import numpy as np
from gym_pybullet_drones.utils.enums import ActionType
import pybullet as p
import torch
torch.set_num_threads(10)

def main(seed):
    task = random_task(sim_dt=SIM_DT, seed=seed)
    env = make_env(task, gui=True)

    model = PPO.load(
        "swarm/submission_template/ppo_policy.zip",
    )
    obs, _ = env.reset(seed=task.map_seed)
    t_sim = 0.0
    act_lo = env.action_space.low.flatten()
    act_hi = env.action_space.high.flatten()
    while t_sim < task.horizon:
        try:
            raw, _ = model.predict(obs, deterministic=True)
            if raw is None:
                raw = np.zeros(5, dtype=np.float32)
        except Exception as e:
            print("Prediction error:", e)
            raw = np.zeros(5, dtype=np.float32)

        act = np.clip(np.asarray(raw, dtype=np.float32).flatten(), act_lo, act_hi)
        if getattr(env, "ACT_TYPE", None) == ActionType.VEL:
            norm = max(float(np.linalg.norm(act[:3])), 1e-6)
            act[:3] *= min(1.0, float(SPEED_LIMIT) / norm)
            act = np.clip(act, act_lo, act_hi)

        obs, _, terminated, truncated, info = env.step(act[None, :])
        t_sim += float(SIM_DT)
        drone_pos = env._getDroneStateVector(0)[:3]

        p.resetDebugVisualizerCamera(
            cameraDistance=2.0,        # zoom (smaller = closer)
            cameraYaw=45,              # horizontal angle
            cameraPitch=-30,           # vertical angle
            cameraTargetPosition=drone_pos
        )
        env.render()

        if terminated or truncated:
            success = bool(info.get("success", False))
            break
    env.close()

if __name__ == "__main__":
    main(int(time. time()))

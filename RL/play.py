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

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("OpenCV not installed. Drone camera window will be disabled.")
    print("Install with: pip install opencv-python")


def get_drone_camera_image(
    env,
    drone_id=0,
    width=320,
    height=240,
    fov=80,
    near=0.02,
    far=50.0,
):
    """
    Render an onboard camera from the drone's point of view.

    Assumes:
    - drone position is state[0:3]
    - drone quaternion is state[3:7] as [x, y, z, w]
    - drone forward direction is body +X
    - drone up direction is body +Z
    """

    state = env._getDroneStateVector(drone_id)

    drone_pos = np.array(state[0:3], dtype=np.float32)
    drone_quat = np.array(state[3:7], dtype=np.float32)

    rot = np.array(p.getMatrixFromQuaternion(drone_quat)).reshape(3, 3)

    forward = rot @ np.array([1.0, 0.0, 0.0])
    up = rot @ np.array([0.0, 0.0, 1.0])

    camera_pos = drone_pos + 0.08 * forward
    camera_target = camera_pos + forward

    view_matrix = p.computeViewMatrix(
        cameraEyePosition=camera_pos,
        cameraTargetPosition=camera_target,
        cameraUpVector=up,
    )

    projection_matrix = p.computeProjectionMatrixFOV(
        fov=fov,
        aspect=float(width) / float(height),
        nearVal=near,
        farVal=far,
    )

    _, _, rgba, _, _ = p.getCameraImage(
        width=width,
        height=height,
        viewMatrix=view_matrix,
        projectionMatrix=projection_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL,
    )

    img = np.asarray(rgba, dtype=np.uint8).reshape(height, width, 4)
    rgb = img[:, :, :3]

    return rgb


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

        # View 2: current PyBullet GUI camera following the drone
        p.resetDebugVisualizerCamera(
            cameraDistance=2.0,
            cameraYaw=45,
            cameraPitch=-30,
            cameraTargetPosition=drone_pos,
        )

        # View 1: onboard drone camera in a separate OpenCV window
        if HAS_CV2:
            drone_cam_rgb = get_drone_camera_image(env, drone_id=0)

            # OpenCV expects BGR, PyBullet gives RGB
            drone_cam_bgr = cv2.cvtColor(drone_cam_rgb, cv2.COLOR_RGB2BGR)

            cv2.imshow("Drone Camera View", drone_cam_bgr)

            # Press q in the drone camera window to quit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        env.render()

        if terminated or truncated:
            success = bool(info.get("success", False))
            print("Success:", success)
            break

    env.close()

    if HAS_CV2:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main(int(time.time()))
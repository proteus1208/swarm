# swarm/validator/reward.py
"""Reward function for flight missions.

Current active reward is a pretraining reward:

- speed in the drone forward direction
- active horizontal movement
- yaw facing goal
- movement aligned with yaw
- small z change
- safe/stable attitude

Important:
- The active reward does NOT use ``state[10:13]`` as velocity.
- Velocity is estimated from ``state_history`` position differences.

Old mission-shaping reward logic is kept as commented reference inside
``flight_reward``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import itertools
import sys

import numpy as np

if TYPE_CHECKING:
    from swarm.protocol import MapTask

from swarm.constants import (
    HOVER_SEC,
    REWARD_W_SAFETY,
    REWARD_W_SUCCESS,
    REWARD_W_TIME,
    SAFETY_DISTANCE_DANGER,
    SAFETY_DISTANCE_SAFE,
    SPEED_LIMIT,
    TYPE_6_SAFETY_DISTANCE_SAFE,
)


_spinner = itertools.cycle("|/-\\")


def spin_print(*args):
    prefix = next(_spinner)
    msg = " ".join(str(a) for a in args)

    sys.stdout.write(f"\r{prefix} {msg}")
    sys.stdout.flush()


SAFETY_DISTANCE_SAFE_BY_TYPE = {
    6: TYPE_6_SAFETY_DISTANCE_SAFE,
}

MAX_HZ = 60
MAX_FRAMES = 3000
MAX_SPEED = 3.0
DECEL_RADIUS = 1.5

# New pretraining constants
HISTORY_VEL_FRAMES = 10
ACTIVE_SPEED_FULL = 1.2
MAX_Z_SPEED_SMALL_CHANGE = 0.35
MOVEMENT_EPS = 1e-8

__all__ = ["flight_reward"]


def scaled_tanh(x, lower, upper):
    return lower + (upper - lower) * ((np.tanh(x) + 1) / 2)


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    """Clamp *value* to the inclusive range [*lower*, *upper*]."""
    return max(lower, min(upper, value))


def _calculate_target_time(task: "MapTask") -> float:
    """Calculate target time based on distance and 6% buffer."""
    start_pos = np.array(task.start)
    goal_pos = np.array(task.goal)
    distance = np.linalg.norm(goal_pos - start_pos)

    min_time = (distance / SPEED_LIMIT) + HOVER_SEC
    return min_time * 1.06


def _calculate_safety_term(
    min_clearance: float,
    collision: bool,
    challenge_type: int = 0,
) -> float:
    """Calculate safety term based on minimum obstacle clearance."""
    if collision:
        return 0.0

    safe = SAFETY_DISTANCE_SAFE_BY_TYPE.get(challenge_type, SAFETY_DISTANCE_SAFE)

    if min_clearance >= safe:
        return 1.0

    if min_clearance <= SAFETY_DISTANCE_DANGER:
        return 0.0

    return (min_clearance - SAFETY_DISTANCE_DANGER) / (
        safe - SAFETY_DISTANCE_DANGER
    )


def disScore(s, e, p):
    """
    Same logic preserved:
    - m1: proximity-to-start signal
    - m2: proximity-to-end signal
    - path consistency penalty unchanged
    - fully 3D compatible
    """
    s = np.asarray(s, dtype=np.float32)
    e = np.asarray(e, dtype=np.float32)
    p = np.asarray(p, dtype=np.float32)

    dist_total = np.linalg.norm(e - s)
    if dist_total < 1e-8:
        return 0.0

    dist_to_s = np.linalg.norm(p - s)
    dist_to_e = np.linalg.norm(p - e)

    m1 = 1.0 - dist_to_s / dist_total
    m2 = 1.0 - dist_to_e / dist_total

    if m2 <= 0.0:
        return 0.0

    path_penalty = np.exp(
        -6.0 * abs(dist_to_s + dist_to_e - dist_total)
    )

    reward = (m1 + m2) * m2 * path_penalty

    return float(np.clip(reward, 0.0, 1.0))


def forwardDirectionReward(current, endpoint, yaw):
    current = np.array(current)
    endpoint = np.array(endpoint)

    direction = endpoint - current

    target_yaw = np.arctan2(direction[1], direction[0])

    angle_error = target_yaw - yaw
    angle_error = np.arctan2(np.sin(angle_error), np.cos(angle_error))

    normalized_error = abs(angle_error) / np.pi

    reward = (1.0 - normalized_error) ** 2

    return float(np.clip(reward, 0.0, 1.0))


def forwardScore(e, p, v, k=8.0):
    e = np.asarray(e, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)

    dir_vec = e - p
    dist = np.linalg.norm(dir_vec)
    speed = np.linalg.norm(v)

    if dist < 1e-8:
        return 0.0

    if speed < 1e-8:
        return 0.0

    dir_unit = dir_vec / dist
    vel_unit = v / speed

    dot = np.clip(np.dot(dir_unit, vel_unit), -1.0, 1.0)
    cross_norm = np.linalg.norm(np.cross(dir_unit, vel_unit))

    alpha = np.arctan2(cross_norm, dot) / np.pi

    direction_reward = np.exp(-k * alpha**2)

    speed_factor = 1.0

    score = direction_reward * speed_factor

    return float(np.clip(score, 0.0, 1.0))


def progressScore(start, goal, state_history):
    start = np.asarray(start, dtype=np.float32)
    goal = np.asarray(goal, dtype=np.float32)

    if state_history is None or state_history.shape[0] < 2:
        return 0.0

    pos_now = state_history[-1, 0:3]
    pos_prev = state_history[-2, 0:3]

    step_vec = pos_now - pos_prev
    goal_vec = goal - pos_now

    step_norm = np.linalg.norm(step_vec) + 1e-6
    goal_norm = np.linalg.norm(goal_vec) + 1e-6

    direction_cos = np.dot(step_vec, goal_vec) / (step_norm * goal_norm)

    dist_now = np.linalg.norm(goal - pos_now)
    dist_prev = np.linalg.norm(goal - pos_prev)

    distance_delta = dist_prev - dist_now

    max_dist = 3.0 / 50.0

    r_progress = direction_cos + (distance_delta / max_dist)

    path_vec = goal - start
    proj_now = np.dot(pos_now - goal, path_vec)

    if proj_now > 0:
        r_progress -= 1.0

    r_near = 1.0 / (dist_now ** (2 / np.e) + 1.0)

    return float(r_progress + r_near)


def stabilityScore(state):
    roll, pitch = state[7], state[8]
    roll_rate, pitch_rate = state[13], state[14]

    tilt_mag = np.sqrt(roll**2 + pitch**2)

    x1 = 0.2

    r_tilt = (
        1 - tilt_mag / 5
        if tilt_mag < x1
        else np.exp(-1.5 * (tilt_mag - x1)) - 0.04
    )

    attitude_penalty = (
        (0.3 * pitch + pitch_rate) ** 2
        + (0.3 * roll + roll_rate) ** 2
    )

    score = r_tilt - 0.0 * attitude_penalty

    return float(np.clip(score, 0, 1))


def z_score(end, current):
    target_z = end[2]
    cur_z = current[2]

    top_z = target_z + 0.2
    bottom_z = target_z - 0.2

    if cur_z > top_z:
        return float(np.exp(-0.5 * (cur_z - top_z)))

    if cur_z < bottom_z:
        return float(np.exp(-0.5 * (bottom_z - cur_z)))

    return 1.0


def vibrationScore_jitter(history):
    if history is None or len(history) < 3:
        return 1.0

    history = np.asarray(history)

    vectors = []
    for i in range(1, len(history)):
        vectors.append(history[i] - history[i - 1])

    jitter = 0.0
    max_jitter = 0.0

    for i in range(1, len(vectors)):
        prev_vec = vectors[i - 1]
        curr_vec = vectors[i]

        jitter += np.linalg.norm(curr_vec - prev_vec)

        max_jitter += np.linalg.norm(prev_vec) + np.linalg.norm(curr_vec)

    if max_jitter <= 1e-8:
        return 1.0

    score = 1 - jitter / max_jitter

    return float(np.clip(score, 0.0, 1.0))


def vibrationScore_len(history):
    if history is None or len(history) < 3:
        return 1.0

    total_length = 0.0
    for i in range(1, len(history)):
        total_length += np.linalg.norm(history[i] - history[i - 1])

    if total_length <= 1e-8:
        return 1.0

    straight_length = np.linalg.norm(history[-1] - history[0])

    score = straight_length / total_length

    return float(np.clip(score, 0.0, 1.0))


def outputVibrationScore(state_history):
    return vibrationScore_jitter(state_history[:, 16:20])


def velocityVibrationScore(state_history):
    return vibrationScore_jitter(state_history[:, 10:13])


def quaternionHistoryScore(state_history):
    return vibrationScore_jitter(state_history[:, 3:7])


def positionVibrationScore(state_history):
    return vibrationScore_len(state_history[:, 0:3])


# Position            : state[0:3]
# Orientation (quat)  : state[3:7]
# Orientation RPY     : state[7:10]
# Linear Velocity     : state[10:13]
# Angular Velocity    : state[13:16]
# last_clipped_action : state[16:20]


def r_sum(items):
    # Each entry:
    # (name, weight, reward_value, color, linestyle, linewidth, orderUpDependencies)
    weight_sum = sum(item[1] for item in items)

    if weight_sum <= 1e-8:
        return 0.0, {}

    result = {}

    for name, w, val, color, linestyle, lw, orderUpDependencies in items:
        result[name] = {
            "v": (w * val) / weight_sum,
            "value": _clamp(val, -0.1, 1.1),
            "label": name,
            "weight": w,
            "color": color,
            "linestyle": linestyle,
            "linewidth": lw,
        }

    # Old dependency logic kept as commented reference.
    # for name, w, val, color, linestyle, lw, orderUpDependencies in items:
    #     for dependency in orderUpDependencies:
    #         rate = np.maximum(1.0, result[name]["v"] / result[dependency]["v"])
    #         result[name]["v"] = result[name]["v"] * rate
    #         result[name]["value"] = result[name]["value"] * rate

    return sum(r["v"] for r in result.values()), result


def r_mult(items):
    res = 1
    for item in items:
        res *= item[0] * item[1]
    return res


# ---------------------------------------------------------------------
# New pretraining helper functions
# ---------------------------------------------------------------------


def history_velocity(
    state_history,
    frames: int = HISTORY_VEL_FRAMES,
    hz: float = MAX_HZ,
):
    """Estimate velocity from state_history positions.

    Important:
    This does NOT use state[10:13].
    """
    if state_history is None:
        return np.zeros(3, dtype=np.float64)

    history = np.asarray(state_history, dtype=np.float64)

    if history.ndim != 2 or history.shape[0] < 2:
        return np.zeros(3, dtype=np.float64)

    frames = min(frames, history.shape[0] - 1)

    if frames <= 0:
        return np.zeros(3, dtype=np.float64)

    pos_now = history[-1, 0:3]
    pos_prev = history[-1 - frames, 0:3]

    measured_velocity = (pos_now - pos_prev) * (hz / frames)

    if not np.all(np.isfinite(measured_velocity)):
        return np.zeros(3, dtype=np.float64)

    return measured_velocity


def yaw_forward_xy(yaw: float):
    return np.array([np.cos(yaw), np.sin(yaw)], dtype=np.float64)


def yaw_to_goal_score(current, goal, yaw):
    current = np.asarray(current, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)

    direction_xy = goal[0:2] - current[0:2]

    if np.linalg.norm(direction_xy) < MOVEMENT_EPS:
        return 1.0

    return forwardDirectionReward(current, goal, yaw)


def moving_forward_score(measured_velocity, yaw):
    """Reward movement direction matching yaw direction.

    Uses measured_velocity from history.
    """
    vel_xy = np.asarray(measured_velocity[0:2], dtype=np.float64)
    speed_xy = np.linalg.norm(vel_xy)

    if speed_xy < MOVEMENT_EPS:
        return 0.0

    vel_dir = vel_xy / speed_xy
    forward_dir = yaw_forward_xy(yaw)

    alignment = np.dot(vel_dir, forward_dir)

    return float(np.clip(alignment, 0.0, 1.0))


def active_speed_score(measured_velocity):
    """Reward active horizontal movement."""
    speed_xy = np.linalg.norm(measured_velocity[0:2])

    return float(np.clip(speed_xy / ACTIVE_SPEED_FULL, 0.0, 1.0))


def speed_to_forward_score(measured_velocity, yaw):
    """Reward speed in the drone's forward direction.

    Uses measured_velocity from history.
    Full score at ACTIVE_SPEED_FULL forward speed.
    """
    vel_xy = np.asarray(measured_velocity[0:2], dtype=np.float64)
    forward_dir = yaw_forward_xy(yaw)

    forward_speed = np.dot(vel_xy, forward_dir)

    return float(np.clip(forward_speed / ACTIVE_SPEED_FULL, 0.0, 1.0))


def small_z_change_score(measured_velocity):
    """Reward small vertical speed."""
    z_speed = abs(float(measured_velocity[2]))

    return float(np.clip(1.0 - z_speed / MAX_Z_SPEED_SMALL_CHANGE, 0.0, 1.0))


def pretraining_safe_score(state, min_clearance=None):
    """Safety score for pretraining."""
    attitude_safe = float(stabilityScore(state))

    if min_clearance is None:
        return attitude_safe

    clearance_safe = _calculate_safety_term(
        min_clearance=min_clearance,
        collision=False,
    )

    return float(min(attitude_safe, clearance_safe))


def add_debug_row(
    result,
    name,
    value,
    color="gray",
    linestyle="dashed",
    linewidth=1.0,
):
    result[name] = {
        "v": float(value),
        "value": float(value),
        "label": name,
        "weight": 0.0,
        "color": color,
        "linestyle": linestyle,
        "linewidth": linewidth,
    }


def flight_reward(
    success: bool,
    t: float,
    horizon: float,
    task: Optional["MapTask"] = None,
    *,
    min_clearance: Optional[float] = None,
    collision: bool = False,
    w_success: float = REWARD_W_SUCCESS,
    w_t: float = REWARD_W_TIME,
    w_safety: float = REWARD_W_SAFETY,
    legitimate_model: bool = True,
    state_history: Optional[np.ndarray] = None,
    state: Optional[np.ndarray] = None,
) -> tuple[float, object]:

    if collision:
        return 0.0, {}

    measured_velocity = history_velocity(state_history)

    if success:
        # Important:
        # Do not use state[10:13] here.
        # Stop score also uses history-measured velocity.
        speed = np.linalg.norm(measured_velocity)
        stop_score = 1.0 - np.clip(speed / MAX_SPEED, 0.0, 1.0)

        reward = 0.5 + 0.5 * stop_score

        return float(reward), {
            "success": {
                "v": 1.0,
                "value": 1.0,
                "weight": 0.0,
                "color": "black",
                "linestyle": "solid",
                "linewidth": 0.5,
            },
            "stop": {
                "v": float(stop_score),
                "value": float(stop_score),
                "weight": 0.0,
                "color": "yellow",
                "linestyle": "solid",
                "linewidth": 2.0,
            },
            "total": {
                "v": float(reward),
                "value": float(reward),
                "weight": 0.0,
                "color": "orange",
                "linestyle": "solid",
                "linewidth": 4.0,
            },
        }

    if state is None or state_history is None or task is None:
        return 0.0, {}

    state_history = np.asarray(state_history, dtype=np.float64)

    if state_history.ndim != 2 or state_history.shape[0] < 2:
        return 0.0, {}

    currentPos = state[0:3]
    currentYaw = state[9]

    startPos = np.array(task.start)
    endPos = np.array(task.goal)

    # ------------------------------------------------------------------
    # OLD LOGIC KEPT AS COMMENTED REFERENCE
    # ------------------------------------------------------------------
    # currentVel = state[10:13]
    # currentAngVel = state[13:16]
    # currentRoll = state[7]
    # currentPitch = state[8]
    # currentQuat = state[3:7]
    #
    # disSE = np.linalg.norm(endPos - startPos)
    # disCE = np.linalg.norm(endPos - currentPos)
    #
    # velocity = 0.0
    # S = MAX_SPEED
    # frames = 10
    # if state_history.shape[0] > frames:
    #     displacement = np.linalg.norm(
    #         state_history[-1, 0:3] - state_history[-1 - frames, 0:3]
    #     )
    #
    #     velocity = (displacement / frames) * MAX_HZ
    #
    # r_success = 1.0 if success else 0.0
    # r_safe = stabilityScore(state)
    # r_z = z_score(endPos, currentPos)
    # r_distance = np.exp(2 * (- disCE / disSE))
    # r_forward = forwardScore(endPos, currentPos, currentVel)
    # r_forward_direction = forwardDirectionReward(currentPos, endPos, currentYaw)
    # raw_progress = progressScore(startPos, endPos, state_history)
    # r_progress = float(np.tanh(max(0.0, raw_progress)))
    # r_time = np.clip(
    #     1 - (t / MAX_FRAMES),
    #     0.0,
    #     1.0
    # )
    #
    # short_state_history = state_history[-60:]
    # r_quaternionHistoryScore = quaternionHistoryScore(short_state_history)
    # r_velocityVibration = velocityVibrationScore(short_state_history)
    # r_positionVibration = positionVibrationScore(short_state_history)
    #
    # forward_dir = np.array([np.cos(currentYaw), np.sin(currentYaw), 0.0])
    # r_forward_velocity = scaled_tanh(np.dot(forward_dir, currentVel), 0, 1)
    #
    # speed = np.linalg.norm(currentVel)
    #
    # target_speed = np.clip(disCE / DECEL_RADIUS, 0.0, 1.0) * MAX_SPEED
    #
    # r_final_speed = 1.0 - np.clip(
    #     abs(speed - target_speed) / MAX_SPEED,
    #     0.0,
    #     1.0
    # )
    #
    # reward, result = r_sum([
    #     ("distance", 1.0, r_distance, "brown", "solid", 1.0, []),
    #     ("progress", 0.3, r_progress, "red", "solid", 1.0, ["z"]),
    #     ("forward_direction", 0.3, r_forward_direction, "brown", "solid", 3.0, []),
    #     ("forward", 0.2, r_forward, "blue", "solid", 2.0, ["progress", "z"]),
    #     ("final_speed", 0.25, r_final_speed, "purple", "dotted", 2.0, []),
    #     ("stability", 0.10, r_safe, "green", "dotted", 1.0, [
    #         "forward",
    #         "progress",
    #         "z",
    #         "vibration_quat",
    #         "vibration_pos",
    #         "forward_velocity",
    #         "final_speed",
    #     ]),
    #     ("z", 0.05, r_z, "purple", "solid", 1.0, []),
    #     ("vibration_quat", 0.08, r_quaternionHistoryScore, "orange", "dotted", 1.0, []),
    #     ("vibration_pos", 0.08, r_positionVibration, "gray", "dashed", 1.0, []),
    #     ("time", 0.05, r_time, "pink", "solid", 1.0, []),
    # ])
    #
    # result["total"] = {
    #     "value": float(reward),
    #     "v": float(reward),
    #     "weight": 0.0,
    #     "color": "orange",
    #     "linestyle": "solid",
    #     "linewidth": 4.0,
    # }
    #
    # R, Z = 0.5, 0.01
    # dis_xy = np.linalg.norm(currentPos[0:2] - startPos[0:2])
    #
    # if dis_xy < R and currentPos[2] < state_history[0][2] + 0.2:
    #     if currentPos[2] < state_history[0][2] + t * 0.1:
    #         spin_print("[bad] take off", state_history[0][2] - currentPos[2], t)
    #         reward = 0
    #
    # result["success"] = {
    #     "v": float(r_success),
    #     "weight": 0.0,
    #     "color": "black",
    #     "linestyle": "solid",
    #     "linewidth": 0.5,
    # }
    #
    # result["SPEED"] = {
    #     "v": result["total"]["value"],
    #     "value": reward,
    #     "weight": 0.0,
    #     "color": "orange",
    #     "linestyle": "solid",
    #     "linewidth": 5.0,
    # }
    #
    # result["total"]["v"] = float(reward)
    # return reward, result
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # NEW ACTIVE PRETRAINING LOGIC
    #
    # Important:
    # - state[10:13] is NOT used.
    # - Velocity is measured from state_history.
    #
    # Reward components:
    # - speed_to_forward:
    #     actual measured speed projected onto yaw-forward direction
    #
    # - active_movement:
    #     yaw faces goal
    #     movement direction follows yaw
    #     horizontal movement is active
    #     z movement is small
    #     attitude / clearance is safe
    # ------------------------------------------------------------------

    r_success = 0.0

    r_speed_to_forward = speed_to_forward_score(
        measured_velocity=measured_velocity,
        yaw=currentYaw,
    )

    r_yaw_to_goal = yaw_to_goal_score(
        current=currentPos,
        goal=endPos,
        yaw=currentYaw,
    )

    r_moving_forward = moving_forward_score(
        measured_velocity=measured_velocity,
        yaw=currentYaw,
    )

    r_active_speed = active_speed_score(
        measured_velocity=measured_velocity,
    )

    r_small_z_change = small_z_change_score(
        measured_velocity=measured_velocity,
    )

    r_safe = pretraining_safe_score(
        state=state,
        min_clearance=min_clearance,
    )

    r_active_movement = (
        0.25 * r_yaw_to_goal
        + 0.25 * r_moving_forward
        + 0.20 * r_active_speed
        + 0.15 * r_small_z_change
        + 0.15 * r_safe
    )

    reward, result = r_sum([
        ("speed_to_forward", 0.55, r_speed_to_forward, "blue", "solid", 2.0, []),
        ("active_movement", 0.45, r_active_movement, "green", "solid", 2.0, []),
    ])

    add_debug_row(result, "yaw_to_goal", r_yaw_to_goal, "brown", "solid", 2.0)
    add_debug_row(result, "moving_forward", r_moving_forward, "purple", "solid", 2.0)
    add_debug_row(result, "active_speed", r_active_speed, "red", "solid", 1.5)
    add_debug_row(result, "small_z_change", r_small_z_change, "pink", "solid", 1.5)
    add_debug_row(result, "safe", r_safe, "orange", "dotted", 1.5)

    measured_speed = np.linalg.norm(measured_velocity)
    measured_speed_xy = np.linalg.norm(measured_velocity[0:2])

    add_debug_row(result, "measured_speed", measured_speed, "gray", "dashed", 1.0)
    add_debug_row(result, "measured_speed_xy", measured_speed_xy, "gray", "dashed", 1.0)
    add_debug_row(result, "measured_vx", measured_velocity[0], "gray", "dashed", 1.0)
    add_debug_row(result, "measured_vy", measured_velocity[1], "gray", "dashed", 1.0)
    add_debug_row(result, "measured_vz", measured_velocity[2], "gray", "dashed", 1.0)

    # Keep old takeoff guard.
    R, Z = 0.5, 0.01
    dis_xy = np.linalg.norm(currentPos[0:2] - startPos[0:2])

    if dis_xy < R and currentPos[2] < state_history[0][2] + 0.2:
        if currentPos[2] < state_history[0][2] + t * 0.1:
            spin_print("[bad] take off", state_history[0][2] - currentPos[2], t)
            reward = 0.0

    reward = float(_clamp(reward, 0.0, 1.0))

    result["total"] = {
        "value": reward,
        "v": reward,
        "weight": 0.0,
        "color": "orange",
        "linestyle": "solid",
        "linewidth": 4.0,
    }

    result["success"] = {
        "v": float(r_success),
        "value": float(r_success),
        "weight": 0.0,
        "color": "black",
        "linestyle": "solid",
        "linewidth": 0.5,
    }

    result["SPEED"] = {
        "v": reward,
        "value": reward,
        "weight": 0.0,
        "color": "orange",
        "linestyle": "solid",
        "linewidth": 5.0,
    }

    return reward, result
# swarm/envs/moving_drone.py
from __future__ import annotations

import math
import os
import io
import queue
import threading
import numpy as np
import gymnasium.spaces as spaces
import pybullet as p
import copy
from PIL import Image

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import (
    DroneModel, Physics, ActionType, ObservationType, ImageType,
)

# ── project‑level utilities ────────────────────────────────────────────────
from swarm.core.env_builder import build_world
from swarm.validator.reward import flight_reward
from swarm.constants import (
    DRONE_HULL_RADIUS, ALTITUDE_RAY_INSET, MAX_RAY_DISTANCE,
    DEPTH_NEAR, DEPTH_FAR, DEPTH_MIN_M, DEPTH_MAX_M,
    SEARCH_AREA_NOISE_Z,
    CAMERA_FOV_BASE, CAMERA_FOV_VARIANCE,
    LIGHT_RANDOMIZATION_ENABLED,
    PLATFORM_MOVEMENT_PATTERNS,
    PLATFORM_SPEED_MIN, PLATFORM_SPEED_MAX,
    PLATFORM_RADIUS_MIN, PLATFORM_RADIUS_MAX,
    PLATFORM_DELAY_MIN, PLATFORM_DELAY_MAX,
    PLATFORM_TRANSITION_MIN, PLATFORM_TRANSITION_MAX,
    PLATFORM_LINEAR_DIRECTIONS,
    PLATFORM_AVOIDANCE_ENABLED, PLATFORM_STEER_ANGLES, PLATFORM_MIN_STEP_M,
    LANDING_PLATFORM_RADIUS,
    SAFETY_DISTANCE_SAFE,
    START_PLATFORM_TAKEOFF_BUFFER,
    LANDING_MAX_VZ, LANDING_MAX_VXY_REL, LANDING_MAX_TILT_RAD, LANDING_STABLE_SEC,
    CULL_VISUAL_RADIUS, CULL_PHYSICS_RADIUS, CULL_INTERVAL_STEPS,
    CULL_MIN_AABB_SPAN, CULL_MIN_FACES, CULL_MIN_TOTAL_FACES,
    SOLVER_ITERATIONS, SOLVER_MIN_ISLAND_SIZE,
)

def _moving_drone_path_monitor_main(
    path_queue: "queue.Queue",
    path_monitor_enabled: bool = False,
) -> None:
    if not path_monitor_enabled:
        return

    import sys
    import io
    import queue
    import numpy as np

    from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel
    from PyQt5.QtCore import Qt, QTimer
    from PyQt5.QtGui import QImage, QPixmap

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    argv = sys.argv if getattr(sys, "argv", None) else ["moving_drone"]
    app = QApplication.instance() or QApplication(argv)

    win = QMainWindow()
    win.setWindowTitle("Drone path monitor")

    central = QWidget()
    win.setCentralWidget(central)

    layout = QVBoxLayout()
    central.setLayout(layout)

    lbl_3d = QLabel()
    lbl_xy = QLabel()
    lbl_reward = QLabel()

    lbl_3d.setAlignment(Qt.AlignCenter)
    lbl_xy.setAlignment(Qt.AlignCenter)
    lbl_reward.setAlignment(Qt.AlignCenter)

    lbl_3d.setMinimumSize(400, 265)
    lbl_xy.setMinimumSize(400, 247)
    lbl_reward.setMinimumSize(400, 212)


    layout.addWidget(lbl_3d, stretch=1)
    layout.addWidget(lbl_xy, stretch=1)
    layout.addWidget(lbl_reward, stretch=2)

    latest = [None]

    # =========================
    # Qt render helper (HIGH DPI SAFE)
    # =========================
    def render_to_label(fig, label, dpi=200):
        w = label.width()
        h = label.height()

        fig.set_size_inches(w / dpi, h / dpi, forward=True)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi)
        plt.close(fig)

        buf.seek(0)

        img = QImage()
        img.loadFromData(buf.read())

        label.setPixmap(
            QPixmap.fromImage(img).scaled(
                w,
                h,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
        )

    def drain_queue():
        try:
            while True:
                latest[0] = path_queue.get_nowait()
        except queue.Empty:
            pass

    # =========================
    # MAIN LOOP
    # =========================
    def refresh():
        drain_queue()

        if latest[0] is None:
            return

        path, startpoint, endpoint, reward_hist = latest[0]

        path = np.asarray(path, dtype=float)
        startpoint = np.asarray(startpoint, dtype=float).reshape(-1)
        endpoint = np.asarray(endpoint, dtype=float).reshape(-1)

        distance = np.linalg.norm(endpoint - path[-1][0:3]) / np.linalg.norm(endpoint - startpoint)
        distance = 0
        # =========================
        # 3D PLOT (TRUE SCALE)
        # =========================
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111, projection="3d")

        ax1.plot(path[:, 0], path[:, 1], path[:, 2],
                 color="blue", linewidth=2)

        ax1.scatter(*startpoint, color="green", s=30)
        if distance < 0.7:
            ax1.scatter(*endpoint, color="red", s=30)

        x = np.concatenate([path[:, 0], [startpoint[0], endpoint[0]]])
        y = np.concatenate([path[:, 1], [startpoint[1], endpoint[1]]])
        z = np.concatenate([path[:, 2], [startpoint[2], endpoint[2]]])

        max_range = np.ptp([x, y, z], axis=1).max()

        if max_range < 1e-6:
            max_range = 1.0

        mid_x = (x.max() + x.min()) / 2
        mid_y = (y.max() + y.min()) / 2
        mid_z = (z.max() + z.min()) / 2

        ax1.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
        ax1.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
        ax1.set_zlim(mid_z - max_range/2, mid_z + max_range/2)

        ax1.set_title("3D Path (1:1 scale)")

        render_to_label(fig1, lbl_3d)

        # =========================
        # XY + SIDE VIEW
        # =========================
        fig = plt.figure()

        ax_xy = fig.add_subplot(1, 2, 1)
        ax_side = fig.add_subplot(1, 2, 2)

        n = min(len(path), len(reward_hist))
        path_plot = np.array(path[:n])

        scores = np.array([
            r.get("total", {}).get("v", 0.0)
            for r in reward_hist[:n]
        ], dtype=float)

        norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
        cmap = plt.get_cmap("viridis")
        colors = cmap(norm(scores))

        # =========================
        # XY (true scale)
        # =========================
        ax_xy.scatter(startpoint[0], startpoint[1], color="green", s=20)
        if distance < 0.7:
            ax_xy.scatter(endpoint[0], endpoint[1], color="red", s=20)
        ax_xy.scatter(path_plot[:, 0], path_plot[:, 1], c=colors, s=10, alpha=0.5)

        # Left panel: drone body +X (same axis as onboard camera in BaseAviary) on XY.
        if path_plot.shape[0] > 0 and path_plot.shape[1] >= 6:
            xy_pts = path_plot[:, 0:2]
            span = float(
                max(
                    np.ptp(xy_pts[:, 0]) if xy_pts.shape[0] else 0.0,
                    np.ptp(xy_pts[:, 1]) if xy_pts.shape[0] else 0.0,
                    np.linalg.norm(endpoint[:2] - startpoint[:2]),
                    0.5,
                )
            )
            arrow_len = max(0.05 * span, 0.15)
            step = max(1, path_plot.shape[0] // 25)
            for i in range(0, path_plot.shape[0], step):
                roll, pitch, yaw = (
                    float(path_plot[i, 3]),
                    float(path_plot[i, 4]),
                    float(path_plot[i, 5]),
                )
                rm = np.array(
                    p.getMatrixFromQuaternion(p.getQuaternionFromEuler([roll, pitch, yaw]))
                ).reshape(3, 3)
                fx, fy = float(rm[0, 0]), float(rm[1, 0])
                nrm = math.hypot(fx, fy)
                if nrm < 1e-8:
                    continue
                fx /= nrm
                fy /= nrm
                px, py = float(path_plot[i, 0]), float(path_plot[i, 1])
                ax_xy.annotate(
                    "",
                    xytext=(px, py),
                    xy=(px + fx * arrow_len, py + fy * arrow_len),
                    arrowprops=dict(
                        arrowstyle="-",
                        color="0.15",
                        lw=0.75,
                        alpha=0.42,
                        shrinkA=0,
                        shrinkB=0,
                    ),
                    zorder=5,
                )
            li = path_plot.shape[0] - 1
            roll, pitch, yaw = (
                float(path_plot[li, 3]),
                float(path_plot[li, 4]),
                float(path_plot[li, 5]),
            )
            rm = np.array(
                p.getMatrixFromQuaternion(p.getQuaternionFromEuler([roll, pitch, yaw]))
            ).reshape(3, 3)
            fx, fy = float(rm[0, 0]), float(rm[1, 0])
            nrm = math.hypot(fx, fy)
            if nrm >= 1e-8:
                fx /= nrm
                fy /= nrm
                px, py = float(path_plot[li, 0]), float(path_plot[li, 1])
                ax_xy.annotate(
                    "",
                    xytext=(px, py),
                    xy=(px + fx * arrow_len * 1.35, py + fy * arrow_len * 1.35),
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color="orangered",
                        lw=1.2,
                        alpha=0.9,
                        shrinkA=0,
                        shrinkB=0,
                    ),
                    zorder=6,
                )

        ax_xy.set_aspect("equal", adjustable="box")
        ax_xy.set_title("XY Path (1:1)")

        # =========================
        # SIDE VIEW
        # =========================
        start_xy = np.array(startpoint[:2])
        end_xy = np.array(endpoint[:2])

        direction = end_xy - start_xy
        direction_norm = direction / (np.linalg.norm(direction) + 1e-8)

        rel = path_plot[:, :2] - start_xy
        dist = np.dot(rel, direction_norm)
        z_vals = path_plot[:, 2]

        ax_side.scatter(dist, z_vals, c=colors, s=10, alpha=0.5)

        if distance < 0.7:
            ax_side.plot([0, np.linalg.norm(direction)],
                     [startpoint[2], endpoint[2]],
                     linestyle="--", color="gray")
        ax_side.set_aspect("equal", adjustable="box")

        ax_side.set_title("Distance vs Z")

        sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
        sm.set_array(scores)

        fig.colorbar(sm, ax=[ax_xy, ax_side], label="Score")

        render_to_label(fig, lbl_xy)

        # =========================
        # REWARD PLOT (SAFE + VALUE CHART)
        # =========================
        if reward_hist and reward_hist[0]:

            fig3 = plt.figure(figsize=(6, 6))
            ax_top = fig3.add_subplot(2, 1, 1)
            ax_bottom = fig3.add_subplot(2, 1, 2)

            n = len(reward_hist)

            # =========================
            # TOP: processed reward ("v")
            # =========================
            for vname, meta in reward_hist[0].items():
                if vname in ("total", "success") or not meta:
                    continue

                ys = np.array([
                    float(reward_hist[i].get(vname, {}).get("v", 0.0))
                    for i in range(n)
                ], dtype=float)

                ax_top.plot(
                    ys,
                    # label=meta.get("label", vname),
                    color=meta.get("color", "black"),
                    linewidth=meta.get("linewidth", 1.5),
                    linestyle=meta.get("linestyle", "--")
                )

            total_v = np.array([
                float(r.get("total", {}).get("v", 0.0))
                for r in reward_hist
            ], dtype=float)

            ax_top.plot(total_v, color="orange", label="Total (v)", linewidth=2, linestyle="solid")

            ax_top.set_title("Reward Components (Processed v)")
            ax_top.set_xlabel("Step")
            ax_top.set_ylabel("Reward (v)")
            ax_top.legend(fontsize="x-small")
            ax_top.grid(True, alpha=0.2)

            # =========================
            # BOTTOM: raw values ("value")
            # =========================
            for vname, meta in reward_hist[0].items():
                if vname in ("total", "success") or not meta:
                    continue

                ys = np.array([
                    float(
                        reward_hist[i]
                        .get(vname, {})
                        .get("value",
                            reward_hist[i].get(vname, {}).get("v", 0.0))
                    )
                    for i in range(n)
                ], dtype=float)

                ax_bottom.plot(
                    ys,
                    linestyle=meta.get("linestyle", "--"),
                    alpha=0.8,
                    # label=meta.get("label", vname),
                    color=meta.get("color", "black"),
                    linewidth=meta.get("linewidth", 1.0)
                )

            total_raw = np.array([
                float(
                    r.get("total", {}).get("value",
                    r.get("total", {}).get("v", 0.0))
                )
                for r in reward_hist
            ], dtype=float)

            ax_bottom.plot(total_raw, color="orange", label="Total (raw)", linewidth=2, linestyle="solid")

            ax_bottom.set_title("Reward Components (Raw values)")
            ax_bottom.set_xlabel("Step")
            ax_bottom.set_ylabel("Reward (raw)", color=meta.get("color", "black"))
            ax_bottom.legend(fontsize="x-small")
            ax_bottom.grid(True, alpha=0.2)

            # =========================
            # FINAL RENDER
            # =========================
            render_to_label(fig3, lbl_reward)

    timer = QTimer()
    timer.timeout.connect(refresh)
    timer.start(200)

    win.resize(440, 560)
    win.show()
    app.exec_()


class MovingDroneAviary(BaseRLAviary):
    """
    Single‑drone environment whose *start*, *goal* and *horizon* are supplied
    via an external `MapTask`.

    The per‑step reward is the **increment** of `flight_reward`, so it can be
    fed directly to PPO/TD3/etc. without extra shaping.
    """
    MAX_TILT_RAD: float = 1.047         # safety cut‑off for roll / pitch (rad)
    _fov: float = 90.0
    PATH_MONITOR_STEP_INTERVAL: int = 120   # Qt path monitor refresh within an episode

    # --------------------------------------------------------------------- #
    # 1. constructor
    # --------------------------------------------------------------------- #
    def __init__(
        self,
        task,
        drone_model : DroneModel   = DroneModel.CF2X,
        physics     : Physics      = Physics.PYB,
        pyb_freq    : int          = 240,
        ctrl_freq   : int          = 30,
        gui         : bool         = False,
        record      : bool         = False,
        path_monitor: bool         = False,
        obs         : ObservationType = ObservationType.RGB,
        act         : ActionType      = ActionType.RPM,
    ):
        """
        Parameters
        ----------
        task : MapTask
            Must expose `.start`, `.goal`, `.horizon`, `.sim_dt`.
        path_monitor : bool
            When True, start the Qt matplotlib path/reward monitor (see also ``SWARM_PATH_MONITOR``).
        Remaining arguments are forwarded to ``BaseRLAviary`` unchanged.
        """
        self._path_monitor_queue = queue.Queue(maxsize=1)
        self._path_monitor_thread = threading.Thread(
            target=_moving_drone_path_monitor_main,
            args=(self._path_monitor_queue, path_monitor),
            daemon=True,
            name="MovingDronePathMonitor",
        )
        self._path_monitor_thread.start()

        self.task       = task
        self._original_start = tuple(task.start)
        self._original_goal = tuple(task.goal)
        self.GOAL_POS   = np.asarray(task.goal, dtype=float)
        self.EP_LEN_SEC = float(task.horizon)
        self._moving    = getattr(task, 'moving_platform', False)

        self._time_alive = 0.0
        self._success = False
        self._collision = False
        self._t_to_goal = None
        self._prev_score = 0.0
        self._step_processed = False
        self._min_clearance_episode = SAFETY_DISTANCE_SAFE
        self._landing_stable_time = 0.0
        self._prev_platform_pos = None
        self._platform_velocity = np.zeros(3, dtype=np.float32)
        
        seed = getattr(task, 'map_seed', 0)
        
        self._platform_orbit_center = self.GOAL_POS.copy()
        self._current_platform_pos = self.GOAL_POS.copy()
        self._movement_pattern = self._get_movement_pattern_from_seed(seed)
        self._platform_offsets = []

        self._init_platform_randomization(seed)
        self._search_area_center = self.GOAL_POS.copy()
        
        self.state_history = []
        self.reward_history = []
        self.last_reward_history = []
        self._path_monitor_ctrl_step = 0
        self._original_start = np.array(task.start)
        self._original_goal = np.array(task.goal)
        self._prev_path_progress_01 = 0.0
        self._schedule_deadline_trunc = False

        fov_rng = np.random.RandomState(seed)
        fov_rng.rand()
        self._fov = CAMERA_FOV_BASE + fov_rng.uniform(-CAMERA_FOV_VARIANCE, CAMERA_FOV_VARIANCE)

        if LIGHT_RANDOMIZATION_ENABLED:
            light_rng = np.random.RandomState(seed)
            light_rng.rand()
            light_rng.rand()
            light_rng.rand()
            angle = light_rng.uniform(0, 2 * np.pi)
            self._light_direction = [
                -np.cos(angle),
                0.1 * np.sin(angle * 3),
                np.sin(angle)
            ]
        else:
            self._light_direction = [0, 0, 1]

        # Let BaseRLAviary set up the PyBullet world
        super().__init__(
            drone_model  = drone_model,
            num_drones   = 1,
            initial_xyzs = np.asarray([task.start]),
            initial_rpys = None,
            physics      = physics,
            pyb_freq     = pyb_freq,
            ctrl_freq    = ctrl_freq,
            gui          = gui,
            record       = record,
            obs          = obs,
            act          = act,
        )

        if self.OBS_TYPE != ObservationType.RGB:
            raise ValueError("MovingDroneAviary only supports ObservationType.RGB observations.")

        enhanced_width, enhanced_height = 128, 128
        self.IMG_RES = np.array([enhanced_width, enhanced_height])
        self.dep = np.ones((self.NUM_DRONES, enhanced_height, enhanced_width), dtype=np.float32)

        action_dim = self.action_space.shape[-1]
        state_dim = 12 + self.ACTION_BUFFER_SIZE * action_dim + 1 + 3
        self._state_dim = state_dim

        depth_shape = (enhanced_height, enhanced_width, 1)
        self.observation_space = spaces.Dict({
            "depth": spaces.Box(
                low=0.0,
                high=1.0,
                shape=depth_shape,
                dtype=np.float32
            ),
            "state": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(state_dim,),
                dtype=np.float32
            ),
        })

        self._cull_targets = []
        self._cull_vis_hidden = set()
        self._cull_phys_disabled = set()
        self._cull_step_counter = 0
        self._cull_enabled = False

    # --------------------------------------------------------------------- #
    # 2. low‑level helpers
    # --------------------------------------------------------------------- #
    @property
    def _sim_dt(self) -> float:
        """Physics step in seconds (1 / CTRL_FREQ)."""
        return 1.0 / self.CTRL_FREQ
    
    def _get_movement_pattern_from_seed(self, seed: int) -> str:
        """Deterministically select movement pattern based on seed."""
        if not self._moving:
            return "static"
        rng = np.random.RandomState(seed)
        rng.rand()
        rng.rand()
        rng.rand()
        rng.rand()
        pattern_idx = rng.randint(0, len(PLATFORM_MOVEMENT_PATTERNS))
        return PLATFORM_MOVEMENT_PATTERNS[pattern_idx]

    def _init_platform_randomization(self, seed: int) -> None:
        """Initialize randomized platform movement parameters."""
        if not self._moving:
            self._platform_speed = 0.0
            self._platform_radius = 0.0
            self._platform_delay = 0.0
            self._platform_transition_time = 0.0
            self._platform_phase = 0.0
            self._platform_linear_dir = "x"
            self._platform_linear_angle = 0.0
            return

        rng = np.random.RandomState((seed + 77777) & 0xFFFFFFFF)
        self._platform_speed = rng.uniform(PLATFORM_SPEED_MIN, PLATFORM_SPEED_MAX)
        self._platform_radius = rng.uniform(PLATFORM_RADIUS_MIN, PLATFORM_RADIUS_MAX)
        self._platform_delay = rng.uniform(PLATFORM_DELAY_MIN, PLATFORM_DELAY_MAX)
        self._platform_transition_time = rng.uniform(PLATFORM_TRANSITION_MIN, PLATFORM_TRANSITION_MAX)
        self._platform_phase = rng.uniform(0, 2 * np.pi)
        dir_idx = rng.randint(0, len(PLATFORM_LINEAR_DIRECTIONS))
        self._platform_linear_dir = PLATFORM_LINEAR_DIRECTIONS[dir_idx]
        self._platform_linear_angle = rng.uniform(0, 2 * np.pi)

    def _get_orbit_position(self, t_eff: float) -> np.ndarray:
        """Calculate orbit position for a given effective time."""
        center = self._platform_orbit_center
        speed = self._platform_speed
        radius = self._platform_radius
        phase = self._platform_phase
        pattern = self._movement_pattern

        if pattern == "circular":
            angle = t_eff * speed * 0.3 + phase
            x = center[0] + radius * math.cos(angle)
            y = center[1] + radius * math.sin(angle)
            return np.array([x, y, center[2]], dtype=np.float32)

        elif pattern == "linear":
            offset = radius * math.sin(t_eff * speed * 0.5 + phase)
            if self._platform_linear_dir == "x":
                x = center[0] + offset
                y = center[1]
            elif self._platform_linear_dir == "y":
                x = center[0]
                y = center[1] + offset
            else:
                x = center[0] + offset * math.cos(self._platform_linear_angle)
                y = center[1] + offset * math.sin(self._platform_linear_angle)
            return np.array([x, y, center[2]], dtype=np.float32)

        elif pattern == "figure8":
            angle = t_eff * speed * 0.3 + phase
            x = center[0] + radius * math.sin(angle)
            y = center[1] + radius * math.sin(2 * angle) / 2
            return np.array([x, y, center[2]], dtype=np.float32)

        return np.array(center, dtype=np.float32)

    def _calculate_platform_position(self, t: float) -> np.ndarray:
        """Calculate platform position at time t with smooth transition."""
        if not self._moving:
            return self._platform_orbit_center.copy()

        delay = self._platform_delay
        transition = self._platform_transition_time
        center = self._platform_orbit_center

        if t < delay:
            return np.array(center, dtype=np.float32)

        orbit_start = self._get_orbit_position(0.0)

        if t < delay + transition:
            t_ratio = (t - delay) / transition
            t_smooth = t_ratio * t_ratio * (3.0 - 2.0 * t_ratio)
            return center + t_smooth * (orbit_start - center)

        t_eff = t - delay - transition
        return self._get_orbit_position(t_eff)
    
    def _platform_path_blocked(self, current_pos, target_pos):
        """Check if path or destination is blocked by obstacles."""
        cli = getattr(self, "CLIENT", 0)
        direction = target_pos - current_pos
        dist = np.linalg.norm(direction[:2])
        if dist < 0.001:
            return False, None

        excluded = set(getattr(self, '_end_platform_uids', []))
        excluded |= set(getattr(self, '_start_platform_uids', []))
        excluded.add(self.DRONE_IDS[0])
        excluded.add(getattr(self, 'PLANE_ID', 0))

        offsets = [
            np.array([0, 0, 0], dtype=np.float32),
            np.array([LANDING_PLATFORM_RADIUS, 0, 0], dtype=np.float32),
            np.array([-LANDING_PLATFORM_RADIUS, 0, 0], dtype=np.float32),
            np.array([0, LANDING_PLATFORM_RADIUS, 0], dtype=np.float32),
            np.array([0, -LANDING_PLATFORM_RADIUS, 0], dtype=np.float32),
        ]

        for offset in offsets:
            ray_from = (current_pos + offset).tolist()
            ray_to = (target_pos + offset).tolist()
            result = p.rayTest(ray_from, ray_to, physicsClientId=cli)
            if result and result[0][0] != -1 and result[0][0] not in excluded:
                return True, np.array(result[0][3], dtype=np.float32)

        end_uids = getattr(self, '_end_platform_uids', [])
        if end_uids:
            plat_uid = end_uids[0]
            saved_pos, saved_orn = p.getBasePositionAndOrientation(plat_uid, physicsClientId=cli)
            p.resetBasePositionAndOrientation(plat_uid, target_pos.tolist(), [0, 0, 0, 1], physicsClientId=cli)
            num_bodies = p.getNumBodies(physicsClientId=cli)
            for body_idx in range(num_bodies):
                body_uid = p.getBodyUniqueId(body_idx, physicsClientId=cli)
                if body_uid in excluded:
                    continue
                mn, mx = p.getAABB(body_uid, physicsClientId=cli)
                if (mx[0] - mn[0]) > 50.0 or (mx[1] - mn[1]) > 50.0:
                    continue
                contacts = p.getClosestPoints(plat_uid, body_uid, distance=0.15, physicsClientId=cli)
                if contacts:
                    for c in contacts:
                        if c[8] < 0.15:
                            p.resetBasePositionAndOrientation(plat_uid, list(saved_pos), list(saved_orn), physicsClientId=cli)
                            return True, target_pos.copy()
            p.resetBasePositionAndOrientation(plat_uid, list(saved_pos), list(saved_orn), physicsClientId=cli)

        return False, None

    def _update_moving_platform(self):
        """Update platform position with obstacle avoidance."""
        nominal_pos = self._calculate_platform_position(self._time_alive)

        if not self._moving:
            self._prev_platform_pos = nominal_pos.copy()
            self._current_platform_pos = nominal_pos
            self._platform_velocity = np.zeros(3, dtype=np.float32)
            return

        current = self._current_platform_pos
        if current is None:
            current = nominal_pos

        blocked, _ = self._platform_path_blocked(current, nominal_pos)

        if not blocked or not PLATFORM_AVOIDANCE_ENABLED:
            new_pos = nominal_pos
        else:
            direction = nominal_pos - current
            raw_dist = np.linalg.norm(direction[:2])
            step = np.clip(raw_dist * 0.3, PLATFORM_MIN_STEP_M, 0.10)
            base_angle = math.atan2(direction[1], direction[0])

            new_pos = current.copy()
            for angle_deg in PLATFORM_STEER_ANGLES:
                angle = base_angle + math.radians(angle_deg)
                candidate = current.copy()
                candidate[0] += step * math.cos(angle)
                candidate[1] += step * math.sin(angle)
                candidate[2] = nominal_pos[2]

                candidate_blocked, _ = self._platform_path_blocked(current, candidate)
                if not candidate_blocked:
                    new_pos = candidate
                    break

        max_step_dist = self._platform_speed * self._sim_dt * 1.5
        disp = new_pos - current
        disp_dist = np.linalg.norm(disp[:2])
        if disp_dist > max_step_dist > 0:
            scale = max_step_dist / disp_dist
            new_pos[0] = current[0] + disp[0] * scale
            new_pos[1] = current[1] + disp[1] * scale

        center = self._platform_orbit_center
        rel = new_pos[:2] - center[:2]
        r = np.linalg.norm(rel)
        r_max = self._platform_radius + 0.3
        if r > r_max:
            new_pos[:2] = center[:2] + rel * (r_max / max(r, 1e-6))

        if self._prev_platform_pos is not None:
            dt = self._sim_dt
            if dt > 0:
                self._platform_velocity = (new_pos - self._prev_platform_pos) / dt

        self._prev_platform_pos = new_pos.copy()
        self._current_platform_pos = new_pos

        if not hasattr(self, '_end_platform_uids') or not self._end_platform_uids:
            return

        cli = getattr(self, "CLIENT", 0)

        if not self._platform_offsets and self._end_platform_uids:
            initial_pos = self._platform_orbit_center
            for uid in self._end_platform_uids:
                pos, _ = p.getBasePositionAndOrientation(uid, physicsClientId=cli)
                offset = np.array(pos, dtype=np.float32) - initial_pos
                self._platform_offsets.append(offset)

        for i, uid in enumerate(self._end_platform_uids):
            offset = self._platform_offsets[i] if i < len(self._platform_offsets) else np.zeros(3)
            final_pos = new_pos + offset
            p.resetBasePositionAndOrientation(
                uid,
                final_pos.tolist(),
                [0, 0, 0, 1],
                physicsClientId=cli
            )

    def _getDroneImages(self, nth_drone, segmentation: bool = False):
        """Get camera images from drone. Returns (rgb, depth, seg) but we only use depth."""
        if self.OBS_TYPE != ObservationType.RGB:
            return super()._getDroneImages(nth_drone, segmentation)
        
        if self.IMG_RES is None:
            print("[ERROR] in MovingDroneAviary._getDroneImages(), IMG_RES not set")
            exit()
        
        cli = getattr(self, "CLIENT", 0)
        drone_pos = np.asarray(self.pos[nth_drone, :], dtype=np.float64)
        if not np.isfinite(drone_pos).all():
            drone_pos = np.nan_to_num(drone_pos, nan=0.0)
        quat = np.asarray(self.quat[nth_drone, :], dtype=np.float64)
        if not np.isfinite(quat).all():
            rot_mat = np.eye(3, dtype=np.float64)
        else:
            rot_mat = np.array(p.getMatrixFromQuaternion(quat.tolist())).reshape(3, 3)
            if not np.isfinite(rot_mat).all():
                rot_mat = np.eye(3, dtype=np.float64)

        forward = rot_mat @ np.array([1.0, 0.0, 0.0])
        fn = float(np.linalg.norm(forward))
        if not np.isfinite(fn) or fn < 1e-9:
            forward = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            forward = forward / fn
        up = rot_mat @ np.array([0.0, 0.0, 1.0])
        
        camera_offset = 0.13
        camera_pos = drone_pos + forward * camera_offset + up * 0.05
        if not np.isfinite(camera_pos).all():
            camera_pos = np.nan_to_num(camera_pos, nan=0.0)

        target = camera_pos + forward * 20.0
        
        DRONE_CAM_VIEW = p.computeViewMatrix(
            cameraEyePosition=camera_pos,
            cameraTargetPosition=target,
            cameraUpVector=up.tolist(),
            physicsClientId=cli
        )
        
        aspect = self.IMG_RES[0] / self.IMG_RES[1]
        DRONE_CAM_PRO = p.computeProjectionMatrixFOV(
            fov=self._fov,
            aspect=aspect,
            nearVal=0.05,
            farVal=DEPTH_FAR,
            physicsClientId=cli
        )
        
        seg_flag = p.ER_NO_SEGMENTATION_MASK
        depth_only_flag = getattr(p, "ER_DEPTH_ONLY", None)
        if depth_only_flag is not None:
            seg_flag |= depth_only_flag
        [w, h, _rgb, dep, _seg] = p.getCameraImage(
            width=self.IMG_RES[0],
            height=self.IMG_RES[1],
            shadow=0,
            renderer=p.ER_TINY_RENDERER,
            viewMatrix=DRONE_CAM_VIEW,
            projectionMatrix=DRONE_CAM_PRO,
            lightDirection=self._light_direction,
            flags=seg_flag,
            physicsClientId=cli
        )
        
        dep = np.reshape(dep, (h, w))
        return None, dep, None

    def _get_altitude_distance(self) -> float:
        """Cast single ray downward for ground/altitude detection."""
        cli = getattr(self, "CLIENT", 0)
        uid = self.DRONE_IDS[0]
        pos, _ = p.getBasePositionAndOrientation(uid, physicsClientId=cli)
        pos = np.asarray(pos, dtype=float)

        ray_origin_offset = DRONE_HULL_RADIUS - ALTITUDE_RAY_INSET
        start = [pos[0], pos[1], pos[2] - ray_origin_offset]
        end = [pos[0], pos[1], pos[2] - MAX_RAY_DISTANCE]

        result = p.rayTest(start, end, physicsClientId=cli)
        hit_uid, _, hit_frac, _, _ = result[0]

        if hit_uid != -1:
            hf = float(hit_frac)
            if not np.isfinite(hf):
                hf = 1.0
            hf = float(np.clip(hf, 0.0, 1.0))
            seg_len = MAX_RAY_DISTANCE - ray_origin_offset
            return min(MAX_RAY_DISTANCE, ray_origin_offset + hf * seg_len)
        return MAX_RAY_DISTANCE

    def _process_depth(self, depth_buffer: np.ndarray) -> np.ndarray:
        """Convert PyBullet depth buffer to normalized depth map [0,1] for 0.5-20m range."""
        depth_buffer = np.nan_to_num(
            np.asarray(depth_buffer, dtype=np.float64),
            nan=1.0,
            posinf=1.0,
            neginf=0.0,
        )
        depth_buffer = np.clip(depth_buffer, 0.0, 1.0)
        
        denominator = DEPTH_FAR - (DEPTH_FAR - DEPTH_NEAR) * depth_buffer
        denominator = np.maximum(denominator, DEPTH_NEAR * 1e-6)
        
        depth_meters = DEPTH_FAR * DEPTH_NEAR / denominator
        depth_clipped = np.clip(depth_meters, DEPTH_MIN_M, DEPTH_MAX_M)
        depth_normalized = (depth_clipped - DEPTH_MIN_M) / (DEPTH_MAX_M - DEPTH_MIN_M)
        return depth_normalized.astype(np.float32)[..., np.newaxis]

    def _generate_search_area_center(self, seed: int = None) -> np.ndarray:
        """Generate search area center position with noise for GPS simulation."""
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = self.np_random
        search_radius = getattr(self.task, 'search_radius', 10.0)
        noise_xy = rng.uniform(-search_radius, search_radius, size=2)
        noise_z = rng.uniform(-SEARCH_AREA_NOISE_Z, SEARCH_AREA_NOISE_Z)
        center = self.GOAL_POS.copy()
        center[0] += noise_xy[0]
        center[1] += noise_xy[1]
        center[2] = max(0.0, center[2] + noise_z)
        return center

    def _check_collision(self) -> tuple:
        """
        Inspect contact points and detect collisions.
        Returns (platform_hit, obstacle_hit) tuple.
        """
        drone_id = self.DRONE_IDS[0]
        contact_points = p.getContactPoints(
            bodyA=drone_id,
            physicsClientId=getattr(self, "CLIENT", 0)
        )

        if not contact_points:
            return False, False

        end_platform_uids = getattr(self, '_end_platform_uids', [])
        start_platform_uids = getattr(self, '_start_platform_uids', [])

        platform_hit = False
        obstacle_hit = False

        for contact in contact_points:
            body_b = contact[2]
            if body_b == -1:
                continue

            normal_force = contact[9]
            if normal_force <= 0.01:
                continue

            if body_b in end_platform_uids:
                platform_hit = True
                continue

            if body_b in start_platform_uids:
                continue

            obstacle_hit = True
            break

        if obstacle_hit:
            self._collision = True

        return platform_hit, obstacle_hit

    def _update_landing_state(self, platform_contact: bool) -> None:
        """Update landing state machine based on contact and drone state."""
        if self._success or self._collision:
            return

        if not platform_contact:
            self._landing_stable_time = 0.0
            return

        if self._moving:
            self._success = True
            self._t_to_goal = self._time_alive
            return

        state = self._getDroneStateVector(0)
        roll, pitch = state[7], state[8]
        vel = state[10:13]

        vz = abs(vel[2])
        drone_vxy = vel[0:2]
        platform_vxy = self._platform_velocity[0:2]
        rel_vxy = np.linalg.norm(drone_vxy - platform_vxy)

        velocity_ok = vz <= LANDING_MAX_VZ and rel_vxy <= LANDING_MAX_VXY_REL
        upright_ok = abs(roll) <= LANDING_MAX_TILT_RAD and abs(pitch) <= LANDING_MAX_TILT_RAD

        if velocity_ok and upright_ok:
            self._landing_stable_time += self._sim_dt
            if self._landing_stable_time >= LANDING_STABLE_SEC:
                self._success = True
                self._t_to_goal = self._time_alive
        else:
            self._landing_stable_time = 0.0

    # --------------------------------------------------------------------- #
    # distance-based culling
    # --------------------------------------------------------------------- #
    @staticmethod
    def _count_mesh_faces(path: str) -> int:
        if not os.path.exists(path):
            return 0
        count = 0
        with open(path) as f:
            for line in f:
                if line[0:2] == "f ":
                    count += 1
        return count

    def _build_cull_targets(self) -> None:
        """Scan scene bodies and build the cull-target list."""
        cli = getattr(self, "CLIENT", 0)
        drone_id = self.DRONE_IDS[0]
        ground_id = getattr(self, "PLANE_ID", 0)
        end_uids = set(getattr(self, "_end_platform_uids", []))
        start_uids = set(getattr(self, "_start_platform_uids", []))
        protected = {drone_id, ground_id} | end_uids | start_uids

        targets = []
        total_faces = 0
        n = p.getNumBodies(physicsClientId=cli)

        for i in range(n):
            uid = p.getBodyUniqueId(i, physicsClientId=cli)
            if uid in protected:
                continue
            mn, mx = p.getAABB(uid, physicsClientId=cli)
            span = max(mx[0] - mn[0], mx[1] - mn[1])
            if span < CULL_MIN_AABB_SPAN:
                continue
            vdata = p.getVisualShapeData(uid, physicsClientId=cli)
            if not vdata:
                continue
            faces = 0
            for v in vdata:
                if v[2] == p.GEOM_MESH:
                    fname = v[4].decode() if isinstance(v[4], bytes) else str(v[4])
                    faces += self._count_mesh_faces(fname)
            if faces < CULL_MIN_FACES:
                continue
            cx = (mn[0] + mx[0]) * 0.5
            cy = (mn[1] + mx[1]) * 0.5
            rgba_orig = list(vdata[0][7])
            targets.append((uid, cx, cy, span / 2.0, rgba_orig))
            total_faces += faces

        self._cull_targets = targets
        self._cull_vis_hidden = set()
        self._cull_phys_disabled = set()
        self._cull_step_counter = 0
        self._cull_enabled = (not getattr(self, "GUI", False)) and total_faces >= CULL_MIN_TOTAL_FACES

    def _apply_distance_cull(self) -> None:
        """Toggle visual/physics state for bodies beyond camera range."""
        if getattr(self, "GUI", False):
            if self._cull_vis_hidden or self._cull_phys_disabled:
                self._restore_culled_bodies()
            return
        if not self._cull_enabled:
            return
        self._cull_step_counter += 1
        if self._cull_step_counter % CULL_INTERVAL_STEPS != 0:
            return

        cli = getattr(self, "CLIENT", 0)
        dp = p.getBasePositionAndOrientation(self.DRONE_IDS[0], physicsClientId=cli)[0]
        dx, dy = dp[0], dp[1]
        vis_hidden = self._cull_vis_hidden
        phys_disabled = self._cull_phys_disabled

        for uid, cx, cy, hs, rgba in self._cull_targets:
            dist = math.sqrt((cx - dx) ** 2 + (cy - dy) ** 2)
            surface_dist = dist - hs

            if surface_dist > CULL_VISUAL_RADIUS:
                if uid not in vis_hidden:
                    p.changeVisualShape(uid, -1, rgbaColor=[0, 0, 0, 0], physicsClientId=cli)
                    vis_hidden.add(uid)
            elif uid in vis_hidden:
                p.changeVisualShape(uid, -1, rgbaColor=rgba, physicsClientId=cli)
                vis_hidden.discard(uid)

            if surface_dist > CULL_PHYSICS_RADIUS:
                if uid not in phys_disabled:
                    p.setCollisionFilterGroupMask(uid, -1, 0, 0, physicsClientId=cli)
                    phys_disabled.add(uid)
            elif uid in phys_disabled:
                p.setCollisionFilterGroupMask(uid, -1, 1, 0xFF, physicsClientId=cli)
                phys_disabled.discard(uid)

    def _restore_culled_bodies(self) -> None:
        """Restore all culled bodies to their original state."""
        cli = getattr(self, "CLIENT", 0)
        for uid, _, _, _, rgba in self._cull_targets:
            if uid in self._cull_vis_hidden:
                p.changeVisualShape(uid, -1, rgbaColor=rgba, physicsClientId=cli)
            if uid in self._cull_phys_disabled:
                p.setCollisionFilterGroupMask(uid, -1, 1, 0xFF, physicsClientId=cli)
        self._cull_vis_hidden.clear()
        self._cull_phys_disabled.clear()

    def _update_min_clearance(self) -> None:
        """Update minimum obstacle clearance for the episode."""
        if self._collision:
            self._min_clearance_episode = 0.0
            return

        cli = getattr(self, "CLIENT", 0)
        drone_id = self.DRONE_IDS[0]
        end_platform_uids = getattr(self, '_end_platform_uids', [])
        start_platform_uids = getattr(self, '_start_platform_uids', [])
        ground_id = getattr(self, 'PLANE_ID', 0)
        excluded = {drone_id, -1, ground_id} | set(end_platform_uids) | set(start_platform_uids)

        min_dist = SAFETY_DISTANCE_SAFE

        d_min, d_max = p.getAABB(drone_id, physicsClientId=cli)
        search_min = [d_min[0] - SAFETY_DISTANCE_SAFE, d_min[1] - SAFETY_DISTANCE_SAFE, d_min[2] - SAFETY_DISTANCE_SAFE]
        search_max = [d_max[0] + SAFETY_DISTANCE_SAFE, d_max[1] + SAFETY_DISTANCE_SAFE, d_max[2] + SAFETY_DISTANCE_SAFE]
        overlapping = p.getOverlappingObjects(search_min, search_max, physicsClientId=cli)

        if overlapping:
            checked = set()
            for body_uid, _link_idx in overlapping:
                if body_uid in excluded or body_uid in checked:
                    continue
                checked.add(body_uid)

                closest = p.getClosestPoints(
                    bodyA=drone_id,
                    bodyB=body_uid,
                    distance=SAFETY_DISTANCE_SAFE,
                    physicsClientId=cli
                )

                for point in closest:
                    dist = point[8]
                    if dist < min_dist:
                        min_dist = dist

        if min_dist < self._min_clearance_episode:
            self._min_clearance_episode = min_dist

    def _enqueue_path_monitor_snapshot(self) -> None:
        """Notify Qt path monitor (non-blocking; queue keeps latest only)."""
        if len(self.state_history) == 0:
            return
        if self.task is None:
            return
        snap = (
            np.asarray(self.state_history, dtype=np.float64).copy(),
            np.asarray(self.task.start, dtype=np.float64).copy(),
            np.asarray(self.task.goal, dtype=np.float64).copy(),
            copy.deepcopy(self.reward_history),
        )
        try:
            self._path_monitor_queue.put_nowait(snap)
        except queue.Full:
            try:
                self._path_monitor_queue.get_nowait()
            except queue.Empty:
                pass
            self._path_monitor_queue.put_nowait(snap)

    # --------------------------------------------------------------------- #
    # 3. OpenAI‑Gym API overrides
    # --------------------------------------------------------------------- #
    def reset(self, **kwargs):
        """Reset environment and internal state for a new episode."""
        if len(self.state_history) > 0:
            self._enqueue_path_monitor_snapshot()
            self.state_history = []

        seed = kwargs.get('seed', None)
        if seed is None:
            seed = getattr(self.task, 'map_seed', None)

        p.resetSimulation(physicsClientId=self.CLIENT)
        self._housekeeping()
        self._updateAndStoreKinematicInformation()
        self._startVideoRecording()

        self._time_alive = 0.0
        self._success = False
        self._collision = False
        self._t_to_goal = None
        self._step_processed = False
        self._min_clearance_episode = SAFETY_DISTANCE_SAFE
        self._landing_stable_time = 0.0
        self._prev_platform_pos = None
        self._platform_velocity = np.zeros(3, dtype=np.float32)
        self._platform_offsets = []
        self._path_monitor_ctrl_step = 0
        self._schedule_deadline_trunc = False
        self._reset_action_buffer()

        (self._prev_score, rewards) = flight_reward(
            success=False,
            t=0.0,
            horizon=self.EP_LEN_SEC,
            task=None,
            min_clearance=self._min_clearance_episode,
            collision=self._collision,
            state=None,
            state_history=None,
        )
        if len(self.reward_history) > 0:
            self.last_reward_history = copy.deepcopy(self.reward_history)
        else:
            self.last_reward_history = []
        self.reward_history = []
        self._spawn_task_world()
        self._search_area_center = self._generate_search_area_center(seed=seed)
        self._updateAndStoreKinematicInformation()
        self._sync_path_progress_baseline()

        cli = getattr(self, "CLIENT", 0)
        p.setPhysicsEngineParameter(
            numSolverIterations=SOLVER_ITERATIONS,
            minimumSolverIslandSize=SOLVER_MIN_ISLAND_SIZE,
            physicsClientId=cli,
        )

        obs_after = self._computeObs()
        if obs_after is not None and "state" in obs_after:
            actual_state_dim = obs_after["state"].shape[0]
            if actual_state_dim != self._state_dim:
                self._state_dim = actual_state_dim
                self.observation_space = spaces.Dict({
                    "depth": self.observation_space["depth"],
                    "state": spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=(actual_state_dim,),
                        dtype=np.float32
                    ),
                })
        info_after = self._computeInfo()
        return obs_after, info_after

    def step(self, action):
        """Execute one control step with post-physics bookkeeping."""
        self._step_processed = False
        if self.RECORD and not self.GUI and self.step_counter % self.CAPTURE_FREQ == 0:
            [w, h, rgb, dep, seg] = p.getCameraImage(
                width=self.VID_WIDTH,
                height=self.VID_HEIGHT,
                shadow=1,
                viewMatrix=self.CAM_VIEW,
                projectionMatrix=self.CAM_PRO,
                renderer=p.ER_TINY_RENDERER,
                flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
                physicsClientId=self.CLIENT,
            )
            (Image.fromarray(np.reshape(rgb, (h, w, 4)), 'RGBA')).save(
                os.path.join(self.IMG_PATH, "frame_" + str(self.FRAME_NUM) + ".png")
            )
            self.FRAME_NUM += 1
            if self.VISION_ATTR:
                for i in range(self.NUM_DRONES):
                    self.rgb[i], self.dep[i], self.seg[i] = self._getDroneImages(i)
                    self._exportImage(
                        img_type=ImageType.RGB,
                        img_input=self.rgb[i],
                        path=self.ONBOARD_IMG_PATH + "/drone_" + str(i) + "/",
                        frame_num=int(self.step_counter / self.IMG_CAPTURE_FREQ),
                    )
        if self.GUI and self.USER_DEBUG:
            current_input_switch = p.readUserDebugParameter(
                self.INPUT_SWITCH,
                physicsClientId=self.CLIENT,
            )
            if current_input_switch > self.last_input_switch:
                self.last_input_switch = current_input_switch
                self.USE_GUI_RPM = not self.USE_GUI_RPM
        if self.USE_GUI_RPM:
            for i in range(4):
                self.gui_input[i] = p.readUserDebugParameter(
                    int(self.SLIDERS[i]),
                    physicsClientId=self.CLIENT,
                )
            clipped_action = np.tile(self.gui_input, (self.NUM_DRONES, 1))
            if self.step_counter % (self.PYB_FREQ / 2) == 0:
                self.GUI_INPUT_TEXT = [
                    p.addUserDebugText(
                        "Using GUI RPM",
                        textPosition=[0, 0, 0],
                        textColorRGB=[1, 0, 0],
                        lifeTime=1,
                        textSize=2,
                        parentObjectUniqueId=self.DRONE_IDS[i],
                        parentLinkIndex=-1,
                        replaceItemUniqueId=int(self.GUI_INPUT_TEXT[i]),
                        physicsClientId=self.CLIENT,
                    ) for i in range(self.NUM_DRONES)
                ]
        else:
            clipped_action = np.reshape(
                self._preprocessAction(action),
                (self.NUM_DRONES, 4),
            )
        self._update_moving_platform()
        for _ in range(self.PYB_STEPS_PER_CTRL):
            if (
                self.PYB_STEPS_PER_CTRL > 1
                and self.PHYSICS in [
                    Physics.DYN,
                    Physics.PYB_GND,
                    Physics.PYB_DRAG,
                    Physics.PYB_DW,
                    Physics.PYB_GND_DRAG_DW,
                ]
            ):
                self._updateAndStoreKinematicInformation()
            for i in range(self.NUM_DRONES):
                if self.PHYSICS == Physics.PYB:
                    self._physics(clipped_action[i, :], i)
                elif self.PHYSICS == Physics.DYN:
                    self._dynamics(clipped_action[i, :], i)
                elif self.PHYSICS == Physics.PYB_GND:
                    self._physics(clipped_action[i, :], i)
                    self._groundEffect(clipped_action[i, :], i)
                elif self.PHYSICS == Physics.PYB_DRAG:
                    self._physics(clipped_action[i, :], i)
                    self._drag(self.last_clipped_action[i, :], i)
                elif self.PHYSICS == Physics.PYB_DW:
                    self._physics(clipped_action[i, :], i)
                    self._downwash(i)
                elif self.PHYSICS == Physics.PYB_GND_DRAG_DW:
                    self._physics(clipped_action[i, :], i)
                    self._groundEffect(clipped_action[i, :], i)
                    self._drag(self.last_clipped_action[i, :], i)
                    self._downwash(i)
            if self.PHYSICS != Physics.DYN:
                p.stepSimulation(physicsClientId=self.CLIENT)
            self.last_clipped_action = clipped_action
        self._updateAndStoreKinematicInformation()
        self._process_step_updates()
        obs = self._computeObs()
        reward = self._computeReward()
        terminated = self._computeTerminated()
        truncated = self._computeTruncated()
        info = self._computeInfo()
        self.step_counter = self.step_counter + (1 * self.PYB_STEPS_PER_CTRL)
        self._path_monitor_ctrl_step += 1
        if self._path_monitor_ctrl_step % self.PATH_MONITOR_STEP_INTERVAL == 0:
            self._enqueue_path_monitor_snapshot()
        return obs, reward, terminated, truncated, info

    def _process_step_updates(self):
        """Handle post-physics episode bookkeeping exactly once per control step."""
        if self._step_processed:
            return
        self._step_processed = True
        self._time_alive += self._sim_dt
        platform_hit, _ = self._check_collision()
        self._update_landing_state(platform_hit)
        self._update_min_clearance()
        self._apply_distance_cull()

    def _reset_action_buffer(self) -> None:
        """Zero the action history so reset observations do not leak prior episodes."""
        action_dim = int(self.action_space.shape[-1])
        self.action_buffer.clear()
        for _ in range(self.ACTION_BUFFER_SIZE):
            self.action_buffer.append(
                np.zeros((self.NUM_DRONES, action_dim), dtype=np.float32)
            )

    def _spawn_task_world(self):
        """Rebuild the procedural world defined by self.task."""
        self.task.start = self._original_start
        self.task.goal = self._original_goal

        cli = getattr(self, "CLIENT", 0)
        result = build_world(
            seed=self.task.map_seed,
            cli=cli,
            start=self.task.start,
            goal=self.task.goal,
            challenge_type=self.task.challenge_type,
        )

        if len(result) >= 6:
            end_platform_uids, start_platform_uids, start_surface_z, goal_surface_z, adj_start, adj_goal = result
        elif len(result) == 4:
            end_platform_uids, start_platform_uids, start_surface_z, goal_surface_z = result
            adj_start = adj_goal = None
        else:
            end_platform_uids, start_platform_uids = result
            start_surface_z = None
            goal_surface_z = None
            adj_start = adj_goal = None

        self._end_platform_uids = end_platform_uids if end_platform_uids else []
        self._start_platform_uids = start_platform_uids if start_platform_uids else []

        if adj_start is not None:
            self.task.start = adj_start
        if adj_goal is not None:
            self.task.goal = adj_goal

        start_xyz = np.array(self.task.start, dtype=float)

        if start_surface_z is not None:
            self.task.start = (
                self.task.start[0],
                self.task.start[1],
                start_surface_z + START_PLATFORM_TAKEOFF_BUFFER
            )
            start_xyz[2] = start_surface_z + START_PLATFORM_TAKEOFF_BUFFER
            self.GOAL_POS = np.array(self.task.goal, dtype=float)

        if goal_surface_z is not None:
            self.task.goal = (
                self.task.goal[0],
                self.task.goal[1],
                goal_surface_z
            )
            self.GOAL_POS[2] = goal_surface_z

        self._platform_orbit_center = self.GOAL_POS.copy()
        self._current_platform_pos = self.GOAL_POS.copy()
        self._prev_platform_pos = None
        self._platform_offsets = []

        start_quat = p.getQuaternionFromEuler([0.0, 0.0, 0.0])

        p.resetBasePositionAndOrientation(
            self.DRONE_IDS[0],
            start_xyz,
            start_quat,
            physicsClientId=cli,
        )

        plane_id = getattr(self, "PLANE_ID", None)
        if plane_id is not None:
            if int(getattr(self.task, "challenge_type", 0)) == 2:
                # TinyRenderer does not reliably hide the default plane via alpha,
                # so move it far below the custom open-terrain mesh instead.
                p.resetBasePositionAndOrientation(
                    plane_id,
                    [0.0, 0.0, -1000.0],
                    [0.0, 0.0, 0.0, 1.0],
                    physicsClientId=cli,
                )
            p.changeVisualShape(
                plane_id, -1, rgbaColor=[0, 0, 0, 0], physicsClientId=cli,
            )

        self._build_cull_targets()

    # -------- reward ----------------------------------------------------- #
    def _computeReward(self) -> float:
        """Compute incremental reward based on current state."""
        (score, rewards) = flight_reward(
            success=self._success,
            t=(self._t_to_goal if self._success else self._time_alive),
            horizon=self.EP_LEN_SEC,
            task=self.task,
            min_clearance=self._min_clearance_episode,
            collision=self._collision,
            state=self._getDroneStateVector(0),
            state_history=np.array(self.state_history, dtype=np.float64),
        )
        self.reward_history.append(rewards)
        self._schedule_deadline_trunc = bool(rewards.get("schedule_deadline_fail", False))

        r_t = score - self._prev_score
        self._prev_score = score
        return float(r_t)

    # -------- termination ------------------------------------------------ #
    def _computeTerminated(self) -> bool:
        """Return True if episode ended via collision or goal reached."""
        return self._collision or self._success

    # -------- truncation (timeout / safety) ------------------------------ #
    def _computeTruncated(self) -> bool:
        """
        Early termination on excessive tilt or elapsed horizon.
        """
        if self._schedule_deadline_trunc:
            return True
        # safety cut‑off
        state = self._getDroneStateVector(0)
        roll, pitch = state[7], state[8]
        if abs(roll) > self.MAX_TILT_RAD or abs(pitch) > self.MAX_TILT_RAD:
            return True

        # timeout
        return self._time_alive >= self.EP_LEN_SEC

    # -------- extra logging --------------------------------------------- #
    def _path_progress_01(self, pos: np.ndarray) -> float:
        """Projection onto start→goal: 0 at start, 1 at goal (can be <0 or >1 off segment)."""
        s = np.asarray(self.task.start, dtype=np.float64)
        g = np.asarray(self.GOAL_POS, dtype=np.float64)
        path = g - s
        path_len2 = float(np.dot(path, path))
        if path_len2 < 1e-18:
            return 0.0
        return float(np.dot(np.asarray(pos, dtype=np.float64) - s, path) / path_len2)

    def _sync_path_progress_baseline(self) -> None:
        """Call after spawn so TensorBoard delta starts from the actual episode origin."""
        state = self._getDroneStateVector(0)
        self._prev_path_progress_01 = self._path_progress_01(state[0:3])

    def _computeInfo(self):
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        dist = float(np.linalg.norm(pos - self.GOAL_POS))
        pp = self._path_progress_01(pos)
        pp_delta = pp - self._prev_path_progress_01
        self._prev_path_progress_01 = pp
        return {
            "distance_to_goal": dist,
            "path_progress_01": pp,
            "path_progress_delta": float(pp_delta),
            "score": self._prev_score,
            "success": self._success,
            "collision": self._collision,
            "t_to_goal": self._t_to_goal,
            "min_clearance": self._min_clearance_episode,
            "landing_stable_time": self._landing_stable_time,
        }

    # -------- observation extension -------------------------------------- #
    def _computeObs(self):
        """
        Build depth + state observation for the single-drone task.
        Optimized: calls _getDroneImages directly, skips parent class overhead.
        """
        # Get depth directly (skip parent class which would also store rgb/seg)
        _, depth_raw, _ = self._getDroneImages(0)

        if depth_raw is None:
            h, w = (self.IMG_RES[1], self.IMG_RES[0]) if self.IMG_RES is not None else (128, 128)
            state_dim = getattr(self, "_state_dim", 115)
            return {
                "depth": np.zeros((h, w, 1), dtype=np.float32),
                "state": np.zeros((state_dim,), dtype=np.float32),
            }

        # Store in self.dep for compatibility
        self.dep[0] = depth_raw
        depth = self._process_depth(depth_raw)

        state_vec = np.nan_to_num(
            np.asarray(self._getDroneStateVector(0), dtype=np.float32),
            nan=0.0,
            posinf=1e6,
            neginf=-1e6,
        )
        obs_12 = np.hstack([
            state_vec[0:3],
            state_vec[7:10],
            state_vec[10:13],
            state_vec[13:16]
        ]).astype(np.float32)
        self.state_history.append(obs_12)

        state_full = np.array([obs_12], dtype=np.float32)
        for i in range(self.ACTION_BUFFER_SIZE):
            state_full = np.hstack([state_full, np.array([self.action_buffer[i][0, :]])])
        state_full = state_full.flatten().astype(np.float32)

        alt_m = float(self._get_altitude_distance())
        if not np.isfinite(alt_m):
            alt_m = float(MAX_RAY_DISTANCE)
        altitude = float(np.clip(alt_m / MAX_RAY_DISTANCE, 0.0, 1.0))
        state_full = np.append(state_full, altitude).astype(np.float32)

        drone_pos = state_vec[0:3]
        search_area_vector = (self._search_area_center - drone_pos).astype(np.float32)
        search_area_vector = np.nan_to_num(search_area_vector, nan=0.0, posinf=1e6, neginf=-1e6)
        state_full = np.append(state_full, search_area_vector).astype(np.float32)

        state_full = np.nan_to_num(
            state_full, nan=0.0, posinf=1e6, neginf=-1e6
        ).astype(np.float32)

        actual_state_dim = state_full.shape[0]
        if actual_state_dim != self._state_dim:
            self._state_dim = actual_state_dim
            self.observation_space = spaces.Dict({
                "depth": self.observation_space["depth"],
                "state": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(actual_state_dim,),
                    dtype=np.float32
                ),
            })

        depth = np.nan_to_num(depth, nan=1.0, posinf=1.0, neginf=0.0)
        depth = np.clip(depth, 0.0, 1.0).astype(np.float32)

        return {
            "depth": depth,
            "state": state_full,
        }

"""Motrix/MuJoCo playback diagnostics for K1 soccer dribble."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

_BALL_RADIUS_M = 0.09167862683534622
_TOUCH_DIST_M = _BALL_RADIUS_M + 0.10
_KEY_GEOMS = (
    "COL_Collider",
    "ball_geom",
    "ball_visual",
    "left_foot_contact_0_geom",
    "right_foot_contact_0_geom",
)


def _load_mujoco_geom_friction(model_file: Path) -> dict[str, tuple[float, float, float]]:
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(model_file))
    out: dict[str, tuple[float, float, float]] = {}
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        if name is None:
            continue
        friction = model.geom_friction[geom_id]
        out[name] = (float(friction[0]), float(friction[1]), float(friction[2]))
    return out


def _resolve_geom_friction(backend: Any, model_file: Path | None) -> dict[str, tuple[float, float, float]]:
    names = backend.get_geom_names()
    try:
        table = backend.get_geom_friction()
    except NotImplementedError:
        table = None

    if table is not None:
        resolved: dict[str, tuple[float, float, float]] = {}
        for geom_id, name in enumerate(names):
            if not name:
                continue
            row = table[geom_id]
            resolved[name] = (float(row[0]), float(row[1]), float(row[2]))
        return resolved

    if model_file is not None and model_file.is_file():
        return _load_mujoco_geom_friction(model_file)

    return {}


class SoccerDribblePlaybackDiagnostics:
    """Print friction table once, then periodic robot/ball stats during play."""

    def __init__(
        self,
        env: Any,
        *,
        log_every_steps: int = 25,
    ) -> None:
        self._env = env
        self._log_every_steps = max(int(log_every_steps), 1)
        self._step = 0
        self._printed_setup = False
        self._prev_ball_vel_w = np.zeros(3, dtype=np.float64)
        self._ctrl_dt = float(env.cfg.ctrl_dt)

    @staticmethod
    def maybe_create(
        env: Any,
        *,
        enabled: bool,
        log_every_steps: int = 25,
    ) -> SoccerDribblePlaybackDiagnostics | None:
        if not enabled:
            return None
        if type(env).__name__ != "K1SoccerDribbleEnv":
            return None
        return SoccerDribblePlaybackDiagnostics(env, log_every_steps=log_every_steps)

    def print_setup_once(self) -> None:
        if self._printed_setup:
            return
        self._printed_setup = True

        backend = self._env._backend
        model_file = Path(str(self._env.cfg.scene.model_file))
        friction = _resolve_geom_friction(backend, model_file)

        print("[soccer-debug] === playback physics snapshot ===")
        print(f"[soccer-debug] model_file={model_file}")
        print(f"[soccer-debug] ctrl_dt={self._ctrl_dt:.4f}s log_every={self._log_every_steps} steps")
        print(
            f"[soccer-debug] sim_dt={float(self._env.cfg.sim_dt):.4f}s "
            f"ctrl_dt={self._ctrl_dt:.4f}s decimation="
            f"{int(round(self._ctrl_dt / float(self._env.cfg.sim_dt)))}"
        )
        print("[soccer-debug] geom friction (slide, spin, roll):")
        for name in _KEY_GEOMS:
            coeffs = friction.get(name)
            if coeffs is None:
                print(f"[soccer-debug]   {name}: <missing>")
            else:
                print(
                    f"[soccer-debug]   {name}: "
                    f"slide={coeffs[0]:.4f} spin={coeffs[1]:.4f} roll={coeffs[2]:.6f}"
                )
        for name, coeffs in sorted(friction.items()):
            if name in _KEY_GEOMS:
                continue
            if "ball" in name.lower() or "foot" in name.lower() or "collider" in name.lower():
                print(
                    f"[soccer-debug]   {name}: "
                    f"slide={coeffs[0]:.4f} spin={coeffs[1]:.4f} roll={coeffs[2]:.6f}"
                )

        try:
            gravity = backend.get_gravity()
            print(f"[soccer-debug] gravity={gravity}")
        except NotImplementedError:
            pass

        print(
            f"[soccer-debug] ball nominal: radius={_BALL_RADIUS_M:.4f} m, "
            f"touch_dist_heuristic<{_TOUCH_DIST_M:.3f} m (center-to-foot)"
        )

    def sync_ball_velocity_baseline(self) -> None:
        _, ball_vel_w, _, _ = self._env._ball_state()
        self._prev_ball_vel_w = ball_vel_w[0].astype(np.float64, copy=True)

    def after_step(self, info: dict[str, Any] | None) -> None:
        self._step += 1
        if self._step % self._log_every_steps != 0:
            return

        ball_pos_w, ball_vel_w, rel_pos_b, rel_vel_b = self._env._ball_state()
        base_lin_vel = self._env._backend.get_base_lin_vel()[0]
        base_pos = self._env._backend.get_base_pos()[0]

        left_foot = self._env._backend.get_sensor_data("left_foot_pos")[0]
        right_foot = self._env._backend.get_sensor_data("right_foot_pos")[0]
        dist_left = float(np.linalg.norm(ball_pos_w[0] - left_foot))
        dist_right = float(np.linalg.norm(ball_pos_w[0] - right_foot))
        touch_left = dist_left < _TOUCH_DIST_M
        touch_right = dist_right < _TOUCH_DIST_M

        ball_v = ball_vel_w[0]
        ball_acc = (ball_v - self._prev_ball_vel_w) / self._ctrl_dt
        self._prev_ball_vel_w = ball_v.astype(np.float64, copy=True)

        ball_speed_xy = float(np.linalg.norm(ball_v[:2]))
        rel_speed_xy = float(np.linalg.norm(rel_vel_b[0, :2]))
        dist_xy = float(np.linalg.norm(rel_pos_b[0, :2]))

        commands = info.get("commands") if info is not None else None
        if commands is None:
            cmd = np.zeros(3, dtype=np.float32)
        else:
            cmd = np.asarray(commands[0], dtype=np.float32)

        sim_time = self._step * self._ctrl_dt
        print(
            f"[soccer-debug] step={self._step:5d} t={sim_time:6.2f}s | "
            f"base_pos=[{base_pos[0]:+.3f},{base_pos[1]:+.3f},{base_pos[2]:+.3f}] "
            f"base_v=[{base_lin_vel[0]:+.3f},{base_lin_vel[1]:+.3f},{base_lin_vel[2]:+.3f}] "
            f"cmd=[{cmd[0]:+.3f},{cmd[1]:+.3f},{cmd[2]:+.3f}]"
        )
        print(
            f"[soccer-debug]   ball_w: pos=[{ball_pos_w[0,0]:+.3f},{ball_pos_w[0,1]:+.3f},{ball_pos_w[0,2]:+.3f}] "
            f"vel=[{ball_v[0]:+.3f},{ball_v[1]:+.3f},{ball_v[2]:+.3f}] "
            f"|v|_xy={ball_speed_xy:.3f} acc=[{ball_acc[0]:+.2f},{ball_acc[1]:+.2f},{ball_acc[2]:+.2f}] m/s^2"
        )
        print(
            f"[soccer-debug]   rel_body: dist_xy={dist_xy:.3f} "
            f"rel_pos=[{rel_pos_b[0,0]:+.3f},{rel_pos_b[0,1]:+.3f},{rel_pos_b[0,2]:+.3f}] "
            f"rel_vel=[{rel_vel_b[0,0]:+.3f},{rel_vel_b[0,1]:+.3f},{rel_vel_b[0,2]:+.3f}] "
            f"|rel_v|_xy={rel_speed_xy:.3f}"
        )
        touch_flags = []
        if touch_left:
            touch_flags.append("L")
        if touch_right:
            touch_flags.append("R")
        touch_txt = ",".join(touch_flags) if touch_flags else "-"
        print(
            f"[soccer-debug]   feet: d_left={dist_left:.3f} d_right={dist_right:.3f} "
            f"touch={touch_txt} (heuristic)"
        )

        if info is not None and "ball_dist_xy" in info:
            print(f"[soccer-debug]   env ball_dist_xy={float(np.mean(info['ball_dist_xy'])):.3f}")

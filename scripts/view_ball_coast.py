"""Motrix 可视化：机器人零动作，仅给足球初速度（无 policy、无踢球）。

用法（打开 3D 窗口，关窗退出）:
  CUDA_HOME=... LD_LIBRARY_PATH=... uv run scripts/view_ball_coast.py \\
    --visual --until-close --vx 0.85

球放远、钉住机器人姿态（只看球在地面上的行为）:
  uv run scripts/view_ball_coast.py --visual --until-close --vx -0.8 --ball-x 2.5 --pin-robot

无窗口 + 日志:
  uv run scripts/view_ball_coast.py --steps 500 --vx 0.85 --log-every 25
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from unilab.base.registry import ensure_registries
from unilab.visualization.soccer_dribble_playback import (
    create_soccer_dribble_env,
    motrix_camera_kwargs,
    soccer_debug_from_cfg,
)


def _ball_dof_layout(model_file: Path) -> tuple[int, int, int, int]:
    """Return (qpos_adr, qpos_len, qvel_adr, qvel_len) for ball free joint."""
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(model_file))
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball-root")
    if joint_id < 0:
        raise RuntimeError("joint 'ball-root' not found in scene")
    qpos_adr = int(model.jnt_qposadr[joint_id])
    qvel_adr = int(model.jnt_dofadr[joint_id])
    return qpos_adr, 7, qvel_adr, 6


def _apply_ball_launch(
    env,
    *,
    vx: float,
    vy: float,
    vz: float,
    ball_x: float | None,
    ball_y: float | None,
) -> None:
    model_file = Path(str(env.cfg.scene.model_file))
    qpos_adr, qpos_len, qvel_adr, qvel_len = _ball_dof_layout(model_file)
    env_ids = np.array([0], dtype=np.int32)
    backend = env._backend
    data0 = backend._data[0]
    qpos = np.asarray(data0.dof_pos, dtype=np.float32).reshape(1, -1).copy()
    qvel = np.asarray(data0.dof_vel, dtype=np.float32).reshape(1, -1).copy()

    if ball_x is not None:
        qpos[0, qpos_adr] = float(ball_x)
    if ball_y is not None:
        qpos[0, qpos_adr + 1] = float(ball_y)

    qvel[0, qvel_adr : qvel_adr + 3] = (float(vx), float(vy), float(vz))
    qvel[0, qvel_adr + 3 : qvel_adr + qvel_len] = 0.0
    backend.set_state(env_ids, qpos, qvel)


def _pin_robot_snapshot(env) -> tuple[np.ndarray, np.ndarray, int, int]:
    model_file = Path(str(env.cfg.scene.model_file))
    qpos_adr, _, qvel_adr, _ = _ball_dof_layout(model_file)
    data0 = env._backend._data[0]
    qpos = np.asarray(data0.dof_pos, dtype=np.float32).reshape(-1).copy()
    qvel = np.asarray(data0.dof_vel, dtype=np.float32).reshape(-1).copy()
    robot_qpos = qpos[:qpos_adr].copy()
    robot_qvel = np.zeros_like(qvel[:qvel_adr])
    return robot_qpos, robot_qvel, qpos_adr, qvel_adr


def _restore_pinned_robot(env, robot_qpos: np.ndarray, robot_qvel: np.ndarray, qpos_adr: int, qvel_adr: int) -> None:
    env_ids = np.array([0], dtype=np.int32)
    backend = env._backend
    data0 = backend._data[0]
    qpos = np.asarray(data0.dof_pos, dtype=np.float32).reshape(1, -1).copy()
    qvel = np.asarray(data0.dof_vel, dtype=np.float32).reshape(1, -1).copy()
    qpos[0, :qpos_adr] = robot_qpos
    qvel[0, :qvel_adr] = robot_qvel
    backend.set_state(env_ids, qpos, qvel)


def main() -> int:
    parser = argparse.ArgumentParser(description="Ball-only coast visual (zero robot action)")
    parser.add_argument("--vx", type=float, default=0.85, help="Ball initial world vx (m/s)")
    parser.add_argument("--vy", type=float, default=0.0, help="Ball initial world vy (m/s)")
    parser.add_argument("--vz", type=float, default=0.0, help="Ball initial world vz (m/s)")
    parser.add_argument("--ball-x", type=float, default=None, help="Override ball x after reset (m)")
    parser.add_argument("--ball-y", type=float, default=None, help="Override ball y after reset (m)")
    parser.add_argument(
        "--pin-robot",
        action="store_true",
        help="Freeze robot qpos/qvel each step; only ball dynamics change",
    )
    parser.add_argument("--visual", action="store_true", help="Open Motrix render window")
    parser.add_argument("--until-close", action="store_true", help="With --visual: run until window closes")
    parser.add_argument("--steps", type=int, default=2000, help="Steps when not using --until-close")
    parser.add_argument("--log-every", type=int, default=25)
    args = parser.parse_args()

    if args.until_close and not args.visual:
        raise ValueError("--until-close requires --visual")

    ensure_registries()
    conf_dir = (ROOT / "conf" / "appo").resolve()
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(conf_dir), version_base="1.3"):
        cfg: DictConfig = compose(
            config_name="config",
            overrides=[
                "task=k1_soccer_dribble/motrix",
                "training.sim_backend=motrix",
                "interactive.soccer_dribble_debug=true",
            ],
        )

    env = create_soccer_dribble_env(cfg, num_envs=1, root_dir=ROOT)
    env.init_state()
    env.set_autoreset(False)
    debug = soccer_debug_from_cfg(env, cfg, log_every_steps=args.log_every)
    if debug is None:
        raise RuntimeError("soccer debug diagnostics required")

    action_dim = int(env.action_space.shape[0])
    zero_action = np.zeros((1, action_dim), dtype=np.float32)
    pin: tuple[np.ndarray, np.ndarray, int, int] | None = None

    def _launch_after_reset() -> None:
        nonlocal pin
        _apply_ball_launch(
            env,
            vx=args.vx,
            vy=args.vy,
            vz=args.vz,
            ball_x=args.ball_x,
            ball_y=args.ball_y,
        )
        if args.pin_robot:
            pin = _pin_robot_snapshot(env)
        debug.sync_ball_velocity_baseline()

    def _initialize() -> np.ndarray:
        env.reset(np.array([0], dtype=np.int32))
        _launch_after_reset()
        debug.print_setup_once()
        print(
            f"[ball-coast] zero action | v0=({args.vx:+.3f}, {args.vy:+.3f}, {args.vz:+.3f}) "
            f"ball_x={args.ball_x} ball_y={args.ball_y} pin_robot={args.pin_robot}"
        )
        if env.state is None:
            raise RuntimeError("env.state missing after reset")
        return np.asarray(env.state.obs["obs"], dtype=np.float32)

    def _step(_obs_np: np.ndarray) -> np.ndarray:
        state = env.step(zero_action)
        if pin is not None:
            _restore_pinned_robot(env, pin[0], pin[1], pin[2], pin[3])
        debug.after_step(state.info)
        return np.asarray(state.obs["obs"], dtype=np.float32)

    ctrl_dt = float(env.cfg.ctrl_dt)
    if args.visual:
        play_steps = None if args.until_close else args.steps
        sim_s = "∞" if play_steps is None else f"{play_steps * ctrl_dt:.1f}"
        print(f"[ball-coast] Motrix window — close to quit (~{sim_s}s sim if finite).")
        try:
            env.run_playback_mode(
                play_render_mode=getattr(cfg.training, "play_render_mode", "auto"),
                play_steps=play_steps,
                output_video=None,
                initialize=_initialize,
                step=_step,
                render_spacing=float(getattr(cfg.training, "render_spacing", 1.0)),
                camera_kwargs=motrix_camera_kwargs(cfg),
            )
        except Exception as exc:
            if "RenderClosedError" in type(exc).__name__:
                print("[ball-coast] render window closed.")
            else:
                raise
    else:
        _initialize()
        for _ in range(args.steps):
            _step(np.zeros(1, dtype=np.float32))
        print(f"[ball-coast] headless done ({args.steps} steps, ~{args.steps * ctrl_dt:.1f}s).")

    print("[ball-coast] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

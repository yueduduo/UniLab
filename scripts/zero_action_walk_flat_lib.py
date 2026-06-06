"""Shared helpers: G1/K1 walk-flat env with action=0 (hold default pose)."""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from unilab.training import ensure_registries

ensure_registries()

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from unilab.envs.locomotion.common.rewards import RewardContext, weighted_pose
from unilab.training.backend_adapter import BackendAdapter
from unilab.training.common import create_env

ROBOTS: dict[str, dict[str, str]] = {
    "g1": {
        "label": "G1",
        "task_name": "G1WalkFlat",
        "hydra_task": "flashsac/g1_walk_flat/mujoco",
    },
    "k1": {
        "label": "K1",
        "task_name": "K1WalkFlat",
        "hydra_task": "flashsac/k1_walk_flat/mujoco",
    },
}

DEFAULT_OUT_DIR = ROOT_DIR / "logs" / "zero_action_walk_flat"


def build_parser(robot_key: str) -> argparse.ArgumentParser:
    meta = ROBOTS[robot_key]
    parser = argparse.ArgumentParser(
        description=f"{meta['label']} walk-flat: zero actions (target = default pose), visualize drift."
    )
    parser.add_argument(
        "--visual",
        action="store_true",
        help="Open MuJoCo passive viewer (close window or Esc to quit).",
    )
    parser.add_argument(
        "--video",
        action="store_true",
        help="Render offscreen video/GIF (no display required).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=400,
        help="Simulation steps after init (default: 400).",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="Print metrics every N steps (default: 10).",
    )
    parser.add_argument(
        "--video-out",
        type=Path,
        default=None,
        help="Output path for --video (.mp4 if ffmpeg plugin available, else .gif).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=50.0,
        help="Video FPS (default: 50 ≈ 1/ctrl_dt).",
    )
    parser.add_argument(
        "--quiet-stand",
        action="store_true",
        help="Minimal reset perturbation (reset_base_qvel_limit=0, recipe DR off).",
    )
    return parser


def compose_walk_cfg(robot_key: str, *, quiet_stand: bool) -> DictConfig:
    conf_dir = str((ROOT_DIR / "conf" / "offpolicy").resolve())
    hydra_task = ROBOTS[robot_key]["hydra_task"]
    overrides = [f"task={hydra_task}"]
    if quiet_stand:
        overrides.extend(
            [
                "env.reset_base_qvel_limit=0",
                "env.domain_rand.push_robots=false",
            ]
        )
    with initialize_config_dir(version_base=None, config_dir=conf_dir):
        return compose(config_name="config", overrides=["algo=flashsac", *overrides])


def make_walk_env(cfg: DictConfig, *, num_envs: int = 1) -> Any:
    adapter = BackendAdapter(cfg, root_dir=ROOT_DIR, algo_name="flashsac")
    env_cfg_override = adapter.build_task_env_cfg_override()
    return create_env(
        cfg,
        num_envs=num_envs,
        env_cfg_override=env_cfg_override,
    )


def _tilt_deg_from_gravity(gravity: np.ndarray) -> np.ndarray:
    g = np.asarray(gravity, dtype=np.float64)
    cos_tilt = np.clip(g[:, 2], -1.0, 1.0)
    return np.rad2deg(np.arccos(cos_tilt))


def measure_state(env: Any, state: Any) -> dict[str, float]:
    linvel = env.get_local_linvel()
    gyro = env.get_gyro()
    gravity = env._backend.get_sensor_data(env._cfg.sensor.upvector)
    dof_pos = env.get_dof_pos()
    dof_vel = env.get_dof_vel()
    info = state.info
    ctx = RewardContext(
        info=info,
        linvel=linvel,
        gyro=gyro,
        dof_pos=dof_pos,
        num_envs=env._num_envs,
        default_angles=env.default_angles,
        pose_weights=np.asarray(env._reward_cfg.pose_weights, dtype=np.float64),
    )
    pose_raw = weighted_pose(ctx)
    scales = env._reward_cfg.scales
    pose_scale = float(scales["pose"] if isinstance(scales, dict) else scales.pose)
    log = info.get("log", {})
    return {
        "total_reward": float(np.mean(state.reward)),
        "base_z": float(np.mean(env._backend.get_base_pos()[:, 2])),
        "tilt_deg": float(np.mean(_tilt_deg_from_gravity(gravity))),
        "pose_raw": float(np.mean(pose_raw)),
        "pose_weighted": float(np.mean(pose_raw) * pose_scale),
        "terminated_frac": float(np.mean(state.terminated)),
        "log_pose": float(log.get("reward/pose", float("nan"))),
        "log_penalty_action_rate": float(log.get("reward/penalty_action_rate", float("nan"))),
    }


def print_metrics_row(step: int, m: dict[str, float], *, label: str) -> None:
    print(
        f"[{label}] step={step:4d}  reward={m['total_reward']:7.3f}  "
        f"base_z={m['base_z']:.4f}  tilt={m['tilt_deg']:5.1f}°  "
        f"pose(w)={m['pose_weighted']:7.3f}  term={m['terminated_frac']:.3f}"
    )


def run_zero_action_loop(
    env: Any,
    *,
    steps: int,
    log_every: int,
    label: str,
    on_step: Any | None = None,
) -> list[dict[str, float]]:
    num_envs = env._num_envs
    zero = np.zeros((num_envs, env._num_action), dtype=np.float32)
    env.init_state()
    history: list[dict[str, float]] = []
    for step in range(steps + 1):
        if step == 0:
            state = env._state
        else:
            state = env.step(zero)
        m = measure_state(env, state)
        history.append(m)
        if step % log_every == 0:
            print_metrics_row(step, m, label=label)
        if on_step is not None:
            on_step(step, state, m)
    return history


def _mujoco_viz_paths(env: Any) -> tuple[Any, Any, Any, int, int]:
    import mujoco

    scene = env.cfg.scene
    backend = env._backend
    parent_xml = getattr(backend, "scene_visual_model_file", None) or scene.model_file
    model = mujoco.MjModel.from_xml_path(str(parent_xml))
    data = mujoco.MjData(model)
    nq = int(model.nq)
    nv = int(model.nv)
    state_spec = mujoco.mjtState.mjSTATE_FULLPHYSICS
    return model, data, state_spec, nq, nv


def _base_body_id(model: Any) -> int:
    import mujoco

    for name in ("pelvis", "Trunk", "base_link"):
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id >= 0:
            return body_id
    return 1


def _camera_on_robot(model: Any, data: Any, *, distance: float = 3.0) -> Any:
    import mujoco

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    body_id = _base_body_id(model)
    camera.lookat[:] = data.xpos[body_id]
    camera.distance = distance
    camera.azimuth = 120.0
    camera.elevation = -15.0
    return camera


def _sync_decoder_to_viz(
    env: Any,
    decoder_model: Any,
    decoder_data: Any,
    viz_model: Any,
    viz_data: Any,
    state_spec: Any,
    nq: int,
    nv: int,
) -> int:
    import mujoco

    phys = env.get_physics_state_snapshot()
    mujoco.mj_setState(decoder_model, decoder_data, phys[0].astype(np.float64), state_spec)
    viz_data.qpos[:nq] = decoder_data.qpos
    viz_data.qvel[:nv] = decoder_data.qvel
    mujoco.mj_forward(viz_model, viz_data)
    return _base_body_id(viz_model)


def run_visual(env: Any, *, steps: int, log_every: int, label: str) -> None:
    import mujoco
    import mujoco.viewer

    model, data, state_spec, nq, nv = _mujoco_viz_paths(env)
    zero = np.zeros((env._num_envs, env._num_action), dtype=np.float32)
    ctrl_dt = float(env.cfg.ctrl_dt)
    env.init_state()

    parent_xml = getattr(env._backend, "scene_visual_model_file", None) or env.cfg.scene.model_file
    decoder_model = mujoco.MjModel.from_xml_path(str(parent_xml))
    decoder_data = mujoco.MjData(decoder_model)

    print(f"[{label}] MuJoCo viewer — action=0, close window to stop.")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        step = 0
        while viewer.is_running() and step <= steps:
            t0 = time.perf_counter()
            if step == 0:
                state = env._state
            else:
                state = env.step(zero)
            if step % log_every == 0:
                print_metrics_row(step, measure_state(env, state), label=label)
            body_id = _sync_decoder_to_viz(
                env, decoder_model, decoder_data, model, data, state_spec, nq, nv
            )
            viewer.cam.lookat[:] = data.xpos[body_id]
            viewer.sync()
            step += 1
            sleep = ctrl_dt - (time.perf_counter() - t0)
            if sleep > 0:
                time.sleep(sleep)


def run_video(
    env: Any,
    *,
    steps: int,
    log_every: int,
    label: str,
    out_path: Path,
    fps: float,
) -> Path:
    import imageio.v2 as imageio
    import mujoco

    out_path.parent.mkdir(parents=True, exist_ok=True)
    model, data, state_spec, nq, nv = _mujoco_viz_paths(env)
    parent_xml = getattr(env._backend, "scene_visual_model_file", None) or env.cfg.scene.model_file
    decoder_model = mujoco.MjModel.from_xml_path(str(parent_xml))
    decoder_data = mujoco.MjData(decoder_model)
    renderer = mujoco.Renderer(model, height=480, width=640)
    camera = _camera_on_robot(model, data)
    option = mujoco.MjvOption()
    zero = np.zeros((env._num_envs, env._num_action), dtype=np.float32)
    env.init_state()
    frames: list[np.ndarray] = []

    for step in range(steps + 1):
        if step == 0:
            state = env._state
        else:
            state = env.step(zero)
        if step % log_every == 0:
            print_metrics_row(step, measure_state(env, state), label=label)
        body_id = _sync_decoder_to_viz(
            env, decoder_model, decoder_data, model, data, state_spec, nq, nv
        )
        camera.lookat[:] = data.xpos[body_id]
        renderer.update_scene(data, camera=camera, scene_option=option)
        frames.append(renderer.render().copy())

    try:
        imageio.mimsave(out_path, frames, fps=fps)
    except ValueError as exc:
        if out_path.suffix.lower() != ".gif":
            gif_path = out_path.with_suffix(".gif")
            imageio.mimsave(gif_path, frames, fps=fps)
            print(f"[{label}] MP4 backend unavailable ({exc}); wrote GIF: {gif_path}")
            return gif_path
        raise
    print(f"[{label}] Wrote video: {out_path}")
    return out_path


def main_for_robot(robot_key: str, argv: Sequence[str] | None = None) -> int:
    if robot_key not in ROBOTS:
        raise SystemExit(f"Unknown robot {robot_key!r}")
    meta = ROBOTS[robot_key]
    args = build_parser(robot_key).parse_args(argv)
    if not args.visual and not args.video:
        args.visual = True
    if args.steps < 0:
        raise SystemExit("--steps must be >= 0")

    cfg = compose_walk_cfg(robot_key, quiet_stand=args.quiet_stand)
    env = make_walk_env(cfg, num_envs=1)
    label = meta["label"]
    try:
        if args.visual:
            run_visual(env, steps=args.steps, log_every=args.log_every, label=label)
        elif args.video:
            out = args.video_out or (DEFAULT_OUT_DIR / f"{robot_key}_zero_action.gif")
            run_video(
                env,
                steps=args.steps,
                log_every=args.log_every,
                label=label,
                out_path=out,
                fps=args.fps,
            )
        else:
            run_zero_action_loop(
                env, steps=args.steps, log_every=args.log_every, label=label
            )
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()
    return 0

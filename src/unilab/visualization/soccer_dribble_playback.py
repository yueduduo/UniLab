"""Shared K1 soccer dribble APPO playback (probe + play_interactive)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict

from unilab.base.backend.mujoco.xml import materialize_scene_visual_override
from unilab.training import BackendAdapter, create_env
from unilab.visualization.interactive_playback import (
    KeyboardCommander,
    build_appo_actor,
    build_appo_inference_policy,
    build_motrix_keyboard_before_step,
    load_algo_config_from_run_dir,
    print_play_keyboard_legend,
    sync_play_velocity_command,
)
from unilab.visualization.soccer_dribble_playback_debug import SoccerDribblePlaybackDiagnostics

SOCCER_DRIBBLE_TASK_NAME = "K1SoccerDribble"


def is_soccer_dribble_task(task_name: str) -> bool:
    return str(task_name) == SOCCER_DRIBBLE_TASK_NAME


def create_soccer_dribble_env(
    cfg: DictConfig,
    *,
    num_envs: int = 1,
    root_dir: str | Path,
    algo_name: str = "appo",
) -> Any:
    """Match ``play_interactive`` env factory: reward + ``cfg.env`` overrides."""
    adapter = BackendAdapter(
        cfg,
        root_dir=Path(root_dir),
        algo_name=algo_name,
        scene_materializer=materialize_scene_visual_override,
    )
    env = create_env(
        cfg,
        num_envs=num_envs,
        env_cfg_override=adapter.build_task_env_cfg_override(),
    )
    if type(env).__name__ != "K1SoccerDribbleEnv":
        raise TypeError(f"expected K1SoccerDribbleEnv, got {type(env).__name__}")
    return env


def load_soccer_dribble_policy(
    env: Any,
    checkpoint_path: str | Path,
    cfg: DictConfig,
    *,
    device: str,
    num_envs: int = 1,
) -> Any:
    checkpoint_path = Path(checkpoint_path)
    rl_cfg = load_algo_config_from_run_dir(str(checkpoint_path)) or OmegaConf.to_container(
        cfg.algo, resolve=True
    )
    obs_dim = int(env.obs_groups_spec["obs"])
    critic_dim = int(env.obs_groups_spec.get("critic", 0))
    action_dim = int(env.action_space.shape[0])

    actor = build_appo_actor(
        rl_cfg,
        obs_dim=obs_dim,
        critic_dim=critic_dim,
        action_dim=action_dim,
        num_envs=num_envs,
        device=device,
    )
    loaded = torch.load(checkpoint_path, map_location=device, weights_only=True)
    actor.load_state_dict(loaded["actor"])
    return build_appo_inference_policy(actor, device=device)


def policy_action_np(
    policy: Any,
    obs_np: np.ndarray,
    *,
    device: str,
    action_dim: int,
    num_envs: int = 1,
) -> np.ndarray:
    """Same TensorDict layout as ``play_interactive_motrix`` raw-env path."""
    obs_t = torch.from_numpy(np.asarray(obs_np, dtype=np.float32)).to(device)
    if obs_t.ndim == 1:
        obs_t = obs_t.unsqueeze(0)
    with torch.inference_mode():
        actions = policy(
            TensorDict(
                {"policy": obs_t},
                batch_size=num_envs,
            )
        )
    return np.asarray(actions.detach().cpu().numpy(), dtype=np.float32).reshape(num_envs, action_dim)


def step_soccer_dribble(
    env: Any,
    policy: Any,
    obs_np: np.ndarray,
    *,
    device: str,
    action_dim: int,
    num_envs: int = 1,
) -> tuple[np.ndarray, dict[str, Any]]:
    action_np = policy_action_np(
        policy,
        obs_np,
        device=device,
        action_dim=action_dim,
        num_envs=num_envs,
    )
    state = env.step(action_np)
    next_obs = np.asarray(state.obs["obs"], dtype=np.float32)
    return next_obs, state.info


def motrix_camera_kwargs(cfg: DictConfig) -> dict[str, Any]:
    """Defaults follow ``conf/appo/config.yaml`` (``cam_tracking: false``)."""
    return {
        "cam_distance": getattr(cfg.training, "cam_distance", 6.0),
        "cam_elevation": getattr(cfg.training, "cam_elevation", -20.0),
        "cam_azimuth": getattr(cfg.training, "cam_azimuth", 90.0),
        "cam_lookat": getattr(cfg.training, "cam_lookat", None),
        "cam_tracking": getattr(cfg.training, "cam_tracking", False),
        "cam_tracking_env_idx": getattr(cfg.training, "cam_tracking_env_idx", 0),
        "cam_tracking_extra_envs": getattr(cfg.training, "cam_tracking_extra_envs", 2),
    }


def soccer_debug_from_cfg(
    env: Any,
    cfg: DictConfig | None,
    *,
    log_every_steps: int = 50,
) -> SoccerDribblePlaybackDiagnostics | None:
    if cfg is None:
        return SoccerDribblePlaybackDiagnostics.maybe_create(env, enabled=True)
    override = OmegaConf.select(cfg, "interactive.soccer_dribble_debug", default=None)
    enabled = bool(override) if override is not None else True
    return SoccerDribblePlaybackDiagnostics.maybe_create(
        env,
        enabled=enabled,
        log_every_steps=log_every_steps,
    )


def _build_keyboard_before_step(
    env: Any,
    *,
    keyboard: bool,
    keyboard_step_lin: float,
    keyboard_step_ang: float,
    log_prefix: str,
) -> Any | None:
    if not keyboard:
        return None

    state = env.state
    command_arr = state.info.get("commands") if state is not None else None
    cmds_cfg = getattr(getattr(env, "cfg", None), "commands", None)
    if not isinstance(command_arr, np.ndarray) or cmds_cfg is None:
        print(f"{log_prefix} interactive.keyboard ignored: task has no velocity 'commands'.")
        return None

    cmds_cfg.heading_command = False
    cmds_cfg.resampling_time = 0.0
    commander = KeyboardCommander.from_vel_limit(
        cmds_cfg.vel_limit,
        step_lin=keyboard_step_lin,
        step_ang=keyboard_step_ang,
    )
    before_step = build_motrix_keyboard_before_step(
        env,
        env._backend,
        commander,
        log_command=lambda message: print(message),
    )
    env.set_autoreset(False)
    print_play_keyboard_legend(action_mode="policy")
    return before_step


def run_headless_soccer_dribble_playback(
    env: Any,
    policy: Any,
    debug: SoccerDribblePlaybackDiagnostics,
    *,
    device: str,
    steps: int,
    num_envs: int = 1,
) -> None:
    obs, _info = env.reset(np.arange(num_envs, dtype=np.int32))
    debug.sync_ball_velocity_baseline()
    obs_np = np.asarray(obs["obs"], dtype=np.float32)
    action_dim = int(env.action_space.shape[0])

    for _ in range(steps):
        obs_np, info = step_soccer_dribble(
            env,
            policy,
            obs_np,
            device=device,
            action_dim=action_dim,
            num_envs=num_envs,
        )
        debug.after_step(info)


def run_motrix_soccer_dribble_playback(
    env: Any,
    policy: Any,
    cfg: DictConfig,
    *,
    device: str,
    debug: SoccerDribblePlaybackDiagnostics | None,
    keyboard: bool,
    keyboard_step_lin: float = 0.1,
    keyboard_step_ang: float = 0.2,
    until_close: bool = False,
    play_steps: int | None = None,
    log_prefix: str = "[play]",
    num_envs: int = 1,
) -> None:
    """Motrix visual loop aligned with ``play_interactive_motrix`` for K1SoccerDribble."""
    action_dim = int(env.action_space.shape[0])
    before_step = _build_keyboard_before_step(
        env,
        keyboard=keyboard,
        keyboard_step_lin=keyboard_step_lin,
        keyboard_step_ang=keyboard_step_ang,
        log_prefix=log_prefix,
    )
    if before_step is not None:
        env.set_autoreset(False)
    elif until_close:

        def _noop_before_step() -> None:
            return None

        before_step = _noop_before_step

    def _initialize() -> np.ndarray:
        env.reset(np.arange(num_envs, dtype=np.int32))
        if before_step is not None:
            sync_play_velocity_command(
                env,
                np.zeros(3, dtype=np.float32),
                flush_obs_history=True,
            )
        if debug is not None:
            debug.print_setup_once()
            debug.sync_ball_velocity_baseline()
        if env.state is None:
            raise RuntimeError("play initialize requires env.state after reset")
        return np.asarray(env.state.obs["obs"], dtype=np.float32)

    def _step(obs_np: np.ndarray) -> np.ndarray:
        next_obs, info = step_soccer_dribble(
            env,
            policy,
            obs_np,
            device=device,
            action_dim=action_dim,
            num_envs=num_envs,
        )
        if debug is not None:
            debug.after_step(info)
        return next_obs

    effective_play_steps = None if until_close else play_steps
    play_render_mode = getattr(cfg.training, "play_render_mode", "auto")

    print(f"{log_prefix} Opening Motrix render window — close to quit.")
    if keyboard:
        print_play_keyboard_legend(action_mode="policy")

    try:
        env.run_playback_mode(
            play_render_mode=play_render_mode,
            play_steps=effective_play_steps,
            output_video=None,
            initialize=_initialize,
            step=_step,
            render_spacing=float(getattr(cfg.training, "render_spacing", 1.0)),
            camera_kwargs=motrix_camera_kwargs(cfg),
            before_step=before_step,
        )
    except Exception as exc:
        if "RenderClosedError" in type(exc).__name__:
            print(f"{log_prefix} Render window closed.")
        else:
            raise

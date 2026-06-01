"""K1 joystick flat locomotion with AMP-compatible stacked observations."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.backend import create_backend
from unilab.base.np_env import NpEnvState
from unilab.base.scene import SceneCfg
from unilab.dtype_config import get_global_dtype
from unilab.envs.locomotion.common import rewards
from unilab.envs.locomotion.common.commands import (
    Commands,
    sample_heading_commands,
    zero_small_xy_commands,
)
from unilab.envs.locomotion.common.domain_rand import DomainRandConfig
from unilab.envs.locomotion.common.dr_provider import LocomotionDRProvider
from unilab.envs.locomotion.common.rewards import RewardContext
from unilab.envs.locomotion.g1.joystick import (
    compute_aggregated_foot_contact,
    compute_feet_phase_contact_targets,
    compute_feet_phase_height_targets,
    compute_forward_command_mask,
    compute_forward_speed_gate,
)
from unilab.envs.locomotion.k1.base import K1BaseCfg, K1BaseEnv
from unilab.envs.locomotion.k1.constants import (
    K1_NUM_ACTION,
    K1_OBS_FRAME_STACK,
    K1_OBS_SINGLE_DIM,
    K1_OBS_STACKED_DIM,
)

LEFT_FOOT_CONTACT_SENSORS = ["left_foot_contact_0"]
RIGHT_FOOT_CONTACT_SENSORS = ["right_foot_contact_0"]


@dataclass
class InitState:
    pos = [0.0, 0.0, 0.55]


def commands_to_amp_obs(commands: np.ndarray) -> np.ndarray:
    """Map physical velocity commands to AMP deploy obs (body-frame flip + yaw scale)."""
    return np.stack(
        (
            -commands[:, 0],
            -commands[:, 1],
            -commands[:, 2] * 0.25,
        ),
        axis=1,
        dtype=get_global_dtype(),
    )


@dataclass
class K1RewardConfig:
    scales: dict[str, float]
    tracking_sigma: float
    gait_frequency: float
    feet_phase_swing_height: float
    feet_phase_tracking_sigma: float
    base_height_target: float
    min_base_height: float
    max_tilt_deg: float
    min_forward_speed_for_gait_reward: float = 0.05
    close_feet_threshold: float = 0.12
    pose_weights: list[float] = field(
        default_factory=lambda: [1.0] * 10 + [5.0] * 12
    )


@dataclass
class K1WalkEnvCfg(K1BaseCfg):
    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(
            model_file=str(ASSETS_ROOT_PATH / "robots" / "k1" / "scene_flat.xml")
        )
    )
    max_episode_seconds: float = 20.0
    init_state: InitState = field(default_factory=InitState)
    commands: Commands = field(
        default_factory=lambda: Commands(
            vel_limit=[
                [-0.35, -0.25, -0.8],
                [0.35, 0.25, 0.8],
            ],
            rel_standing_envs=0.1,
        )
    )
    reward_config: K1RewardConfig | None = None
    domain_rand: DomainRandConfig = field(default_factory=DomainRandConfig)
    gait_phase_init_mode: str = "offset_phase"
    reset_base_qvel_limit: float = 0.05
    obs_frame_stack: int = K1_OBS_FRAME_STACK


class K1WalkDomainRandomizationProvider(LocomotionDRProvider):
    def _get_qvel_limit(self, env: Any) -> float:
        return float(env.cfg.reset_base_qvel_limit)

    def _build_extra_info_updates(self, env: Any, num_reset: int) -> dict[str, np.ndarray]:
        updates: dict[str, np.ndarray] = {
            "gait_phase": self._sample_gait_phase(env, num_reset),
        }
        if getattr(env.cfg.commands, "heading_command", False):
            updates["heading_commands"] = sample_heading_commands(env, num_reset)
        return updates

    def _sample_commands(self, env: Any, num_reset: int) -> np.ndarray:
        commands = super()._sample_commands(env, num_reset)
        zero_small_xy_commands(commands, threshold=0.05)
        standing_prob = float(getattr(env.cfg.commands, "rel_standing_envs", 0.0))
        if standing_prob > 0.0:
            standing = np.random.uniform(size=(num_reset,)) < min(standing_prob, 1.0)
            commands[standing] = 0.0
        if getattr(env.cfg.commands, "heading_command", False):
            commands[:, 2] = 0.0
        return commands

    def _sample_gait_phase(self, env: Any, num_reset: int) -> np.ndarray:
        mode = env.cfg.gait_phase_init_mode
        if mode == "independent":
            left = np.random.uniform(0.0, 2.0 * np.pi, size=(num_reset,))
            right = np.random.uniform(0.0, 2.0 * np.pi, size=(num_reset,))
            return np.asarray(np.column_stack([left, right]), dtype=get_global_dtype())

        phase = np.random.uniform(0.0, 2.0 * np.pi, size=(num_reset,))
        return np.asarray(np.column_stack([phase, phase + np.pi]), dtype=get_global_dtype())

    def _compute_reset_obs(
        self,
        env: Any,
        env_ids: Any,
        info_updates: Any,
        linvel: Any,
        gyro: Any,
        gravity: Any,
        dof_pos: Any,
        dof_vel: Any,
    ) -> dict[str, np.ndarray]:
        return env._compute_obs(  # type: ignore[no-any-return]
            info_updates,
            linvel,
            gyro,
            gravity,
            dof_pos,
            dof_vel,
            env_ids=np.asarray(env_ids, dtype=np.intp),
            is_reset=True,
        )


class K1WalkEnv(K1BaseEnv):
    _cfg: K1WalkEnvCfg
    _reward_cfg: K1RewardConfig

    def __init__(self, cfg: K1WalkEnvCfg, num_envs=1, backend_type="mujoco"):
        if cfg.reward_config is None:
            raise ValueError("reward_config must be provided via Hydra configuration")
        backend = create_backend(
            backend_type,
            cfg.scene,
            num_envs,
            cfg.sim_dt,
            base_name=cfg.asset.base_name,
            push_body_name=cfg.domain_rand.push_body_name,
            motrix_max_iterations=cfg.motrix_max_iterations,
            post_step_forward_sensor=cfg.post_step_forward_sensor,
        )
        super().__init__(cfg, backend, num_envs)
        self._enable_reward_log = True
        self._reward_cfg = cfg.reward_config
        self._gait_phase_delta = float(
            2.0 * math.pi * self._reward_cfg.gait_frequency * cfg.ctrl_dt
        )
        self._pose_weights = np.asarray(self._reward_cfg.pose_weights, dtype=np.float32)
        if self._pose_weights.shape[0] != self._num_action:
            raise ValueError("pose_weights length mismatch")
        if self._num_action != K1_NUM_ACTION:
            raise ValueError(f"expected {K1_NUM_ACTION} actuators, got {self._num_action}")

        stack = cfg.obs_frame_stack
        self._obs_history = np.zeros((num_envs, stack, K1_OBS_SINGLE_DIM), dtype=np.float32)
        self._critic_obs_history = np.zeros((num_envs, stack, K1_OBS_SINGLE_DIM), dtype=np.float32)

        self._init_reward_functions()
        self._init_domain_randomization(K1WalkDomainRandomizationProvider())

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        return {"obs": K1_OBS_STACKED_DIM, "critic": K1_OBS_STACKED_DIM + 3}

    def _init_reward_functions(self) -> None:
        self._reward_fns: dict[str, Any] = {
            "tracking_lin_vel": rewards.tracking_lin_vel,
            "tracking_ang_vel": rewards.tracking_ang_vel,
            "forward_progress": rewards.forward_progress,
            "under_speed": rewards.under_speed,
            "lin_vel_z": rewards.lin_vel_z,
            "orientation": rewards.orientation,
            "ang_vel_xy": rewards.ang_vel_xy,
            "action_rate": rewards.action_rate,
            "base_height": rewards.base_height,
            "pose": rewards.weighted_pose,
            "feet_phase": self._reward_feet_phase,
            "feet_phase_contrast": self._reward_feet_phase_contrast,
            "feet_phase_contact": self._reward_feet_phase_contact,
            "feet_double_stance": self._reward_feet_double_stance,
            "penalty_close_feet_xy": self._reward_close_feet_xy,
            "alive": rewards.alive,
        }

    def _history_slice(self, env_ids: np.ndarray | None) -> slice | np.ndarray:
        return slice(None) if env_ids is None else env_ids

    def _update_obs_history(
        self,
        *,
        env_ids: np.ndarray | None,
        single_actor: np.ndarray,
        single_critic: np.ndarray,
        is_reset: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        sel = self._history_slice(env_ids)
        num_rows = single_actor.shape[0]
        if is_reset:
            self._obs_history[sel, :] = single_actor[:, None, :]
            self._critic_obs_history[sel, :] = single_critic[:, None, :]
        else:
            self._obs_history[sel, :-1] = self._obs_history[sel, 1:]
            self._obs_history[sel, -1] = single_actor
            self._critic_obs_history[sel, :-1] = self._critic_obs_history[sel, 1:]
            self._critic_obs_history[sel, -1] = single_critic
        actor = self._obs_history[sel].reshape(num_rows, K1_OBS_STACKED_DIM)
        critic_base = self._critic_obs_history[sel].reshape(num_rows, K1_OBS_STACKED_DIM)
        return actor, critic_base

    def _build_single_frame_obs(
        self,
        *,
        command: np.ndarray,
        gravity: np.ndarray,
        gyro: np.ndarray,
        joint_diff: np.ndarray,
        dof_vel: np.ndarray,
        last_actions: np.ndarray,
        noisy: bool,
    ) -> np.ndarray:
        noise_cfg = self._cfg.noise_config
        cmd_obs = commands_to_amp_obs(command)
        if noisy:
            gyro_obs = self._obs_noise(gyro, noise_cfg.scale_gyro) * 0.25
            gravity_obs = self._obs_noise(gravity, noise_cfg.scale_gravity)
            diff_obs = self._obs_noise(joint_diff, noise_cfg.scale_joint_angle)
            vel_obs = self._obs_noise(dof_vel, noise_cfg.scale_joint_vel) * 0.05
        else:
            gyro_obs = gyro * 0.25
            gravity_obs = gravity
            diff_obs = joint_diff
            vel_obs = dof_vel * 0.05

        return np.concatenate(
            (cmd_obs, gravity_obs, gyro_obs, diff_obs, vel_obs, last_actions),
            axis=1,
            dtype=np.float32,
        )

    def _compute_obs(
        self,
        info: dict,
        linvel,
        gyro,
        gravity,
        dof_pos,
        dof_vel,
        *,
        env_ids: np.ndarray | None = None,
        is_reset: bool = False,
    ) -> dict[str, np.ndarray]:
        num_rows = linvel.shape[0]
        diff = dof_pos - self.default_angles
        command = info["commands"]
        last_actions = info.get("current_actions", np.zeros((num_rows, self._num_action)))
        single_actor = self._build_single_frame_obs(
            command=command,
            gravity=-gravity,
            gyro=gyro,
            joint_diff=diff,
            dof_vel=dof_vel,
            last_actions=last_actions,
            noisy=True,
        )
        single_critic = self._build_single_frame_obs(
            command=command,
            gravity=-gravity,
            gyro=gyro,
            joint_diff=diff,
            dof_vel=dof_vel,
            last_actions=last_actions,
            noisy=False,
        )
        actor, critic_base = self._update_obs_history(
            env_ids=env_ids,
            single_actor=single_actor,
            single_critic=single_critic,
            is_reset=is_reset,
        )
        critic = np.concatenate(
            (critic_base, np.asarray(linvel * 2.0, dtype=np.float32)),
            axis=1,
            dtype=np.float32,
        )
        return {"obs": actor, "critic": critic}

    def update_state(self, state: NpEnvState) -> NpEnvState:
        linvel = self.get_local_linvel()
        gyro = self.get_gyro()
        gravity = self._backend.get_sensor_data(self._cfg.sensor.upvector)
        dof_pos = self.get_dof_pos()
        dof_vel = self.get_dof_vel()

        max_tilt_rad = np.deg2rad(self._reward_cfg.max_tilt_deg)
        tilt = np.arccos(np.clip(gravity[:, 2], -1, 1))
        terminated = np.logical_or(
            tilt > max_tilt_rad,
            self._backend.get_base_pos()[:, 2] < self._reward_cfg.min_base_height,
        )

        reward = self._compute_reward(state.info, linvel, gyro, gravity, dof_pos, dof_vel)
        obs = self._compute_obs(state.info, linvel, gyro, gravity, dof_pos, dof_vel)
        return state.replace(obs=obs, reward=reward, terminated=terminated)

    def _build_reward_context(
        self, info: dict, linvel, gyro, gravity, dof_pos, dof_vel
    ) -> RewardContext:
        return RewardContext(
            info=info,
            linvel=linvel,
            gyro=gyro,
            dof_pos=dof_pos,
            num_envs=self._num_envs,
            default_angles=self.default_angles,
            tracking_sigma=self._reward_cfg.tracking_sigma,
            base_height_target=self._reward_cfg.base_height_target,
            base_height=self._backend.get_base_pos()[:, 2],
            gravity=gravity,
            dof_vel=dof_vel,
            pose_weights=self._pose_weights,
        )

    def _compute_reward(self, info: dict, linvel, gyro, gravity, dof_pos, dof_vel) -> np.ndarray:
        ctx = self._build_reward_context(info, linvel, gyro, gravity, dof_pos, dof_vel)
        return rewards.run_reward_dispatch(
            scales=self._reward_cfg.scales,
            fns=self._reward_fns,
            ctx=ctx,
            info=info,
            enable_log=self._enable_reward_log,
            ctrl_dt=self._cfg.ctrl_dt,
        )

    def _gait_reward_gate(self, linvel: np.ndarray) -> np.ndarray:
        return compute_forward_speed_gate(linvel, self._reward_cfg.min_forward_speed_for_gait_reward)

    def _reward_feet_phase(self, ctx: RewardContext):
        left_foot = self._backend.get_sensor_data("left_foot_pos")
        right_foot = self._backend.get_sensor_data("right_foot_pos")
        gait_phase = ctx.info.get(
            "gait_phase", np.zeros((self._num_envs, 2), dtype=get_global_dtype())
        )
        swing_height = self._reward_cfg.feet_phase_swing_height
        left_target, right_target = compute_feet_phase_height_targets(gait_phase, swing_height)
        left_error = np.square(left_foot[:, 2] - left_target)
        right_error = np.square(right_foot[:, 2] - right_target)
        reward = np.exp(-(left_error + right_error) / self._reward_cfg.feet_phase_tracking_sigma)
        return np.asarray(reward * self._gait_reward_gate(ctx.linvel), dtype=get_global_dtype())

    def _reward_feet_phase_contrast(self, ctx: RewardContext):
        left_foot = self._backend.get_sensor_data("left_foot_pos")
        right_foot = self._backend.get_sensor_data("right_foot_pos")
        gait_phase = ctx.info.get(
            "gait_phase", np.zeros((self._num_envs, 2), dtype=get_global_dtype())
        )
        swing_height = self._reward_cfg.feet_phase_swing_height
        left_target, right_target = compute_feet_phase_height_targets(gait_phase, swing_height)
        actual_delta = left_foot[:, 2] - right_foot[:, 2]
        target_delta = left_target - right_target
        error = np.square(actual_delta - target_delta)
        reward = np.exp(-error / self._reward_cfg.feet_phase_tracking_sigma)
        return np.asarray(reward * self._gait_reward_gate(ctx.linvel), dtype=get_global_dtype())

    def _reward_feet_phase_contact(self, ctx: RewardContext):
        gait_phase = ctx.info.get(
            "gait_phase", np.zeros((self._num_envs, 2), dtype=get_global_dtype())
        )
        swing_height = self._reward_cfg.feet_phase_swing_height
        left_target_contact, right_target_contact = compute_feet_phase_contact_targets(
            gait_phase, swing_height
        )
        left_contact = compute_aggregated_foot_contact(self._backend, LEFT_FOOT_CONTACT_SENSORS)
        right_contact = compute_aggregated_foot_contact(self._backend, RIGHT_FOOT_CONTACT_SENSORS)
        left_match = np.asarray(left_contact == left_target_contact, dtype=get_global_dtype())
        right_match = np.asarray(right_contact == right_target_contact, dtype=get_global_dtype())
        reward = np.asarray(0.5 * (left_match + right_match), dtype=get_global_dtype())
        return np.asarray(reward * self._gait_reward_gate(ctx.linvel), dtype=get_global_dtype())

    def _reward_feet_double_stance(self, ctx: RewardContext):
        commands = ctx.info.get("commands", np.zeros((self._num_envs, 3), dtype=get_global_dtype()))
        left_contact = compute_aggregated_foot_contact(self._backend, LEFT_FOOT_CONTACT_SENSORS)
        right_contact = compute_aggregated_foot_contact(self._backend, RIGHT_FOOT_CONTACT_SENSORS)
        double_stance = np.asarray(
            np.logical_and(left_contact, right_contact), dtype=get_global_dtype()
        )
        return np.asarray(
            double_stance * compute_forward_command_mask(commands), dtype=get_global_dtype()
        )

    def _reward_close_feet_xy(self, ctx: RewardContext):
        left_foot = self._backend.get_sensor_data("left_foot_pos")
        right_foot = self._backend.get_sensor_data("right_foot_pos")
        feet_dist = np.linalg.norm(left_foot[:, :2] - right_foot[:, :2], axis=1)
        return np.where(
            feet_dist < self._reward_cfg.close_feet_threshold,
            np.square(feet_dist - self._reward_cfg.close_feet_threshold),
            0.0,
        )

    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> np.ndarray:
        state.info["last_actions"] = state.info.get("current_actions", np.zeros_like(actions))
        state.info["current_actions"] = actions

        gait_phase = state.info.get(
            "gait_phase", np.zeros((self._num_envs, 2), dtype=get_global_dtype())
        )
        gait_phase[:, 0] = (gait_phase[:, 0] + self._gait_phase_delta) % (2 * np.pi)
        gait_phase[:, 1] = (gait_phase[:, 1] + self._gait_phase_delta) % (2 * np.pi)
        state.info["gait_phase"] = gait_phase

        return actions * self._cfg.control_config.action_scale + self.default_angles


@registry.envcfg("K1WalkFlat")
@dataclass
class K1WalkFlatCfg(K1WalkEnvCfg):
    reward_config: K1RewardConfig | None = None


registry.register_env("K1WalkFlat", K1WalkEnv, sim_backend="mujoco")
registry.register_env("K1WalkFlat", K1WalkEnv, sim_backend="motrix")

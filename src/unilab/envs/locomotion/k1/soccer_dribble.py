"""K1 soccer dribble task on minimal robot+ball scene."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.scene import SceneCfg
from unilab.dtype_config import get_global_dtype
from unilab.envs.common.rotation import np_quat_apply_inverse
from unilab.envs.locomotion.common import rewards
from unilab.envs.locomotion.common.commands import Commands
from unilab.envs.locomotion.common.domain_rand import DomainRandConfig
from unilab.envs.locomotion.common.rewards import RewardContext
from unilab.dr import ResetPlan
from unilab.dr.dr_utils import zero_actions
from unilab.envs.locomotion.k1.base import NoiseConfig
from unilab.envs.locomotion.k1.joystick import (
    K1_NUM_ACTION,
    K1_OBS_SINGLE_DIM,
    K1RewardConfig,
    K1WalkDomainRandomizationProvider,
    K1WalkEnv,
    K1WalkEnvCfg,
    commands_to_amp_obs,
)
from unilab.envs.locomotion.k1.constants import K1_ACTUATOR_JOINT_ORDER

_LEFT_HIP_YAW_IDX = K1_ACTUATOR_JOINT_ORDER.index("Left_Hip_Yaw")
_RIGHT_HIP_YAW_IDX = K1_ACTUATOR_JOINT_ORDER.index("Right_Hip_Yaw")


class K1SoccerDomainRandomizationProvider(K1WalkDomainRandomizationProvider):
    """Deterministic reset: fixed keyframe qpos/qvel for robot and ball."""

    def _sample_commands(self, env: Any, num_reset: int) -> np.ndarray:
        # Filled in K1SoccerDribbleEnv.reset() after sim state is applied.
        return np.zeros((num_reset, 3), dtype=get_global_dtype())

    def _build_extra_info_updates(self, env: Any, num_reset: int) -> dict[str, np.ndarray]:
        phase = np.zeros((num_reset,), dtype=get_global_dtype())
        return {
            "gait_phase": np.asarray(
                np.column_stack([phase, phase + np.pi]), dtype=get_global_dtype()
            ),
        }

    def build_reset_plan(self, env: Any, env_ids: np.ndarray) -> ResetPlan:
        num_reset = len(env_ids)
        qpos = np.tile(env._init_qpos, (num_reset, 1))
        qvel = np.zeros_like(np.tile(env._init_qvel, (num_reset, 1)))
        yaw = np.zeros((num_reset,), dtype=get_global_dtype())
        qpos[:, 0:3] = env._spawn.apply_spawn(env_ids, qpos[:, 0:3], yaw=yaw)
        info_updates: dict[str, Any] = {
            "commands": self._sample_commands(env, num_reset),
            "current_actions": zero_actions(num_reset, env._num_action),
            "last_actions": zero_actions(num_reset, env._num_action),
        }
        info_updates.update(self._build_extra_info_updates(env, num_reset))
        env._spawn.record_episode_start(env_ids, qpos[:, 0:3])
        return ResetPlan(
            env_ids=env_ids,
            qpos=qpos,
            qvel=qvel,
            info_updates=info_updates,
            randomization=None,
        )


@dataclass
class K1SoccerDribbleRewardConfig(K1RewardConfig):
    ball_keep_distance: float = 0.45
    ball_keep_sigma: float = 0.08
    ball_front_lateral_sigma: float = 0.10
    ball_speed_sigma: float = 0.12
    ball_lost_distance: float = 1.2
    ball_lost_distance_hard: float = 2.0
    ball_speed_cap: float = 0.35
    ball_move_speed_target: float = 0.25
    ball_move_speed_sigma: float = 0.15
    ball_approach_speed_cap: float = 0.6
    ball_still_speed_threshold: float = 0.06
    ball_cmd_deadzone: float = 0.08
    ball_cmd_kp: float = 1.2
    fixed_cmd_lin_speed: float = 0.3
    feet_step_travel_cap: float = 0.10


@dataclass
class K1SoccerAsset:
    base_name = "Trunk"
    foot_name = "left_foot_link"
    ground = "COL_Collider"


@dataclass
class K1SoccerNoiseConfig(NoiseConfig):
    level: float = 0.0


@dataclass
class K1SoccerDribbleCfg(K1WalkEnvCfg):
    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(
            model_file=str(ASSETS_ROOT_PATH / "robots" / "k1" / "scene_soccer_dribble_minimal.xml")
        )
    )
    commands: Commands = field(
        default_factory=lambda: Commands(
            vel_limit=[
                [0.35, 0.0, 0.0],
                [0.75, 0.0, 0.0],
            ],
            rel_standing_envs=0.0,
        )
    )
    asset: K1SoccerAsset = field(default_factory=K1SoccerAsset)
    noise_config: K1SoccerNoiseConfig = field(default_factory=K1SoccerNoiseConfig)  # type: ignore[assignment]
    domain_rand: DomainRandConfig = field(default_factory=DomainRandConfig)
    reset_base_qvel_limit: float = 0.0
    reward_config: K1SoccerDribbleRewardConfig | None = None


class K1SoccerDribbleEnv(K1WalkEnv):
    _cfg: K1SoccerDribbleCfg
    _reward_cfg: K1SoccerDribbleRewardConfig

    def __init__(self, cfg: K1SoccerDribbleCfg, num_envs=1, backend_type="mujoco"):
        super().__init__(cfg, num_envs=num_envs, backend_type=backend_type)
        self._init_domain_randomization(K1SoccerDomainRandomizationProvider())
        self._reward_cfg = cfg.reward_config
        self._ball_body_ids = self._backend.get_body_ids(["ball"])
        if self._ball_body_ids.shape != (1,):
            raise ValueError("soccer dribble task expects exactly one body named 'ball'")
        self._obs_single_dim = K1_OBS_SINGLE_DIM + 3
        self._obs_stacked_dim = self._obs_single_dim * int(self._cfg.obs_frame_stack)
        self._obs_history = np.zeros(
            (self._num_envs, self._cfg.obs_frame_stack, self._obs_single_dim), dtype=np.float32
        )
        self._critic_obs_history = np.zeros(
            (self._num_envs, self._cfg.obs_frame_stack, self._obs_single_dim), dtype=np.float32
        )
        left_foot, right_foot = self._foot_positions()
        self._prev_foot_pos = np.stack([left_foot, right_foot], axis=1).astype(np.float32)

    def reset(self, env_indices: np.ndarray) -> tuple[dict[str, np.ndarray], dict]:
        obs, info = super().reset(env_indices)
        info["commands"] = self._command_toward_ball(env_indices)
        self._sync_prev_foot_pos(env_indices)
        return obs, info

    def _foot_positions(self) -> tuple[np.ndarray, np.ndarray]:
        left_foot = self._backend.get_sensor_data("left_foot_pos")
        right_foot = self._backend.get_sensor_data("right_foot_pos")
        return left_foot, right_foot

    def _sync_prev_foot_pos(self, env_ids: np.ndarray) -> None:
        left_foot, right_foot = self._foot_positions()
        foot_pos = np.stack([left_foot, right_foot], axis=1)
        sel = np.asarray(env_ids, dtype=np.intp)
        self._prev_foot_pos[sel] = foot_pos[sel]

    def _update_feet_step_travel(self, info: dict) -> None:
        left_foot, right_foot = self._foot_positions()
        foot_pos = np.stack([left_foot, right_foot], axis=1)
        step_delta = foot_pos - self._prev_foot_pos
        left_travel = np.linalg.norm(step_delta[:, 0, :], axis=1)
        right_travel = np.linalg.norm(step_delta[:, 1, :], axis=1)
        travel = left_travel + right_travel
        cap = float(self._reward_cfg.feet_step_travel_cap)
        info["feet_step_travel"] = np.asarray(np.clip(travel, 0.0, cap), dtype=get_global_dtype())
        self._prev_foot_pos[:] = foot_pos

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        return {"obs": self._obs_stacked_dim, "critic": self._obs_stacked_dim + 3}

    def _init_reward_functions(self) -> None:
        super()._init_reward_functions()
        self._reward_fns.update(
            {
                "ball_progress": self._reward_ball_progress,
                "ball_keep": self._reward_ball_keep,
                "ball_front": self._reward_ball_front,
                "ball_speed_match": self._reward_ball_speed_match,
                "ball_approach": self._reward_ball_approach,
                "ball_moving": self._reward_ball_moving,
                "ball_still": self._reward_ball_still,
                "ball_lost": self._reward_ball_lost,
                "ball_over_speed": self._reward_ball_over_speed,
                "ball_orbit": self._reward_ball_orbit,
                "feet_inward_yaw": self._reward_feet_inward_yaw,
                "feet_step_travel": self._reward_feet_step_travel,
                "termination_bad": self._reward_termination_bad,
            }
        )

    def _ball_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        base_pos = self._backend.get_base_pos()
        base_quat = self._backend.get_base_quat()
        base_lin_vel = self._backend.get_base_lin_vel()
        ball_pos_w = self._backend.get_body_pos_w(self._ball_body_ids)[:, 0, :]
        ball_vel_w = self._backend.get_body_lin_vel_w(self._ball_body_ids)[:, 0, :]
        rel_pos_w = ball_pos_w - base_pos
        rel_pos_b = np_quat_apply_inverse(base_quat, rel_pos_w)
        rel_vel_b = np_quat_apply_inverse(base_quat, ball_vel_w - base_lin_vel)
        return ball_pos_w, ball_vel_w, rel_pos_b, rel_vel_b

    def _build_single_frame_obs(
        self,
        *,
        command: np.ndarray,
        ball_obs: np.ndarray,
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
            (cmd_obs, ball_obs, gravity_obs, gyro_obs, diff_obs, vel_obs, last_actions),
            axis=1,
            dtype=np.float32,
        )

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
        actor = self._obs_history[sel].reshape(num_rows, self._obs_stacked_dim)
        critic_base = self._critic_obs_history[sel].reshape(num_rows, self._obs_stacked_dim)
        return actor, critic_base

    def _ball_obs(self, env_ids: np.ndarray | None = None) -> np.ndarray:
        _, _, rel_pos_b, rel_vel_b = self._ball_state()
        ball_obs_all = np.stack(
            [rel_pos_b[:, 0], rel_pos_b[:, 1], rel_vel_b[:, 0]],
            axis=1,
            dtype=np.float32,
        )
        if env_ids is None:
            return ball_obs_all
        return ball_obs_all[np.asarray(env_ids, dtype=np.intp)]

    def _command_toward_ball(self, env_ids: np.ndarray | None = None) -> np.ndarray:
        """Body-frame cmd: fixed linear speed, direction toward the ball."""
        _, _, rel_pos_b, _ = self._ball_state()
        if env_ids is not None:
            rel_pos_b = rel_pos_b[np.asarray(env_ids, dtype=np.intp)]
        dist_xy = np.linalg.norm(rel_pos_b[:, :2], axis=1)
        dir_xy = rel_pos_b[:, :2] / np.maximum(dist_xy[:, None], 1e-6)
        default_forward = np.asarray([1.0, 0.0], dtype=get_global_dtype())
        dir_xy = np.where(dist_xy[:, None] < 1e-3, default_forward, dir_xy)

        speed = float(self._reward_cfg.fixed_cmd_lin_speed)
        cmd_xy = dir_xy * speed

        vel_limits = np.asarray(self._cfg.commands.vel_limit, dtype=np.float32)
        low = vel_limits[0]
        high = vel_limits[1]
        cmd_x = np.clip(cmd_xy[:, 0], low[0], high[0])
        cmd_y = np.clip(cmd_xy[:, 1], low[1], high[1])
        cmd_yaw = np.zeros_like(cmd_x)
        return np.asarray(np.stack((cmd_x, cmd_y, cmd_yaw), axis=1), dtype=get_global_dtype())

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
        command = self._command_toward_ball(env_ids)
        last_actions = info.get("current_actions", np.zeros((num_rows, self._num_action)))
        ball_obs = self._ball_obs(env_ids)

        single_actor = self._build_single_frame_obs(
            command=command,
            ball_obs=ball_obs,
            gravity=-gravity,
            gyro=gyro,
            joint_diff=diff,
            dof_vel=dof_vel,
            last_actions=last_actions,
            noisy=True,
        )
        single_critic = self._build_single_frame_obs(
            command=command,
            ball_obs=ball_obs,
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

    def update_state(self, state):
        linvel = self.get_local_linvel()
        gyro = self.get_gyro()
        gravity = self._backend.get_sensor_data(self._cfg.sensor.upvector)
        dof_pos = self.get_dof_pos()
        dof_vel = self.get_dof_vel()

        _, _, rel_pos_b, _ = self._ball_state()
        ball_dist_xy = np.linalg.norm(rel_pos_b[:, :2], axis=1)
        state.info["commands"] = self._command_toward_ball()

        max_tilt_rad = np.deg2rad(self._reward_cfg.max_tilt_deg)
        tilt = np.arccos(np.clip(gravity[:, 2], -1, 1))
        term_fall = tilt > max_tilt_rad
        term_low = self._backend.get_base_pos()[:, 2] < self._reward_cfg.min_base_height
        term_ball_lost = ball_dist_xy > float(self._reward_cfg.ball_lost_distance_hard)
        term_bad = np.logical_or(term_fall, term_low)
        terminated = np.logical_or(term_bad, term_ball_lost)

        state.info["ball_dist_xy"] = np.asarray(ball_dist_xy, dtype=get_global_dtype())
        state.info["term_bad"] = np.asarray(term_bad, dtype=get_global_dtype())
        self._update_feet_step_travel(state.info)
        reward = self._compute_reward(state.info, linvel, gyro, gravity, dof_pos, dof_vel)
        obs = self._compute_obs(state.info, linvel, gyro, gravity, dof_pos, dof_vel)
        return state.replace(obs=obs, reward=reward, terminated=terminated)

    def _reward_ball_progress(self, ctx: RewardContext):
        _, ball_vel_w, _, _ = self._ball_state()
        return np.asarray(np.maximum(ball_vel_w[:, 0], 0.0), dtype=get_global_dtype())

    def _reward_ball_keep(self, ctx: RewardContext):
        _, _, rel_pos_b, _ = self._ball_state()
        dist_xy = np.linalg.norm(rel_pos_b[:, :2], axis=1)
        err = dist_xy - float(self._reward_cfg.ball_keep_distance)
        sigma = max(float(self._reward_cfg.ball_keep_sigma), 1e-6)
        return np.asarray(np.exp(-(err * err) / sigma), dtype=get_global_dtype())

    def _reward_ball_front(self, ctx: RewardContext):
        _, _, rel_pos_b, _ = self._ball_state()
        dist_xy = np.linalg.norm(rel_pos_b[:, :2], axis=1)
        front = np.clip(rel_pos_b[:, 0] / np.maximum(dist_xy, 1e-6), 0.0, 1.0)
        sigma = max(float(self._reward_cfg.ball_front_lateral_sigma), 1e-6)
        lateral = np.exp(-(rel_pos_b[:, 1] * rel_pos_b[:, 1]) / sigma)
        return np.asarray(front * lateral, dtype=get_global_dtype())

    def _reward_ball_speed_match(self, ctx: RewardContext):
        _, ball_vel_w, _, _ = self._ball_state()
        target_vx = ctx.info["commands"][:, 0]
        err = ball_vel_w[:, 0] - target_vx
        sigma = max(float(self._reward_cfg.ball_speed_sigma), 1e-6)
        return np.asarray(np.exp(-(err * err) / sigma), dtype=get_global_dtype())

    def _reward_ball_approach(self, ctx: RewardContext):
        _, _, rel_pos_b, rel_vel_b = self._ball_state()
        dist_xy = np.linalg.norm(rel_pos_b[:, :2], axis=1)
        dist_gap = np.maximum(dist_xy - float(self._reward_cfg.ball_keep_distance), 0.0)
        radial_closing_speed = -np.sum(rel_pos_b[:, :2] * rel_vel_b[:, :2], axis=1) / np.maximum(
            dist_xy, 1e-6
        )
        approach_cap = float(self._reward_cfg.ball_approach_speed_cap)
        approach_speed = np.clip(radial_closing_speed, 0.0, approach_cap)
        active = np.asarray(dist_gap > 0.0, dtype=get_global_dtype())
        return np.asarray(approach_speed * active, dtype=get_global_dtype())

    def _reward_ball_lost(self, ctx: RewardContext):
        _, _, rel_pos_b, _ = self._ball_state()
        dist_xy = np.linalg.norm(rel_pos_b[:, :2], axis=1)
        threshold = float(self._reward_cfg.ball_lost_distance)
        return np.asarray(np.maximum(dist_xy - threshold, 0.0), dtype=get_global_dtype())

    def _reward_ball_moving(self, ctx: RewardContext):
        _, ball_vel_w, _, _ = self._ball_state()
        ball_speed = np.linalg.norm(ball_vel_w[:, :2], axis=1)
        speed_target = float(self._reward_cfg.ball_move_speed_target)
        sigma = max(float(self._reward_cfg.ball_move_speed_sigma), 1e-6)
        speed_shortfall = np.maximum(speed_target - ball_speed, 0.0)
        return np.asarray(np.exp(-(speed_shortfall * speed_shortfall) / sigma), dtype=get_global_dtype())

    def _reward_ball_still(self, ctx: RewardContext):
        _, ball_vel_w, _, _ = self._ball_state()
        ball_speed = np.linalg.norm(ball_vel_w[:, :2], axis=1)
        threshold = float(self._reward_cfg.ball_still_speed_threshold)
        still_penalty = np.maximum(threshold - ball_speed, 0.0)
        return np.asarray(still_penalty, dtype=get_global_dtype())

    def _reward_ball_over_speed(self, ctx: RewardContext):
        _, ball_vel_w, _, _ = self._ball_state()
        ball_speed = np.linalg.norm(ball_vel_w[:, :2], axis=1)
        cap = float(self._reward_cfg.ball_speed_cap)
        return np.asarray(np.maximum(ball_speed - cap, 0.0), dtype=get_global_dtype())

    def _reward_ball_orbit(self, ctx: RewardContext):
        _, _, rel_pos_b, rel_vel_b = self._ball_state()
        dist_xy = np.linalg.norm(rel_pos_b[:, :2], axis=1)
        tangential_speed = np.abs(
            rel_pos_b[:, 0] * rel_vel_b[:, 1] - rel_pos_b[:, 1] * rel_vel_b[:, 0]
        ) / np.maximum(dist_xy, 1e-6)
        return np.asarray(tangential_speed, dtype=get_global_dtype())

    def _reward_feet_inward_yaw(self, ctx: RewardContext):
        dof_diff = ctx.dof_pos - ctx.default_angles
        left_yaw = dof_diff[:, _LEFT_HIP_YAW_IDX]
        right_yaw = dof_diff[:, _RIGHT_HIP_YAW_IDX]
        # Penalize only the confirmed "feet toward each other" directions:
        # left inward: left hip yaw < 0; right inward: right hip yaw > 0.
        left_inward = np.maximum(-left_yaw, 0.0)
        right_inward = np.maximum(right_yaw, 0.0)
        inward_penalty = np.square(left_inward) + np.square(right_inward)
        return np.asarray(inward_penalty, dtype=get_global_dtype())

    def _reward_feet_step_travel(self, ctx: RewardContext) -> np.ndarray:
        """Per-step world-frame foot travel; larger displacement yields higher reward."""
        return np.asarray(ctx.info["feet_step_travel"], dtype=get_global_dtype())

    def _reward_termination_bad(self, ctx: RewardContext) -> np.ndarray:
        """One-shot penalty on fall / too-low termination (not ball lost)."""
        return np.asarray(ctx.info["term_bad"], dtype=get_global_dtype())


@registry.envcfg("K1SoccerDribble")
@dataclass
class K1SoccerDribbleFlatCfg(K1SoccerDribbleCfg):
    reward_config: K1SoccerDribbleRewardConfig | None = None


registry.register_env("K1SoccerDribble", K1SoccerDribbleEnv, sim_backend="mujoco")
registry.register_env("K1SoccerDribble", K1SoccerDribbleEnv, sim_backend="motrix")


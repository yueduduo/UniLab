"""K1 penalty-kick task: run-up + shoot into sim_soccer-sized goal (no goalkeeper)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.scene import SceneCfg
from unilab.dtype_config import get_global_dtype
from unilab.envs.common.rotation import np_quat_apply_inverse
from unilab.envs.locomotion.common.rewards import RewardContext
from unilab.envs.locomotion.common import rewards as locomotion_rewards
from unilab.envs.locomotion.k1.base import NoiseConfig
from unilab.envs.locomotion.k1.constants import K1_NUM_ACTION, k1_keyframe_robot_joint_qpos
from unilab.envs.locomotion.k1.joystick import (
    K1DomainRandConfig,
    K1RewardConfig,
    K1WalkDomainRandomizationProvider,
    K1WalkEnv,
    K1WalkEnvCfg,
)
from unilab.envs.locomotion.k1.soccer_penalty_constants import (
    K1_PENALTY_KICK_PHASE_MAX_STEPS,
    K1_PENALTY_GOAL_HEIGHT_M,
    K1_PENALTY_GOAL_LINE_X,
    K1_PENALTY_GOAL_WIDTH_M,
    K1_PENALTY_KICK_REWARD_KEYS,
    K1_PENALTY_MARKINGS_FIELD_WIDTH_M,
    K1_PENALTY_MAX_STEPS,
    K1_PENALTY_RESET_Y_OFFSET_M,
    K1_PENALTY_RIGHT_FOOT_STRIKE_DIST_M,
    K1_PENALTY_RUNUP_BALL_DIST_M,
    K1_PENALTY_RUNUP_MAX_STEPS,
    K1_PENALTY_SPOT_XY,
    k1_soccer_penalty_actor_obs_dim,
    k1_soccer_penalty_critic_obs_dim,
)
from unilab.envs.locomotion.common.commands import Commands
from unilab.dr import DomainRandomizationManager


class K1PenaltyKickDomainRandomizationProvider(K1WalkDomainRandomizationProvider):
    """Fixed penalty run-up line (+X); small lateral y jitter only."""

    def _sample_reset_xy_offset(self, env: Any, num_reset: int) -> np.ndarray:
        offset = np.zeros((num_reset, 2), dtype=np.float64)
        if env._runup_start_distance_curriculum_enabled():
            distance = env._sample_runup_start_distance(num_reset)
            target_x = float(K1_PENALTY_SPOT_XY[0]) - distance
            offset[:, 0] = target_x - float(env._init_qpos[0])
        limit = float(K1_PENALTY_RESET_Y_OFFSET_M)
        offset[:, 1] = np.random.uniform(-limit, limit, (num_reset,))
        return offset

    def _sample_reset_yaw(self, env: Any, num_reset: int) -> np.ndarray:
        return np.zeros(num_reset, dtype=get_global_dtype())

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
            env_ids=env_ids,
        )


@dataclass
class K1SoccerPenaltyKickRewardConfig(K1RewardConfig):
    runup_reach_sigma: float = 0.25
    runup_reach_sigma_2: float = 2.0
    right_foot_ball_sigma: float = 0.30
    right_foot_ball_sigma_2: float = 2.0
    right_foot_strike_dist: float = K1_PENALTY_RIGHT_FOOT_STRIKE_DIST_M
    runup_min_forward_speed: float = 0.35
    runup_speed_curriculum_initial: float = 0.0
    runup_speed_curriculum_increment: float = 0.1
    runup_speed_curriculum_interval_iterations: int = 1000
    runup_speed_curriculum_max_speed: float = 0.95
    runup_speed_success_curriculum_enabled: bool = False
    runup_speed_success_threshold: float = 0.05
    runup_speed_success_promotion_updates: int = 5
    runup_success_rate_ema_alpha: float = 0.2
    runup_start_distance_curriculum_enabled: bool = False
    runup_start_distance_phase1: list[float] = field(default_factory=lambda: [0.4, 0.7])
    runup_start_distance_phase2: list[float] = field(default_factory=lambda: [0.8, 1.2])
    runup_start_distance_phase3: list[float] = field(default_factory=lambda: [2.0, 2.0])
    runup_start_distance_success_threshold: float = 0.05
    runup_start_distance_promotion_updates: int = 5
    right_foot_ball_progress_clip: float = 1.0
    ball_to_goal_sigma: float = 0.35
    ball_kick_speed_target: float = 2.0
    ball_kick_speed_sigma: float = 1.0
    fixed_cmd_lin_speed: float = 0.55
    runup_ball_dist: float = K1_PENALTY_RUNUP_BALL_DIST_M
    runup_max_steps: int = K1_PENALTY_RUNUP_MAX_STEPS
    kick_phase_max_steps: int = K1_PENALTY_KICK_PHASE_MAX_STEPS
    max_steps: int = K1_PENALTY_MAX_STEPS
    runup_timeout_enable_after_iteration: int = 0
    vel_direction_min_speed: float = 0.05
    ball_fail_max_height: float = K1_PENALTY_GOAL_HEIGHT_M + 0.35
    ball_fail_backward_dist: float = 0.45


@dataclass
class K1PenaltyKickAsset:
    base_name = "Trunk"
    foot_name = "left_foot_link"
    ground = "COL_Collider"


@dataclass
class K1PenaltyKickNoiseConfig(NoiseConfig):
    level: float = 0.0


@dataclass
class K1SoccerPenaltyKickCfg(K1WalkEnvCfg):
    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(
            model_file=str(ASSETS_ROOT_PATH / "robots" / "k1" / "scene_soccer_penalty_kick.xml")
        )
    )
    max_episode_seconds: float = 12.0
    commands: Commands = field(
        default_factory=lambda: Commands(
            vel_limit=[
                [0.65, 0.0, 0.0],
                [0.95, 0.0, 0.0],
            ],
            rel_standing_envs=0.0,
        )
    )
    asset: K1PenaltyKickAsset = field(default_factory=K1PenaltyKickAsset)
    noise_config: K1PenaltyKickNoiseConfig = field(default_factory=K1PenaltyKickNoiseConfig)  # type: ignore[assignment]
    domain_rand: K1DomainRandConfig = field(
        default_factory=lambda: K1DomainRandConfig(
            randomize_kp=False,
            randomize_kd=False,
            randomize_base_mass=False,
            randomize_body_mass=False,
            random_com=False,
            randomize_gravity=False,
            randomize_ground_friction=False,
            randomize_dof_armature=False,
            push_robots=False,
        )
    )
    reset_base_qvel_limit: float = 0.5
    sim_dt: float = 0.002
    obs_frame_stack: int = 1
    add_body_sensors: bool = True
    reward_config: K1SoccerPenaltyKickRewardConfig | None = None


class K1SoccerPenaltyKickEnv(K1WalkEnv):
    _cfg: K1SoccerPenaltyKickCfg
    _reward_cfg: K1SoccerPenaltyKickRewardConfig

    def __init__(self, cfg: K1SoccerPenaltyKickCfg, num_envs=1, backend_type="mujoco"):
        super().__init__(cfg, num_envs=num_envs, backend_type=backend_type)
        self.default_angles = np.asarray(
            k1_keyframe_robot_joint_qpos(self._init_qpos, self._num_action),
            dtype=self.default_angles.dtype,
        )
        self._dr_manager = DomainRandomizationManager(
            self, K1PenaltyKickDomainRandomizationProvider()
        )
        self._reward_cfg = cfg.reward_config
        self._ball_body_ids = self._backend.get_body_ids(["ball"])
        if self._ball_body_ids.shape != (1,):
            raise ValueError("penalty kick task expects exactly one body named 'ball'")
        if int(self._cfg.obs_frame_stack) != 1:
            raise ValueError("K1SoccerPenaltyKick requires obs_frame_stack=1")
        self._actor_obs_dim = k1_soccer_penalty_actor_obs_dim(self._num_action)
        self._critic_obs_dim = k1_soccer_penalty_critic_obs_dim(self._num_action)
        self._runup_reached = np.zeros(self._num_envs, dtype=bool)
        self._runup_reach_reward_frozen = np.zeros(self._num_envs, dtype=np.float32)
        self._right_foot_ball_reward_frozen = np.zeros(self._num_envs, dtype=np.float32)
        self._runup_success_pending = np.zeros(self._num_envs, dtype=bool)
        self._goal_scored = np.zeros(self._num_envs, dtype=bool)
        self._kick_failed = np.zeros(self._num_envs, dtype=bool)
        self._kick_start_steps = np.full(self._num_envs, -1, dtype=np.int32)
        self._kick_target_dir_xy = np.asarray([1.0, 0.0], dtype=np.float32)
        self._training_iteration = 0
        self._runup_success_rate_ema = 0.0
        self._runup_success_rate_initialized = False
        self._runup_start_distance_phase_idx = 0
        self._runup_start_distance_success_streak = 0
        self._runup_speed_success_level_idx = 0
        self._runup_speed_success_streak = 0
        self._prev_right_foot_ball_dist = self._right_foot_ball_distance_xy().astype(
            np.float32, copy=True
        )
        self._apply_runup_speed_curriculum(self._runup_speed_curriculum_level())

    def set_training_iteration(self, iteration: int) -> None:
        self._training_iteration = int(iteration)
        self._apply_runup_speed_curriculum(self._runup_speed_curriculum_level())

    def _runup_speed_curriculum_level(self) -> float:
        initial = float(getattr(self._reward_cfg, "runup_speed_curriculum_initial", 0.0))
        if initial <= 0.0:
            return float(self._reward_cfg.runup_min_forward_speed)
        increment = float(getattr(self._reward_cfg, "runup_speed_curriculum_increment", 0.1))
        max_speed = float(getattr(self._reward_cfg, "runup_speed_curriculum_max_speed", initial))
        if bool(getattr(self._reward_cfg, "runup_speed_success_curriculum_enabled", False)):
            level_idx = max(int(getattr(self, "_runup_speed_success_level_idx", 0)), 0)
            return float(min(initial + increment * level_idx, max_speed))
        interval = max(int(getattr(self._reward_cfg, "runup_speed_curriculum_interval_iterations", 1000)), 1)
        iter_index = max(1, self._training_iteration)
        level_idx = (iter_index - 1) // interval
        return float(min(initial + increment * level_idx, max_speed))

    def _runup_speed_curriculum_max_level_idx(self) -> int:
        initial = float(getattr(self._reward_cfg, "runup_speed_curriculum_initial", 0.0))
        increment = float(getattr(self._reward_cfg, "runup_speed_curriculum_increment", 0.1))
        max_speed = float(getattr(self._reward_cfg, "runup_speed_curriculum_max_speed", initial))
        if initial <= 0.0 or increment <= 0.0 or max_speed <= initial:
            return 0
        return max(int(math.ceil((max_speed - initial) / increment)), 0)

    def _runup_speed_curriculum_max(self) -> float:
        return max(
            float(getattr(self._reward_cfg, "runup_speed_curriculum_max_speed", 0.95)),
            self._runup_speed_curriculum_level(),
        )

    def _apply_runup_speed_curriculum(self, min_speed: float) -> None:
        min_speed = max(float(min_speed), 1e-6)
        max_speed = max(self._runup_speed_curriculum_max(), min_speed)
        self._reward_cfg.runup_min_forward_speed = min_speed
        cfg = getattr(self, "_cfg", None)
        if cfg is not None and getattr(cfg, "commands", None) is not None:
            cfg.commands.vel_limit = [
                [min_speed, 0.0, 0.0],
                [max_speed, 0.0, 0.0],
            ]

    def _runup_timeout_enabled(self) -> bool:
        enable_after = int(
            getattr(self._reward_cfg, "runup_timeout_enable_after_iteration", 0)
        )
        if enable_after <= 0:
            return True
        return self._training_iteration >= enable_after

    def _runup_start_distance_curriculum_enabled(self) -> bool:
        return bool(getattr(self._reward_cfg, "runup_start_distance_curriculum_enabled", False))

    def _runup_start_distance_range(self) -> tuple[float, float]:
        phase_ranges = (
            getattr(self._reward_cfg, "runup_start_distance_phase1", [0.4, 0.7]),
            getattr(self._reward_cfg, "runup_start_distance_phase2", [0.8, 1.2]),
            getattr(self._reward_cfg, "runup_start_distance_phase3", [2.0, 2.0]),
        )
        idx = min(max(int(getattr(self, "_runup_start_distance_phase_idx", 0)), 0), len(phase_ranges) - 1)
        selected = list(phase_ranges[idx])
        if len(selected) != 2:
            raise ValueError("runup start distance curriculum phases must be [min, max] pairs")
        low, high = sorted((float(selected[0]), float(selected[1])))
        return low, high

    def _sample_runup_start_distance(self, num_reset: int) -> np.ndarray:
        low, high = self._runup_start_distance_range()
        if high <= low:
            return np.full((num_reset,), low, dtype=np.float64)
        return np.random.uniform(low, high, (num_reset,)).astype(np.float64, copy=False)

    def _update_success_curricula(self, done_indices: np.ndarray) -> None:
        if len(done_indices) == 0:
            return
        batch_rate = float(np.mean(self._runup_reached[done_indices]))
        alpha = float(getattr(self._reward_cfg, "runup_success_rate_ema_alpha", 0.2))
        alpha = float(np.clip(alpha, 0.0, 1.0))
        if not self._runup_success_rate_initialized:
            self._runup_success_rate_ema = batch_rate
            self._runup_success_rate_initialized = True
        else:
            self._runup_success_rate_ema = (
                (1.0 - alpha) * self._runup_success_rate_ema + alpha * batch_rate
            )

        if self._runup_start_distance_curriculum_enabled():
            threshold = float(getattr(self._reward_cfg, "runup_start_distance_success_threshold", 0.05))
            needed = max(int(getattr(self._reward_cfg, "runup_start_distance_promotion_updates", 5)), 1)
            if self._runup_success_rate_ema >= threshold:
                self._runup_start_distance_success_streak += 1
                if self._runup_start_distance_success_streak >= needed:
                    self._runup_start_distance_phase_idx = min(
                        self._runup_start_distance_phase_idx + 1, 2
                    )
                    self._runup_start_distance_success_streak = 0
            else:
                self._runup_start_distance_success_streak = 0

        if bool(getattr(self._reward_cfg, "runup_speed_success_curriculum_enabled", False)):
            threshold = float(getattr(self._reward_cfg, "runup_speed_success_threshold", 0.05))
            needed = max(int(getattr(self._reward_cfg, "runup_speed_success_promotion_updates", 5)), 1)
            if self._runup_success_rate_ema >= threshold:
                self._runup_speed_success_streak += 1
                if self._runup_speed_success_streak >= needed:
                    self._runup_speed_success_level_idx = min(
                        self._runup_speed_success_level_idx + 1,
                        self._runup_speed_curriculum_max_level_idx(),
                    )
                    self._runup_speed_success_streak = 0
                    self._apply_runup_speed_curriculum(self._runup_speed_curriculum_level())
            else:
                self._runup_speed_success_streak = 0

    def get_dof_pos(self) -> np.ndarray:
        dof_pos = super().get_dof_pos()
        if dof_pos.shape[1] > self._num_action:
            return dof_pos[:, : self._num_action]
        return dof_pos

    def get_dof_vel(self) -> np.ndarray:
        dof_vel = super().get_dof_vel()
        if dof_vel.shape[1] > self._num_action:
            return dof_vel[:, : self._num_action]
        return dof_vel

    def reset(self, env_indices: np.ndarray) -> tuple[dict[str, np.ndarray], dict]:
        obs, info = super().reset(env_indices)
        sel = np.asarray(env_indices, dtype=np.intp)
        self._runup_reached[sel] = False
        self._runup_reach_reward_frozen[sel] = 0.0
        self._right_foot_ball_reward_frozen[sel] = 0.0
        self._runup_success_pending[sel] = False
        self._goal_scored[sel] = False
        self._kick_failed[sel] = False
        self._kick_start_steps[sel] = -1
        self._prev_right_foot_ball_dist[sel] = self._right_foot_ball_distance_xy()[sel].astype(
            np.float32, copy=False
        )
        return obs, info

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        return {"obs": self._actor_obs_dim, "critic": self._critic_obs_dim}

    def _init_reward_functions(self) -> None:
        super()._init_reward_functions()
        self._reward_fns.update(
            {
                "runup_reach": self._reward_runup_reach,
                "runup_speed": self._reward_runup_speed,
                "right_foot_to_ball": self._reward_right_foot_to_ball,
                "right_foot_to_ball_progress": self._reward_right_foot_to_ball_progress,
                "runup_success": self._reward_runup_success,
                "ball_to_goal": self._reward_ball_to_goal,
                "ball_goal_progress": self._reward_ball_goal_progress,
                "ball_kick_speed": self._reward_ball_kick_speed,
                "kick_stability": self._reward_kick_stability,
                "goal_scored": self._reward_goal_scored,
                "kick_failed": self._reward_kick_failed,
                "termination_bad": self._reward_termination_bad,
                "term_fall": self._reward_term_fall,
                "term_low": self._reward_term_low,
                "term_runup_timeout": self._reward_term_runup_timeout,
                "term_kick_timeout": self._reward_term_kick_timeout,
                "term_ball_missed_goal": self._reward_term_ball_missed_goal,
                "term_ball_flew_high": self._reward_term_ball_flew_high,
                "term_ball_backward": self._reward_term_ball_backward,
                "term_ball_out_y": self._reward_term_ball_out_y,
                "penalty_vel_direction": self._reward_penalty_vel_direction,
                "tracking_lin_vel": self._reward_tracking_lin_vel,
                "under_speed": self._reward_under_speed,
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

    def _robot_ball_distance_xy(self) -> np.ndarray:
        base_pos = self._backend.get_base_pos()
        ball_pos_w, _, _, _ = self._ball_state()
        delta = ball_pos_w[:, :2] - base_pos[:, :2]
        return np.linalg.norm(delta, axis=1)

    def _right_foot_ball_distance_xy(self) -> np.ndarray:
        right_foot = self._backend.get_sensor_data("right_foot_pos")
        ball_pos_w, _, _, _ = self._ball_state()
        delta = ball_pos_w[:, :2] - right_foot[:, :2]
        return np.linalg.norm(delta, axis=1)

    def _ball_in_goal(self, ball_pos_w: np.ndarray) -> np.ndarray:
        half_goal_w = 0.5 * float(K1_PENALTY_GOAL_WIDTH_M)
        return (
            (ball_pos_w[:, 0] >= float(K1_PENALTY_GOAL_LINE_X))
            & (np.abs(ball_pos_w[:, 1]) <= half_goal_w)
            & (ball_pos_w[:, 2] <= float(K1_PENALTY_GOAL_HEIGHT_M))
        )

    def _ball_missed_goal(self, ball_pos_w: np.ndarray) -> np.ndarray:
        crossed_line = ball_pos_w[:, 0] >= float(K1_PENALTY_GOAL_LINE_X)
        return crossed_line & ~self._ball_in_goal(ball_pos_w)

    def _ball_flew_high(self, ball_pos_w: np.ndarray) -> np.ndarray:
        return ball_pos_w[:, 2] > float(self._reward_cfg.ball_fail_max_height)

    def _ball_went_backward(self, ball_pos_w: np.ndarray) -> np.ndarray:
        min_x = float(K1_PENALTY_SPOT_XY[0]) - float(self._reward_cfg.ball_fail_backward_dist)
        return ball_pos_w[:, 0] < min_x

    def _ball_out_of_play_y(self, ball_pos_w: np.ndarray) -> np.ndarray:
        half_wid = 0.5 * float(K1_PENALTY_MARKINGS_FIELD_WIDTH_M)
        return np.abs(ball_pos_w[:, 1]) > half_wid

    def _update_runup_reached(self) -> np.ndarray:
        dist = self._right_foot_ball_distance_xy()
        inside = dist <= float(self._reward_cfg.right_foot_strike_dist)
        newly_reached = inside & ~self._runup_reached
        self._runup_reached |= inside
        return newly_reached

    def _runup_reach_raw_from_distance(self, dist: np.ndarray) -> np.ndarray:
        sigma1 = max(float(self._reward_cfg.runup_reach_sigma), 1e-6)
        sigma2 = max(float(self._reward_cfg.runup_reach_sigma_2), 1e-6)
        near = 1.0 - np.tanh(dist / sigma1)
        far = 1.0 - np.tanh(dist / sigma2)
        return np.asarray(near + far, dtype=get_global_dtype())

    def _freeze_runup_reach_reward(self, newly_reached: np.ndarray) -> None:
        if not np.any(newly_reached):
            return
        dist = self._robot_ball_distance_xy()
        reach = self._runup_reach_raw_from_distance(dist)
        self._runup_reach_reward_frozen[newly_reached] = reach[newly_reached].astype(
            np.float32, copy=False
        )

    def _right_foot_ball_raw_from_distance(self, dist: np.ndarray) -> np.ndarray:
        sigma1 = max(float(self._reward_cfg.right_foot_ball_sigma), 1e-6)
        sigma2 = max(float(self._reward_cfg.right_foot_ball_sigma_2), 1e-6)
        near = 1.0 - np.tanh(dist / sigma1)
        far = 1.0 - np.tanh(dist / sigma2)
        return np.asarray(near + far, dtype=get_global_dtype())

    def _freeze_right_foot_ball_reward(self, newly_reached: np.ndarray) -> None:
        if not np.any(newly_reached):
            return
        dist = self._right_foot_ball_distance_xy()
        reward = self._right_foot_ball_raw_from_distance(dist)
        self._right_foot_ball_reward_frozen[newly_reached] = reward[newly_reached].astype(
            np.float32, copy=False
        )

    def _command_for_phase(self, info: dict, env_ids: np.ndarray | None = None) -> np.ndarray:
        cmd = np.array(info["commands"], copy=True)
        if env_ids is None:
            kick_phase = self._runup_reached
        else:
            kick_phase = self._runup_reached[np.asarray(env_ids, dtype=np.intp)]
        if not np.any(kick_phase):
            return cmd
        cmd_kick = self._command_toward_goal(env_ids)
        cmd[kick_phase] = cmd_kick[kick_phase]
        return cmd

    def _command_toward_goal(self, env_ids: np.ndarray | None = None) -> np.ndarray:
        num_rows = self._num_envs if env_ids is None else len(env_ids)
        speed = float(self._reward_cfg.fixed_cmd_lin_speed)
        cmd_x = np.full((num_rows,), speed, dtype=get_global_dtype())
        cmd_y = np.zeros_like(cmd_x)
        cmd_yaw = np.zeros_like(cmd_x)
        return np.asarray(np.stack((cmd_x, cmd_y, cmd_yaw), axis=1), dtype=get_global_dtype())

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
        gait_phase: np.ndarray,
        noisy: bool,
    ) -> np.ndarray:
        noise_cfg = self._cfg.noise_config
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
            [
                gyro_obs,
                gravity_obs,
                diff_obs,
                vel_obs,
                last_actions,
                command,
                gait_phase,
                ball_obs,
            ],
            axis=1,
            dtype=get_global_dtype(),
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
    ) -> dict[str, np.ndarray]:
        num_rows = linvel.shape[0]
        diff = dof_pos - self.default_angles
        command = self._command_for_phase(info, env_ids)
        last_actions = info.get("current_actions", np.zeros((num_rows, self._num_action)))
        gait_phase = info.get(
            "gait_phase", np.zeros((num_rows, 2), dtype=get_global_dtype())
        )
        ball_obs = self._ball_obs(env_ids)

        actor = self._build_single_frame_obs(
            command=command,
            ball_obs=ball_obs,
            gravity=-gravity,
            gyro=gyro,
            joint_diff=diff,
            dof_vel=dof_vel,
            last_actions=last_actions,
            gait_phase=gait_phase,
            noisy=True,
        )
        critic_base = self._build_single_frame_obs(
            command=command,
            ball_obs=ball_obs,
            gravity=-gravity,
            gyro=gyro,
            joint_diff=diff,
            dof_vel=dof_vel,
            last_actions=last_actions,
            gait_phase=gait_phase,
            noisy=False,
        )
        critic = np.concatenate(
            (critic_base, np.asarray(linvel * 2.0, dtype=get_global_dtype())),
            axis=1,
            dtype=get_global_dtype(),
        )
        return {"obs": actor, "critic": critic}

    def update_state(self, state):
        linvel = self.get_local_linvel()
        gyro = self.get_gyro()
        gravity = self._backend.get_sensor_data(self._cfg.sensor.upvector)
        dof_pos = self.get_dof_pos()
        dof_vel = self.get_dof_vel()
        step_count = np.asarray(state.info.get("steps", np.zeros(self._num_envs, dtype=np.uint32)))

        ball_pos_w, ball_vel_w, _, _ = self._ball_state()
        right_foot_ball_dist = self._right_foot_ball_distance_xy()
        right_foot_ball_progress = np.maximum(
            self._prev_right_foot_ball_dist.astype(get_global_dtype(), copy=False)
            - right_foot_ball_dist,
            0.0,
        )
        self._runup_success_pending[:] = False
        newly_reached = self._update_runup_reached()
        self._freeze_runup_reach_reward(newly_reached)
        self._freeze_right_foot_ball_reward(newly_reached)
        if np.any(newly_reached):
            self._runup_success_pending[newly_reached] = True
            self._kick_start_steps[newly_reached] = step_count[newly_reached].astype(np.int32)
        newly_scored = self._ball_in_goal(ball_pos_w) & ~self._goal_scored
        self._goal_scored |= self._ball_in_goal(ball_pos_w)

        state.info["runup_reached"] = np.asarray(self._runup_reached, dtype=get_global_dtype())
        state.info["kick_phase"] = state.info["runup_reached"]
        state.info["goal_scored"] = np.asarray(self._goal_scored, dtype=get_global_dtype())
        state.info["runup_success_pending"] = np.asarray(
            self._runup_success_pending, dtype=get_global_dtype()
        )
        state.info["commands"] = self._command_for_phase(state.info)

        max_tilt_rad = np.deg2rad(self._reward_cfg.max_tilt_deg)
        tilt = np.arccos(np.clip(gravity[:, 2], -1, 1))
        term_fall = tilt > max_tilt_rad
        term_low = self._backend.get_base_pos()[:, 2] < self._reward_cfg.min_base_height
        runup_step_limit = int(getattr(self._reward_cfg, "runup_max_steps", K1_PENALTY_RUNUP_MAX_STEPS))
        kick_phase_step_limit = int(
            getattr(self._reward_cfg, "kick_phase_max_steps", K1_PENALTY_KICK_PHASE_MAX_STEPS)
        )
        max_steps = int(getattr(self._reward_cfg, "max_steps", K1_PENALTY_MAX_STEPS))
        runup_timeout_enabled = self._runup_timeout_enabled()
        term_runup_timeout = runup_timeout_enabled & np.logical_and(
            ~self._runup_reached, step_count >= runup_step_limit
        )
        term_episode_timeout = step_count >= max_steps
        kick_elapsed = np.where(
            self._kick_start_steps >= 0,
            step_count.astype(np.int32) - self._kick_start_steps,
            0,
        )
        term_kick_timeout = (
            self._runup_reached
            & ~self._goal_scored
            & (kick_elapsed >= kick_phase_step_limit)
        )
        term_ball_missed_goal = self._runup_reached & self._ball_missed_goal(ball_pos_w)
        term_ball_flew_high = self._runup_reached & self._ball_flew_high(ball_pos_w)
        term_ball_backward = self._runup_reached & self._ball_went_backward(ball_pos_w)
        term_ball_out_y = self._runup_reached & self._ball_out_of_play_y(ball_pos_w)
        term_bad = np.logical_or(term_fall, term_low)
        self._kick_failed = np.logical_or.reduce(
            [
                term_runup_timeout,
                term_kick_timeout,
                term_ball_missed_goal,
                term_ball_flew_high,
                term_ball_backward,
                term_ball_out_y,
            ]
        )
        terminated = np.logical_or.reduce(
            [term_bad, self._kick_failed, self._goal_scored]
        )

        state.info["robot_ball_dist_xy"] = np.asarray(
            self._robot_ball_distance_xy(), dtype=get_global_dtype()
        )
        state.info["right_foot_ball_dist_xy"] = np.asarray(
            right_foot_ball_dist, dtype=get_global_dtype()
        )
        state.info["right_foot_ball_progress"] = np.asarray(
            right_foot_ball_progress, dtype=get_global_dtype()
        )
        state.info["ball_x"] = np.asarray(ball_pos_w[:, 0], dtype=get_global_dtype())
        state.info["ball_y"] = np.asarray(ball_pos_w[:, 1], dtype=get_global_dtype())
        state.info["ball_z"] = np.asarray(ball_pos_w[:, 2], dtype=get_global_dtype())
        state.info["kick_elapsed_steps"] = np.asarray(kick_elapsed, dtype=get_global_dtype())
        state.info["term_fall"] = np.asarray(term_fall, dtype=get_global_dtype())
        state.info["term_low"] = np.asarray(term_low, dtype=get_global_dtype())
        state.info["term_bad"] = np.asarray(term_bad, dtype=get_global_dtype())
        state.info["term_runup_timeout"] = np.asarray(term_runup_timeout, dtype=get_global_dtype())
        state.info["term_kick_timeout"] = np.asarray(term_kick_timeout, dtype=get_global_dtype())
        state.info["term_ball_missed_goal"] = np.asarray(
            term_ball_missed_goal, dtype=get_global_dtype()
        )
        state.info["term_ball_flew_high"] = np.asarray(term_ball_flew_high, dtype=get_global_dtype())
        state.info["term_ball_backward"] = np.asarray(term_ball_backward, dtype=get_global_dtype())
        state.info["term_ball_out_y"] = np.asarray(term_ball_out_y, dtype=get_global_dtype())
        state.info["kick_failed"] = np.asarray(self._kick_failed, dtype=get_global_dtype())
        state.info["term_episode_timeout"] = np.asarray(
            term_episode_timeout, dtype=get_global_dtype()
        )
        state.info["newly_scored"] = np.asarray(newly_scored, dtype=get_global_dtype())
        state.info["terminated"] = np.asarray(terminated, dtype=get_global_dtype())

        reward = self._compute_reward(state.info, linvel, gyro, gravity, dof_pos, dof_vel)
        obs = self._compute_obs(state.info, linvel, gyro, gravity, dof_pos, dof_vel)
        truncated = term_episode_timeout & ~terminated
        state = state.replace(obs=obs, reward=reward, terminated=terminated, truncated=truncated)
        self._prev_right_foot_ball_dist[:] = right_foot_ball_dist.astype(np.float32, copy=False)

        done = state.terminated | state.truncated
        if self._episode_tracker is None or self._penalty_curriculum is None or not np.any(done):
            return state

        done_indices = np.where(done)[0]
        self._update_success_curricula(done_indices)
        episode_lengths = state.info["steps"][done_indices] + 1
        self._episode_tracker.update(episode_lengths)
        self._penalty_curriculum.update(self._episode_tracker.average_length)

        if "log" not in state.info:
            state.info["log"] = {}
        state.info["log"]["curriculum/average_episode_length"] = float(
            self._episode_tracker.average_length
        )
        state.info["log"]["curriculum/penalty_scale"] = float(
            self._penalty_curriculum.current_scale
        )
        state.info["log"]["curriculum/kick_phase_fraction"] = float(np.mean(self._runup_reached))
        state.info["log"]["curriculum/goal_rate"] = float(np.mean(self._goal_scored))
        state.info["log"]["curriculum/runup_success_rate_ema"] = float(
            self._runup_success_rate_ema
        )
        state.info["log"]["curriculum/runup_start_distance_phase"] = float(
            self._runup_start_distance_phase_idx + 1
        )
        state.info["log"]["curriculum/runup_speed_level"] = float(
            self._runup_speed_curriculum_level()
        )
        return state

    def _compute_reward(self, info: dict, linvel, gyro, gravity, dof_pos, dof_vel) -> np.ndarray:
        ctx = self._build_reward_context(info, linvel, gyro, gravity, dof_pos, dof_vel)
        kick_mask = np.asarray(self._runup_reached, dtype=get_global_dtype())
        dtype = get_global_dtype()
        reward = np.zeros((self._num_envs,), dtype=dtype)
        step_count = info.get("steps", np.zeros((self._num_envs,), dtype=np.uint32))
        terminal_event = np.any(
            np.asarray(info.get("terminated", np.zeros((self._num_envs,), dtype=bool)))
        )
        should_log = self._enable_reward_log and (int(step_count[0]) % 4 == 0 or terminal_event)
        log = {} if should_log else info.get("log", {})

        for name, scale in self._reward_cfg.scales.items():
            if scale == 0 or name not in self._reward_fns:
                continue
            rew = self._reward_fns[name](ctx)
            weighted_rew = rew * scale
            if name in K1_PENALTY_KICK_REWARD_KEYS:
                weighted_rew = weighted_rew * kick_mask
            reward += weighted_rew
            if should_log:
                log[f"reward/{name}"] = float(np.mean(weighted_rew))

        if should_log:
            if self._penalty_curriculum is not None:
                log["reward/penalty_scale"] = float(self._penalty_curriculum.current_scale)
            log["curriculum/runup_forward_speed"] = float(self._runup_speed_curriculum_level())
            log["curriculum/kick_phase_fraction"] = float(np.mean(kick_mask))
            log["curriculum/goal_rate"] = float(np.mean(info.get("goal_scored", 0.0)))
            log["curriculum/runup_success_rate_ema"] = float(self._runup_success_rate_ema)
            log["curriculum/runup_start_distance_phase"] = float(
                self._runup_start_distance_phase_idx + 1
            )
            info["log"] = log
        return reward * self._cfg.ctrl_dt

    def _reward_runup_reach(self, ctx: RewardContext) -> np.ndarray:
        dist = ctx.info["robot_ball_dist_xy"]
        dist_reward = self._runup_reach_raw_from_distance(dist)
        reached = self._runup_reached
        reward = dist_reward.copy()
        if np.any(reached):
            reward[reached] = self._runup_reach_reward_frozen[reached]
        runup_phase = 1.0 - np.asarray(ctx.info["kick_phase"], dtype=get_global_dtype())
        return np.asarray(reward * runup_phase, dtype=get_global_dtype())

    def _reward_runup_speed(self, ctx: RewardContext) -> np.ndarray:
        runup_phase = 1.0 - np.asarray(ctx.info["kick_phase"], dtype=get_global_dtype())
        forward_speed = np.maximum(ctx.linvel[:, 0], 0.0)
        min_speed = max(self._runup_speed_curriculum_level(), 1e-6)
        max_speed = max(self._runup_speed_curriculum_max(), min_speed)
        # Minimum-speed floor: below min is penalized linearly; at/above min, faster is better up to max.
        ratio = forward_speed / min_speed
        ratio_cap = max_speed / min_speed
        return np.asarray(np.clip(ratio, 0.0, ratio_cap) * runup_phase, dtype=get_global_dtype())

    def _reward_tracking_lin_vel(self, ctx: RewardContext) -> np.ndarray:
        kick_phase = np.asarray(ctx.info["kick_phase"], dtype=get_global_dtype())
        runup_phase = 1.0 - kick_phase
        kick_rew = locomotion_rewards.tracking_lin_vel(ctx)

        min_speed = max(self._runup_speed_curriculum_level(), 1e-6)
        forward_speed = np.maximum(ctx.linvel[:, 0], 0.0)
        vx_shortfall = np.maximum(min_speed - forward_speed, 0.0)
        runup_error = np.square(vx_shortfall) + np.square(ctx.linvel[:, 1])
        runup_rew = np.exp(-runup_error / ctx.tracking_sigma)
        return np.asarray(runup_phase * runup_rew + kick_phase * kick_rew, dtype=get_global_dtype())

    def _reward_under_speed(self, ctx: RewardContext) -> np.ndarray:
        kick_phase = np.asarray(ctx.info["kick_phase"], dtype=get_global_dtype())
        runup_phase = 1.0 - kick_phase
        kick_pen = locomotion_rewards.under_speed(ctx)

        min_speed = max(self._runup_speed_curriculum_level(), 1e-6)
        forward_speed = np.maximum(ctx.linvel[:, 0], 0.0)
        runup_gap = np.maximum(min_speed - forward_speed, 0.0)
        runup_pen = runup_gap / min_speed
        return np.asarray(runup_phase * runup_pen + kick_phase * kick_pen, dtype=get_global_dtype())

    def _reward_right_foot_to_ball(self, ctx: RewardContext) -> np.ndarray:
        dist = ctx.info["right_foot_ball_dist_xy"]
        current = self._right_foot_ball_raw_from_distance(dist)
        runup_phase = 1.0 - np.asarray(ctx.info["kick_phase"], dtype=get_global_dtype())
        reward = current * runup_phase
        return np.asarray(reward, dtype=get_global_dtype())

    def _reward_right_foot_to_ball_progress(self, ctx: RewardContext) -> np.ndarray:
        progress = np.asarray(ctx.info["right_foot_ball_progress"], dtype=get_global_dtype())
        clip = max(float(getattr(self._reward_cfg, "right_foot_ball_progress_clip", 1.0)), 1e-6)
        runup_phase = 1.0 - np.asarray(ctx.info["kick_phase"], dtype=get_global_dtype())
        # Convert per-step distance reduction to a per-second shaping signal.
        rate = np.clip(progress / max(float(self._cfg.ctrl_dt), 1e-6), 0.0, clip)
        return np.asarray(rate * runup_phase, dtype=get_global_dtype())

    def _event_reward(self, event: np.ndarray) -> np.ndarray:
        # Event bonuses are configured as actual episode bonuses, not per-second rates.
        return np.asarray(event, dtype=get_global_dtype()) / max(float(self._cfg.ctrl_dt), 1e-6)

    def _reward_runup_success(self, ctx: RewardContext) -> np.ndarray:
        # Dense bonus for every step after run-up reaches the ball (scale sets magnitude).
        reached = np.asarray(ctx.info.get("kick_phase", 0.0), dtype=get_global_dtype())
        return np.asarray(reached, dtype=get_global_dtype())

    def _reward_ball_to_goal(self, ctx: RewardContext) -> np.ndarray:
        ball_pos_w, ball_vel_w, _, _ = self._ball_state()
        speed = np.linalg.norm(ball_vel_w[:, :2], axis=1)
        goal_xy = np.asarray([float(K1_PENALTY_GOAL_LINE_X), 0.0], dtype=np.float32)
        target = goal_xy[None, :] - ball_pos_w[:, :2]
        target_norm = np.maximum(np.linalg.norm(target, axis=1, keepdims=True), 1e-6)
        target_dir = target / target_norm
        toward = np.maximum(np.sum(ball_vel_w[:, :2] * target_dir, axis=1), 0.0)
        sigma = max(float(self._reward_cfg.ball_to_goal_sigma), 1e-6)
        active = speed > 0.05
        return np.asarray(np.where(active, toward / sigma, 0.0), dtype=get_global_dtype())

    def _reward_ball_goal_progress(self, ctx: RewardContext) -> np.ndarray:
        ball_x = ctx.info["ball_x"]
        start_x = float(K1_PENALTY_SPOT_XY[0])
        progress = (ball_x - start_x) / max(float(K1_PENALTY_GOAL_LINE_X) - start_x, 1e-6)
        return np.asarray(np.clip(progress, 0.0, 1.0), dtype=get_global_dtype())

    def _reward_ball_kick_speed(self, ctx: RewardContext) -> np.ndarray:
        _, ball_vel_w, _, _ = self._ball_state()
        speed = np.linalg.norm(ball_vel_w[:, :2], axis=1)
        target = float(self._reward_cfg.ball_kick_speed_target)
        sigma = max(float(self._reward_cfg.ball_kick_speed_sigma), 1e-6)
        shortfall = np.maximum(target - speed, 0.0)
        return np.asarray(np.exp(-(shortfall * shortfall) / sigma), dtype=get_global_dtype())

    def _reward_kick_stability(self, ctx: RewardContext) -> np.ndarray:
        assert ctx.gravity is not None
        upright_err = np.sum(np.square(ctx.gravity[:, :2]), axis=1)
        height_err = np.square(ctx.base_height - float(self._reward_cfg.base_height_target))
        return np.asarray(np.exp(-(upright_err + height_err) / 0.15), dtype=get_global_dtype())

    def _reward_goal_scored(self, ctx: RewardContext) -> np.ndarray:
        return self._event_reward(ctx.info.get("newly_scored", 0.0))

    def _reward_kick_failed(self, ctx: RewardContext) -> np.ndarray:
        return self._event_reward(ctx.info.get("kick_failed", 0.0))

    def _reward_termination_bad(self, ctx: RewardContext) -> np.ndarray:
        return self._event_reward(ctx.info["term_bad"])

    def _reward_term_fall(self, ctx: RewardContext) -> np.ndarray:
        return self._event_reward(ctx.info["term_fall"])

    def _reward_term_low(self, ctx: RewardContext) -> np.ndarray:
        return self._event_reward(ctx.info["term_low"])

    def _reward_term_runup_timeout(self, ctx: RewardContext) -> np.ndarray:
        return self._event_reward(ctx.info["term_runup_timeout"])

    def _reward_term_kick_timeout(self, ctx: RewardContext) -> np.ndarray:
        return self._event_reward(ctx.info["term_kick_timeout"])

    def _reward_term_ball_missed_goal(self, ctx: RewardContext) -> np.ndarray:
        return self._event_reward(ctx.info["term_ball_missed_goal"])

    def _reward_term_ball_flew_high(self, ctx: RewardContext) -> np.ndarray:
        return self._event_reward(ctx.info["term_ball_flew_high"])

    def _reward_term_ball_backward(self, ctx: RewardContext) -> np.ndarray:
        return self._event_reward(ctx.info["term_ball_backward"])

    def _reward_term_ball_out_y(self, ctx: RewardContext) -> np.ndarray:
        return self._event_reward(ctx.info["term_ball_out_y"])

    def _reward_penalty_vel_direction(self, ctx: RewardContext) -> np.ndarray:
        vel_xy = self._backend.get_base_lin_vel()[:, :2]
        speed = np.linalg.norm(vel_xy, axis=1)
        target = self._kick_target_dir_xy
        parallel = (vel_xy @ target)[:, None] * target[None, :]
        lateral = np.linalg.norm(vel_xy - parallel, axis=1)
        min_speed = float(self._reward_cfg.vel_direction_min_speed)
        active = speed > min_speed
        return np.asarray(np.where(active, lateral, 0.0), dtype=get_global_dtype())


@registry.envcfg("K1SoccerPenaltyKick")
@dataclass
class K1SoccerPenaltyKickFlatCfg(K1SoccerPenaltyKickCfg):
    reward_config: K1SoccerPenaltyKickRewardConfig | None = None


registry.register_env("K1SoccerPenaltyKick", K1SoccerPenaltyKickEnv, sim_backend="mujoco")
registry.register_env("K1SoccerPenaltyKick", K1SoccerPenaltyKickEnv, sim_backend="motrix")

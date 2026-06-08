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
    K1_PENALTY_STANCE_FOOT_WIDTH_XY_M,
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
            foot_offset = np.asarray(
                getattr(env, "_right_foot_offset_from_base_xy", np.zeros(2, dtype=np.float64)),
                dtype=np.float64,
            )
            target_x = float(K1_PENALTY_SPOT_XY[0]) - distance - float(foot_offset[0])
            offset[:, 0] = target_x - float(env._init_qpos[0])
        limit = float(env._runup_reset_y_jitter_limit())
        jitter = np.random.uniform(-limit, limit, (num_reset,)) if limit > 0.0 else 0.0
        foot_offset_y = float(
            np.asarray(
                getattr(env, "_right_foot_offset_from_base_xy", np.zeros(2, dtype=np.float64)),
                dtype=np.float64,
            )[1]
        )
        target_y = float(K1_PENALTY_SPOT_XY[1]) - foot_offset_y
        offset[:, 1] = target_y - float(env._init_qpos[1]) + jitter
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
    runup_reach_sigma: float = 0.30
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
    runup_start_distance_linear_enabled: bool = False
    runup_start_distance_linear_start: float = 0.10
    runup_start_distance_linear_end: float = 1.10
    runup_start_distance_linear_duration_iterations: int = 10000
    runup_start_distance_linear_jitter: float = 0.02
    runup_start_distance_linear_success_gate_enabled: bool = True
    runup_start_distance_linear_success_threshold: float = 0.20
    runup_start_distance_linear_y_jitter_start_distance: float = 0.50
    runup_start_distance_linear_y_jitter_end_distance: float = 1.10
    runup_start_distance_linear_y_jitter_max: float = 0.05
    kick_success_min_ball_forward_progress: float = 0.30
    right_foot_ball_progress_clip: float = 1.0
    kick_phase_min_ball_forward_speed: float = 0.05
    kick_phase_min_robot_forward_speed: float = 0.15
    kick_phase_max_left_foot_ball_x_error: float = 0.08
    kick_phase_require_plant_foot_lead: bool = True
    kick_phase_min_base_height: float = 0.40
    kick_plant_foot_x_sigma: float = 0.10
    kick_plant_foot_min_lead_m: float = 0.02
    runup_speed_bonus_cap: float = 1.0
    ball_to_goal_sigma: float = 0.35
    ball_goal_progress_sigma: float = 0.05
    ball_kick_speed_sigma: float = 2.0
    ball_kick_speed_min_forward: float = 0.08
    ball_kick_speed_max_forward: float = 2.5
    kick_pose_sigma: float = 0.15
    kick_robot_still_sigma: float = 0.15
    runup_feet_still_sigma: float = 0.015
    ball_movement_min_speed: float = 0.05
    fixed_cmd_lin_speed: float = 0.55
    runup_ball_dist: float = K1_PENALTY_RUNUP_BALL_DIST_M
    runup_max_steps: int = K1_PENALTY_RUNUP_MAX_STEPS
    kick_phase_max_steps: int = K1_PENALTY_KICK_PHASE_MAX_STEPS
    max_steps: int = K1_PENALTY_MAX_STEPS
    runup_timeout_enable_after_iteration: int = 0
    vel_direction_min_speed: float = 0.05
    ball_fail_max_height: float = K1_PENALTY_GOAL_HEIGHT_M + 0.35
    ball_fail_backward_dist: float = 0.45
    ball_pin_max_xy_dist: float = 0.15
    ball_pin_max_ball_speed: float = 0.08
    ball_pin_max_robot_speed: float = 0.08
    ball_pin_min_foot_above_ball_z: float = 0.02


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
        self._right_foot_offset_from_base_xy = self._compute_right_foot_offset_from_base_xy()
        self._runup_reach_left_target_offset_from_ball_xy = (
            self._compute_runup_reach_left_target_offset_from_ball_xy()
        )
        self._dr_manager = DomainRandomizationManager(
            self, K1PenaltyKickDomainRandomizationProvider()
        )
        if cfg.reward_config is None:
            raise ValueError("reward_config must be provided via Hydra configuration")
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
        self._kick_success_rate_ema = 0.0
        self._kick_success_rate_initialized = False
        self._goal_success_rate_ema = 0.0
        self._goal_success_rate_initialized = False
        self._runup_start_distance_phase_idx = 0
        self._runup_start_distance_success_streak = 0
        self._runup_start_distance_linear_current_m = (
            self._runup_start_distance_linear_start()
        )
        self._runup_start_distance_linear_last_iteration = 0
        self._runup_speed_success_level_idx = 0
        self._runup_speed_success_streak = 0
        self._prev_right_foot_ball_dist = self._right_foot_ball_distance_xy().astype(
            np.float32, copy=True
        )
        ball_pos_w, _, _, _ = self._ball_state()
        self._prev_ball_x = ball_pos_w[:, 0].astype(np.float32, copy=True)
        self._prev_ball_y = ball_pos_w[:, 1].astype(np.float32, copy=True)
        left_foot, right_foot = self._foot_positions_w()
        self._prev_foot_pos = np.stack([left_foot, right_foot], axis=1).astype(np.float32)
        self._apply_runup_speed_curriculum(self._runup_speed_curriculum_level())

    def set_training_iteration(self, iteration: int) -> None:
        next_iteration = int(iteration)
        self._advance_runup_start_distance_linear(next_iteration)
        self._training_iteration = next_iteration
        self._apply_runup_speed_curriculum(self._runup_speed_curriculum_level())

    def _compute_right_foot_offset_from_base_xy(self) -> np.ndarray:
        right_foot = self._backend.get_sensor_data("right_foot_pos")
        base_pos = self._backend.get_base_pos()
        if right_foot.shape[0] == 0 or base_pos.shape[0] == 0:
            return np.zeros(2, dtype=np.float64)
        return np.asarray(right_foot[0, :2] - base_pos[0, :2], dtype=np.float64)

    def _compute_runup_reach_left_target_offset_from_ball_xy(self) -> np.ndarray:
        """Left-foot runup target sits one stance foot-width from the ball center."""
        left_foot = self._backend.get_sensor_data("left_foot_pos")
        right_foot = self._backend.get_sensor_data("right_foot_pos")
        if left_foot.shape[0] == 0 or right_foot.shape[0] == 0:
            return np.array([0.0, float(K1_PENALTY_STANCE_FOOT_WIDTH_XY_M)], dtype=np.float64)
        offset = np.asarray(left_foot[0, :2] - right_foot[0, :2], dtype=np.float64)
        norm = float(np.linalg.norm(offset))
        if norm < 1e-6:
            return np.array([0.0, float(K1_PENALTY_STANCE_FOOT_WIDTH_XY_M)], dtype=np.float64)
        return offset

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

    def _runup_start_distance_linear_enabled(self) -> bool:
        return bool(getattr(self._reward_cfg, "runup_start_distance_linear_enabled", False))

    def _runup_start_distance_linear_start(self) -> float:
        return float(getattr(self._reward_cfg, "runup_start_distance_linear_start", 0.10))

    def _runup_start_distance_linear_end(self) -> float:
        start = self._runup_start_distance_linear_start()
        end = float(getattr(self._reward_cfg, "runup_start_distance_linear_end", start))
        return max(end, start)

    def _runup_start_distance_linear_step(self) -> float:
        duration = max(
            int(getattr(self._reward_cfg, "runup_start_distance_linear_duration_iterations", 1)),
            1,
        )
        return (self._runup_start_distance_linear_end() - self._runup_start_distance_linear_start()) / duration

    def _runup_start_distance_linear_gate_passed(self) -> bool:
        if not bool(
            getattr(self._reward_cfg, "runup_start_distance_linear_success_gate_enabled", True)
        ):
            return True
        if not (
            getattr(self, "_kick_success_rate_initialized", False)
            or getattr(self, "_goal_success_rate_initialized", False)
        ):
            return True
        threshold = float(
            getattr(self._reward_cfg, "runup_start_distance_linear_success_threshold", 0.20)
        )
        recent_success = max(
            float(getattr(self, "_kick_success_rate_ema", 0.0)),
            float(getattr(self, "_goal_success_rate_ema", 0.0)),
        )
        return recent_success >= threshold

    def _advance_runup_start_distance_linear(self, next_iteration: int) -> None:
        if not self._runup_start_distance_linear_enabled():
            return
        last_iteration = int(
            getattr(self, "_runup_start_distance_linear_last_iteration", self._training_iteration)
        )
        delta = max(int(next_iteration) - last_iteration, 0)
        if delta > 0 and self._runup_start_distance_linear_gate_passed():
            current = float(
                getattr(
                    self,
                    "_runup_start_distance_linear_current_m",
                    self._runup_start_distance_linear_start(),
                )
            )
            current += float(delta) * self._runup_start_distance_linear_step()
            self._runup_start_distance_linear_current_m = min(
                current,
                self._runup_start_distance_linear_end(),
            )
        self._runup_start_distance_linear_last_iteration = int(next_iteration)

    def _runup_start_distance_current(self) -> float:
        if self._runup_start_distance_linear_enabled():
            return float(
                getattr(
                    self,
                    "_runup_start_distance_linear_current_m",
                    self._runup_start_distance_linear_start(),
                )
            )
        low, high = self._runup_start_distance_range()
        return 0.5 * (low + high)

    def _runup_reset_y_jitter_limit(self) -> float:
        if not self._runup_start_distance_linear_enabled():
            return float(K1_PENALTY_RESET_Y_OFFSET_M)
        current = self._runup_start_distance_current()
        start = float(
            getattr(self._reward_cfg, "runup_start_distance_linear_y_jitter_start_distance", 0.50)
        )
        end = max(
            float(
                getattr(
                    self._reward_cfg,
                    "runup_start_distance_linear_y_jitter_end_distance",
                    start,
                )
            ),
            start,
        )
        max_jitter = max(
            float(getattr(self._reward_cfg, "runup_start_distance_linear_y_jitter_max", 0.0)),
            0.0,
        )
        if max_jitter <= 0.0 or current <= start:
            return 0.0
        if current >= end:
            return max_jitter
        alpha = (current - start) / max(end - start, 1e-6)
        return float(alpha * max_jitter)

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
        if self._runup_start_distance_linear_enabled():
            target = self._runup_start_distance_current()
            jitter = max(
                float(getattr(self._reward_cfg, "runup_start_distance_linear_jitter", 0.0)),
                0.0,
            )
            low = max(self._runup_start_distance_linear_start(), target - jitter)
            high = min(self._runup_start_distance_linear_end(), target + jitter)
            if high <= low:
                return np.full((num_reset,), target, dtype=np.float64)
            return np.random.uniform(low, high, (num_reset,)).astype(np.float64, copy=False)
        low, high = self._runup_start_distance_range()
        if high <= low:
            return np.full((num_reset,), low, dtype=np.float64)
        return np.random.uniform(low, high, (num_reset,)).astype(np.float64, copy=False)

    def _update_success_curricula(self, done_indices: np.ndarray) -> None:
        if len(done_indices) == 0:
            return
        batch_rate = float(np.mean(self._runup_reached[done_indices]))
        if hasattr(self, "_backend"):
            ball_pos_w, _, _, _ = self._ball_state()
            ball_forward_progress = ball_pos_w[done_indices, 0] - float(K1_PENALTY_SPOT_XY[0])
            kick_success_threshold = float(
                getattr(self._reward_cfg, "kick_success_min_ball_forward_progress", 0.30)
            )
            kick_success = self._goal_scored[done_indices] | (
                ball_forward_progress >= kick_success_threshold
            )
            kick_success_rate = float(np.mean(kick_success))
            goal_success_rate = float(np.mean(self._goal_scored[done_indices]))
        else:
            kick_success_rate = 0.0
            goal_success_rate = 0.0
        alpha = float(getattr(self._reward_cfg, "runup_success_rate_ema_alpha", 0.2))
        alpha = float(np.clip(alpha, 0.0, 1.0))
        if not self._runup_success_rate_initialized:
            self._runup_success_rate_ema = batch_rate
            self._runup_success_rate_initialized = True
        else:
            self._runup_success_rate_ema = (
                (1.0 - alpha) * self._runup_success_rate_ema + alpha * batch_rate
            )

        if not getattr(self, "_kick_success_rate_initialized", False):
            self._kick_success_rate_ema = kick_success_rate
            self._kick_success_rate_initialized = True
        else:
            self._kick_success_rate_ema = (
                (1.0 - alpha) * self._kick_success_rate_ema + alpha * kick_success_rate
            )

        if not getattr(self, "_goal_success_rate_initialized", False):
            self._goal_success_rate_ema = goal_success_rate
            self._goal_success_rate_initialized = True
        else:
            self._goal_success_rate_ema = (
                (1.0 - alpha) * self._goal_success_rate_ema + alpha * goal_success_rate
            )

        if (
            self._runup_start_distance_curriculum_enabled()
            and not self._runup_start_distance_linear_enabled()
        ):
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
        ball_pos_w, _, _, _ = self._ball_state()
        self._prev_ball_x[sel] = ball_pos_w[sel, 0].astype(np.float32, copy=False)
        self._prev_ball_y[sel] = ball_pos_w[sel, 1].astype(np.float32, copy=False)
        self._sync_prev_foot_pos(sel)
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
                "penalty_ball_vy": self._reward_penalty_ball_vy,
                "penalty_ball_lateral_progress": self._reward_penalty_ball_lateral_progress,
                "kick_stability": self._reward_kick_stability,
                "kick_plant_foot_x": self._reward_kick_plant_foot_x,
                "kick_plant_foot_order": self._reward_kick_plant_foot_order,
                "penalty_kick_plant_foot_order": self._reward_penalty_kick_plant_foot_order,
                "kick_pose": self._reward_kick_pose,
                "kick_robot_still": self._reward_kick_robot_still,
                "penalty_ball_pin": self._reward_penalty_ball_pin,
                "forward_progress": self._reward_forward_progress,
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
                "penalty_runup_feet_still": self._reward_penalty_runup_feet_still,
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

    def _foot_positions_w(self) -> tuple[np.ndarray, np.ndarray]:
        left_foot = self._backend.get_sensor_data("left_foot_pos")
        right_foot = self._backend.get_sensor_data("right_foot_pos")
        return left_foot, right_foot

    def _sync_prev_foot_pos(self, env_ids: np.ndarray) -> None:
        left_foot, right_foot = self._foot_positions_w()
        foot_pos = np.stack([left_foot, right_foot], axis=1)
        sel = np.asarray(env_ids, dtype=np.intp)
        self._prev_foot_pos[sel] = foot_pos[sel]

    def _update_feet_step_travel(self, info: dict) -> None:
        left_foot, right_foot = self._foot_positions_w()
        foot_pos = np.stack([left_foot, right_foot], axis=1)
        step_delta = foot_pos - self._prev_foot_pos
        left_travel = np.linalg.norm(step_delta[:, 0, :], axis=1)
        right_travel = np.linalg.norm(step_delta[:, 1, :], axis=1)
        info["feet_step_travel"] = np.asarray(
            left_travel + right_travel, dtype=get_global_dtype()
        )
        self._prev_foot_pos[:] = foot_pos.astype(np.float32, copy=False)

    def _right_foot_ball_distance_xy(self) -> np.ndarray:
        right_foot = self._backend.get_sensor_data("right_foot_pos")
        ball_pos_w, _, _, _ = self._ball_state()
        delta = ball_pos_w[:, :2] - right_foot[:, :2]
        return np.linalg.norm(delta, axis=1)

    def _ball_pin_mask(self) -> np.ndarray:
        """Detect reward hack: right foot on / pinning a static ball without kicking."""
        right_foot, _ = self._foot_positions_w()
        ball_pos_w, ball_vel_w, _, _ = self._ball_state()
        dist_xy = self._right_foot_ball_distance_xy()
        max_xy = float(getattr(self._reward_cfg, "ball_pin_max_xy_dist", 0.15))
        max_ball_speed = float(getattr(self._reward_cfg, "ball_pin_max_ball_speed", 0.08))
        max_robot_speed = float(getattr(self._reward_cfg, "ball_pin_max_robot_speed", 0.08))
        min_foot_above = float(
            getattr(self._reward_cfg, "ball_pin_min_foot_above_ball_z", 0.02)
        )
        near = dist_xy <= max_xy
        ball_static = np.linalg.norm(ball_vel_w[:, :2], axis=1) <= max_ball_speed
        robot_static = (
            np.linalg.norm(self._backend.get_base_lin_vel()[:, :2], axis=1) <= max_robot_speed
        )
        foot_on_ball = right_foot[:, 2] > ball_pos_w[:, 2] + min_foot_above
        return near & ball_static & (foot_on_ball | robot_static)

    def _runup_reach_ball_left_target_xy_w(self) -> np.ndarray:
        ball_pos_w, _, _, _ = self._ball_state()
        offset = np.asarray(
            getattr(self, "_runup_reach_left_target_offset_from_ball_xy", np.zeros(2)),
            dtype=np.float64,
        )
        return ball_pos_w[:, :2] + offset[None, :]

    def _left_foot_runup_reach_distance_xy(self) -> np.ndarray:
        left_foot, _ = self._foot_positions_w()
        target_xy = self._runup_reach_ball_left_target_xy_w()
        delta = target_xy - left_foot[:, :2]
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

    def _kick_phase_ready_mask(self) -> np.ndarray:
        dist = self._right_foot_ball_distance_xy()
        inside = dist <= float(self._reward_cfg.right_foot_strike_dist)
        ball_pos_w, ball_vel_w, _, _ = self._ball_state()
        min_ball_vx = float(
            getattr(self._reward_cfg, "kick_phase_min_ball_forward_speed", 0.0)
        )
        ball_moving_forward = ball_vel_w[:, 0] >= min_ball_vx
        min_robot_vx = float(
            getattr(self._reward_cfg, "kick_phase_min_robot_forward_speed", 0.0)
        )
        robot_forward = self.get_local_linvel()[:, 0] >= min_robot_vx
        left_foot, right_foot = self._foot_positions_w()
        max_x_err = max(
            float(getattr(self._reward_cfg, "kick_phase_max_left_foot_ball_x_error", 0.08)),
            1e-6,
        )
        plant_x_ok = np.abs(left_foot[:, 0] - ball_pos_w[:, 0]) <= max_x_err
        if bool(getattr(self._reward_cfg, "kick_phase_require_plant_foot_lead", True)):
            min_lead = max(float(getattr(self._reward_cfg, "kick_plant_foot_min_lead_m", 0.02)), 0.0)
            plant_order_ok = (left_foot[:, 0] - right_foot[:, 0]) >= min_lead
        else:
            plant_order_ok = np.ones(self._num_envs, dtype=bool)
        min_height = float(
            getattr(self._reward_cfg, "kick_phase_min_base_height", self._reward_cfg.min_base_height)
        )
        height_ok = self._backend.get_base_pos()[:, 2] >= min_height
        return (
            inside
            & ball_moving_forward
            & robot_forward
            & plant_x_ok
            & plant_order_ok
            & height_ok
        )

    def _update_runup_reached(self) -> np.ndarray:
        reached = self._kick_phase_ready_mask()
        newly_reached = reached & ~self._runup_reached
        self._runup_reached |= reached
        return newly_reached

    def _runup_reach_raw_from_distance(self, dist: np.ndarray) -> np.ndarray:
        return self._right_foot_ball_raw_from_distance(dist)

    def _freeze_runup_reach_reward(self, newly_reached: np.ndarray) -> None:
        if not np.any(newly_reached):
            return
        dist = self._left_foot_runup_reach_distance_xy()
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
        left_foot, right_foot = self._foot_positions_w()
        right_foot_ball_dist = self._right_foot_ball_distance_xy()
        right_foot_ball_progress = np.maximum(
            self._prev_right_foot_ball_dist.astype(get_global_dtype(), copy=False)
            - right_foot_ball_dist,
            0.0,
        )
        ball_forward_progress = np.maximum(
            ball_pos_w[:, 0].astype(get_global_dtype(), copy=False)
            - self._prev_ball_x.astype(get_global_dtype(), copy=False),
            0.0,
        )
        ball_lateral_progress = np.abs(
            ball_pos_w[:, 1].astype(get_global_dtype(), copy=False)
            - self._prev_ball_y.astype(get_global_dtype(), copy=False)
        )
        self._update_feet_step_travel(state.info)
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
        left_foot_runup_reach_dist = self._left_foot_runup_reach_distance_xy()
        state.info["left_foot_runup_reach_dist_xy"] = np.asarray(
            left_foot_runup_reach_dist, dtype=get_global_dtype()
        )
        ball_pin = self._ball_pin_mask()
        state.info["ball_pin"] = np.asarray(ball_pin, dtype=get_global_dtype())
        state.info["right_foot_ball_dist_xy"] = np.asarray(
            right_foot_ball_dist, dtype=get_global_dtype()
        )
        state.info["right_foot_ball_progress"] = np.asarray(
            right_foot_ball_progress, dtype=get_global_dtype()
        )
        state.info["ball_forward_progress"] = np.asarray(
            ball_forward_progress, dtype=get_global_dtype()
        )
        state.info["ball_vx_forward"] = np.asarray(
            np.maximum(ball_vel_w[:, 0], 0.0), dtype=get_global_dtype()
        )
        state.info["ball_lateral_progress"] = np.asarray(
            ball_lateral_progress, dtype=get_global_dtype()
        )
        state.info["ball_vy_lateral"] = np.asarray(
            np.abs(ball_vel_w[:, 1]), dtype=get_global_dtype()
        )
        state.info["ball_x"] = np.asarray(ball_pos_w[:, 0], dtype=get_global_dtype())
        state.info["ball_y"] = np.asarray(ball_pos_w[:, 1], dtype=get_global_dtype())
        state.info["ball_z"] = np.asarray(ball_pos_w[:, 2], dtype=get_global_dtype())
        state.info["left_foot_x"] = np.asarray(left_foot[:, 0], dtype=get_global_dtype())
        state.info["right_foot_x"] = np.asarray(right_foot[:, 0], dtype=get_global_dtype())
        state.info["left_foot_ball_x_error"] = np.asarray(
            np.abs(left_foot[:, 0] - ball_pos_w[:, 0]), dtype=get_global_dtype()
        )
        state.info["kick_plant_foot_lead_x"] = np.asarray(
            left_foot[:, 0] - right_foot[:, 0], dtype=get_global_dtype()
        )
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
        self._prev_ball_x[:] = ball_pos_w[:, 0].astype(np.float32, copy=False)
        self._prev_ball_y[:] = ball_pos_w[:, 1].astype(np.float32, copy=False)

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
        state.info["log"]["curriculum/ball_pin_fraction"] = float(np.mean(ball_pin))
        state.info["log"]["curriculum/kick_phase_fraction"] = float(np.mean(self._runup_reached))
        state.info["log"]["curriculum/goal_rate"] = float(np.mean(self._goal_scored))
        state.info["log"]["curriculum/mean_robot_ball_dist_xy"] = float(
            np.mean(state.info["robot_ball_dist_xy"])
        )
        state.info["log"]["curriculum/mean_right_foot_ball_dist_xy"] = float(
            np.mean(state.info["right_foot_ball_dist_xy"])
        )
        state.info["log"]["curriculum/runup_start_distance_target"] = float(
            self._runup_start_distance_current()
        )
        state.info["log"]["curriculum/runup_reset_y_jitter_limit"] = float(
            self._runup_reset_y_jitter_limit()
        )
        state.info["log"]["curriculum/runup_success_rate_ema"] = float(
            self._runup_success_rate_ema
        )
        state.info["log"]["curriculum/kick_success_rate_ema"] = float(
            self._kick_success_rate_ema
        )
        state.info["log"]["curriculum/goal_success_rate_ema"] = float(
            self._goal_success_rate_ema
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
            log["curriculum/mean_robot_ball_dist_xy"] = float(
                np.mean(info.get("robot_ball_dist_xy", 0.0))
            )
            log["curriculum/mean_right_foot_ball_dist_xy"] = float(
                np.mean(info.get("right_foot_ball_dist_xy", 0.0))
            )
            log["curriculum/runup_start_distance_target"] = float(
                self._runup_start_distance_current()
            )
            log["curriculum/runup_reset_y_jitter_limit"] = float(
                self._runup_reset_y_jitter_limit()
            )
            log["curriculum/goal_rate"] = float(np.mean(info.get("goal_scored", 0.0)))
            log["curriculum/runup_success_rate_ema"] = float(self._runup_success_rate_ema)
            log["curriculum/kick_success_rate_ema"] = float(self._kick_success_rate_ema)
            log["curriculum/goal_success_rate_ema"] = float(self._goal_success_rate_ema)
            log["curriculum/runup_start_distance_phase"] = float(
                self._runup_start_distance_phase_idx + 1
            )
            info["log"] = log
        return reward * self._cfg.ctrl_dt

    def _ball_pin_gate(self, ctx: RewardContext) -> np.ndarray:
        return 1.0 - np.asarray(ctx.info.get("ball_pin", 0.0), dtype=get_global_dtype())

    def _reward_runup_reach(self, ctx: RewardContext) -> np.ndarray:
        dist = ctx.info["left_foot_runup_reach_dist_xy"]
        dist_reward = self._runup_reach_raw_from_distance(dist)
        reached = self._runup_reached
        reward = dist_reward.copy()
        if np.any(reached):
            reward[reached] = self._runup_reach_reward_frozen[reached]
        runup_phase = 1.0 - np.asarray(ctx.info["kick_phase"], dtype=get_global_dtype())
        return np.asarray(reward * runup_phase * self._ball_pin_gate(ctx), dtype=get_global_dtype())

    def _reward_runup_speed(self, ctx: RewardContext) -> np.ndarray:
        runup_phase = 1.0 - np.asarray(ctx.info["kick_phase"], dtype=get_global_dtype())
        forward_speed = np.maximum(ctx.linvel[:, 0], 0.0)
        min_speed = max(self._runup_speed_curriculum_level(), 1e-6)
        max_speed = max(self._runup_speed_curriculum_max(), min_speed)
        meet_min = np.clip(forward_speed / min_speed, 0.0, 1.0)
        bonus_cap = max(float(getattr(self._reward_cfg, "runup_speed_bonus_cap", 1.0)), 0.0)
        speed_bonus = np.clip(
            (forward_speed - min_speed) / max(max_speed - min_speed, 1e-6),
            0.0,
            bonus_cap,
        )
        return np.asarray((meet_min + speed_bonus) * runup_phase, dtype=get_global_dtype())

    def _reward_forward_progress(self, ctx: RewardContext) -> np.ndarray:
        runup_phase = 1.0 - np.asarray(ctx.info["kick_phase"], dtype=get_global_dtype())
        forward_speed = np.maximum(ctx.linvel[:, 0], 0.0)
        max_speed = max(self._runup_speed_curriculum_max(), 1e-6)
        return np.asarray(
            runup_phase * np.clip(forward_speed / max_speed, 0.0, 1.5),
            dtype=get_global_dtype(),
        )

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
        reward = current * runup_phase * self._ball_pin_gate(ctx)
        return np.asarray(reward, dtype=get_global_dtype())

    def _reward_right_foot_to_ball_progress(self, ctx: RewardContext) -> np.ndarray:
        progress = np.asarray(ctx.info["right_foot_ball_progress"], dtype=get_global_dtype())
        clip = max(float(getattr(self._reward_cfg, "right_foot_ball_progress_clip", 1.0)), 1e-6)
        runup_phase = 1.0 - np.asarray(ctx.info["kick_phase"], dtype=get_global_dtype())
        # Convert per-step distance reduction to a per-second shaping signal.
        rate = np.clip(progress / max(float(self._cfg.ctrl_dt), 1e-6), 0.0, clip)
        return np.asarray(rate * runup_phase * self._ball_pin_gate(ctx), dtype=get_global_dtype())

    def _reward_penalty_ball_pin(self, ctx: RewardContext) -> np.ndarray:
        runup_phase = 1.0 - np.asarray(ctx.info["kick_phase"], dtype=get_global_dtype())
        pin = np.asarray(ctx.info.get("ball_pin", 0.0), dtype=get_global_dtype())
        return np.asarray(pin * runup_phase, dtype=get_global_dtype())

    def _reward_penalty_runup_feet_still(self, ctx: RewardContext) -> np.ndarray:
        travel = np.asarray(ctx.info["feet_step_travel"], dtype=get_global_dtype())
        sigma = max(float(getattr(self._reward_cfg, "runup_feet_still_sigma", 0.015)), 1e-6)
        stillness = np.exp(-travel / sigma)
        runup_phase = 1.0 - np.asarray(ctx.info["kick_phase"], dtype=get_global_dtype())
        return np.asarray(stillness * runup_phase * self._ball_pin_gate(ctx), dtype=get_global_dtype())

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

    def _ball_forward_vx(self) -> np.ndarray:
        _, ball_vel_w, _, _ = self._ball_state()
        return np.maximum(ball_vel_w[:, 0], 0.0)

    def _reward_lateral_magnitude_exp(
        self,
        magnitude: np.ndarray,
        *,
        sigma: float,
        min_magnitude: float,
    ) -> np.ndarray:
        sigma = max(float(sigma), 1e-6)
        min_magnitude = max(float(min_magnitude), 0.0)
        shaped = 1.0 - np.exp(-magnitude / sigma)
        return np.asarray(
            np.where(magnitude >= min_magnitude, shaped, 0.0),
            dtype=get_global_dtype(),
        )

    def _reward_ball_goal_progress(self, ctx: RewardContext) -> np.ndarray:
        forward_m = np.asarray(ctx.info["ball_forward_progress"], dtype=get_global_dtype())
        sigma = max(float(getattr(self._reward_cfg, "ball_goal_progress_sigma", 0.05)), 1e-6)
        shaped = 1.0 - np.exp(-forward_m / sigma)
        return np.asarray(np.where(forward_m > 0.0, shaped, 0.0), dtype=get_global_dtype())

    def _reward_ball_kick_speed(self, ctx: RewardContext) -> np.ndarray:
        forward_speed = self._ball_forward_vx()
        sigma = max(float(getattr(self._reward_cfg, "ball_kick_speed_sigma", 2.0)), 1e-6)
        min_forward = float(getattr(self._reward_cfg, "ball_kick_speed_min_forward", 0.08))
        max_forward = max(float(getattr(self._reward_cfg, "ball_kick_speed_max_forward", 2.5)), 0.0)
        capped_speed = np.minimum(forward_speed, max_forward)
        shaped = np.expm1(capped_speed / sigma)
        return np.asarray(
            np.where(forward_speed >= min_forward, shaped, 0.0),
            dtype=get_global_dtype(),
        )

    def _reward_kick_pose(self, ctx: RewardContext) -> np.ndarray:
        pose_err = locomotion_rewards.weighted_pose(ctx)
        sigma = max(float(getattr(self._reward_cfg, "kick_pose_sigma", 0.15)), 1e-6)
        return np.asarray(np.exp(-pose_err / sigma), dtype=get_global_dtype())

    def _reward_kick_robot_still(self, ctx: RewardContext) -> np.ndarray:
        speed = np.linalg.norm(ctx.linvel[:, :3], axis=1)
        sigma = max(float(getattr(self._reward_cfg, "kick_robot_still_sigma", 0.15)), 1e-6)
        return np.asarray(np.exp(-speed / sigma), dtype=get_global_dtype())

    def _reward_penalty_ball_vy(self, ctx: RewardContext) -> np.ndarray:
        lateral_speed = np.asarray(ctx.info["ball_vy_lateral"], dtype=get_global_dtype())
        sigma = float(getattr(self._reward_cfg, "ball_kick_speed_sigma", 2.0))
        min_lateral = float(getattr(self._reward_cfg, "ball_kick_speed_min_forward", 0.08))
        return self._reward_lateral_magnitude_exp(
            lateral_speed,
            sigma=sigma,
            min_magnitude=min_lateral,
        )

    def _reward_penalty_ball_lateral_progress(self, ctx: RewardContext) -> np.ndarray:
        lateral_m = np.asarray(ctx.info["ball_lateral_progress"], dtype=get_global_dtype())
        sigma = max(float(getattr(self._reward_cfg, "ball_goal_progress_sigma", 0.05)), 1e-6)
        return self._reward_lateral_magnitude_exp(
            lateral_m,
            sigma=sigma,
            min_magnitude=0.0,
        )

    def _reward_kick_stability(self, ctx: RewardContext) -> np.ndarray:
        assert ctx.gravity is not None
        upright_err = np.sum(np.square(ctx.gravity[:, :2]), axis=1)
        height_err = np.square(ctx.base_height - float(self._reward_cfg.base_height_target))
        return np.asarray(np.exp(-(upright_err + height_err) / 0.15), dtype=get_global_dtype())

    def _reward_kick_plant_foot_x(self, ctx: RewardContext) -> np.ndarray:
        x_error = np.asarray(ctx.info["left_foot_ball_x_error"], dtype=get_global_dtype())
        sigma = max(float(getattr(self._reward_cfg, "kick_plant_foot_x_sigma", 0.10)), 1e-6)
        return np.asarray(np.exp(-np.square(x_error) / (sigma * sigma)), dtype=get_global_dtype())

    def _reward_kick_plant_foot_order(self, ctx: RewardContext) -> np.ndarray:
        lead = np.asarray(ctx.info["kick_plant_foot_lead_x"], dtype=get_global_dtype())
        min_lead = max(float(getattr(self._reward_cfg, "kick_plant_foot_min_lead_m", 0.02)), 0.0)
        return np.asarray(np.clip(lead / max(min_lead, 1e-6), 0.0, 1.0), dtype=get_global_dtype())

    def _reward_penalty_kick_plant_foot_order(self, ctx: RewardContext) -> np.ndarray:
        lead = np.asarray(ctx.info["kick_plant_foot_lead_x"], dtype=get_global_dtype())
        return np.asarray(lead <= 0.0, dtype=get_global_dtype())

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

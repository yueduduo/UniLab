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
from unilab.envs.locomotion.common.rewards import RewardContext
from unilab.envs.locomotion.k1.base import NoiseConfig
from unilab.envs.locomotion.k1.constants import (
    K1_ACTUATOR_JOINT_ORDER,
    K1_NUM_ACTION,
    K1_SOCCER_CURRICULUM_PHASE1_WAYPOINT_XY,
    K1_SOCCER_CURRICULUM_START_XY,
    K1_SOCCER_RESET_X_OFFSET_M,
    K1_SOCCER_PHASE1_MAX_STEPS,
    K1_SOCCER_PHASE1_REWARD_KEYS,
    K1_SOCCER_PHASE2_REWARD_KEYS,
    K1_SOCCER_PHASE1_WAYPOINT_GEOM_PENDING,
    K1_SOCCER_PHASE1_WAYPOINT_GEOM_REACHED,
    K1_SOCCER_PHASE1_WAYPOINT_MARKER_HIDDEN_Z,
    K1_SOCCER_PHASE1_WAYPOINT_SPHERE_RADIUS,
    K1_SOCCER_PHASE1_WAYPOINT_Z,
    k1_keyframe_robot_joint_qpos,
    k1_soccer_push_actor_obs_dim,
    k1_soccer_push_critic_obs_dim,
)
from unilab.dr import DomainRandomizationManager
from unilab.envs.locomotion.k1.joystick import (
    K1DomainRandConfig,
    K1RewardConfig,
    K1WalkDomainRandomizationProvider,
    K1WalkEnv,
    K1WalkEnvCfg,
)

_LEFT_HIP_YAW_IDX = K1_ACTUATOR_JOINT_ORDER.index("Left_Hip_Yaw")
_RIGHT_HIP_YAW_IDX = K1_ACTUATOR_JOINT_ORDER.index("Right_Hip_Yaw")


class K1SoccerDomainRandomizationProvider(K1WalkDomainRandomizationProvider):
    """Walk-flat reset/command sampling; colinear curriculum keeps keyframe yaw (+X)."""

    def _sample_reset_xy_offset(self, env: Any, num_reset: int) -> np.ndarray:
        offset = np.zeros((num_reset, 2), dtype=np.float64)
        limit = float(K1_SOCCER_RESET_X_OFFSET_M)
        offset[:, 0] = np.random.uniform(-limit, limit, (num_reset,))
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
class K1SoccerDribbleRewardConfig(K1RewardConfig):
    ball_keep_distance: float = 0.25
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
    phase1_max_steps: int = 200
    phase1_reach_sigma: float = 0.3
    phase1_vel_direction_min_speed: float = 0.05
    phase1_vel_direction_phase2_scale: float = 0.25


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
                [0.4, 0.0, 0.0],
                [0.7, 0.0, 0.0],
            ],
            rel_standing_envs=0.0,
        )
    )
    asset: K1SoccerAsset = field(default_factory=K1SoccerAsset)
    noise_config: K1SoccerNoiseConfig = field(default_factory=K1SoccerNoiseConfig)  # type: ignore[assignment]
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
    # Motrix impulse solver 在 box(COL_Collider)+mesh(ball) 下需 0.002 才稳定（0.005 发散）。
    # ctrl_dt 保持 0.02 → decimation=10。见 scene_soccer_dribble_minimal.xml / zzz_debug/BALL_PHYSICS.md。
    sim_dt: float = 0.002
    obs_frame_stack: int = 1
    add_body_sensors: bool = True
    reward_config: K1SoccerDribbleRewardConfig | None = None


class K1SoccerDribbleEnv(K1WalkEnv):
    _cfg: K1SoccerDribbleCfg
    _reward_cfg: K1SoccerDribbleRewardConfig

    def __init__(self, cfg: K1SoccerDribbleCfg, num_envs=1, backend_type="mujoco"):
        super().__init__(cfg, num_envs=num_envs, backend_type=backend_type)
        # Keyframe qpos ends with ball free-joint coords; do not use tail [-num_action:].
        self.default_angles = np.asarray(
            k1_keyframe_robot_joint_qpos(self._init_qpos, self._num_action),
            dtype=self.default_angles.dtype,
        )
        # K1WalkEnv already materialized the backend; swap DR provider only (soccer reset obs).
        self._dr_manager = DomainRandomizationManager(self, K1SoccerDomainRandomizationProvider())
        self._reward_cfg = cfg.reward_config
        self._ball_body_ids = self._backend.get_body_ids(["ball"])
        if self._ball_body_ids.shape != (1,):
            raise ValueError("soccer dribble task expects exactly one body named 'ball'")
        if int(self._cfg.obs_frame_stack) != 1:
            raise ValueError("K1SoccerDribble push profile requires obs_frame_stack=1")
        self._actor_obs_dim = k1_soccer_push_actor_obs_dim(self._num_action)
        self._critic_obs_dim = k1_soccer_push_critic_obs_dim(self._num_action)
        left_foot, right_foot = self._foot_positions()
        self._prev_foot_pos = np.stack([left_foot, right_foot], axis=1).astype(np.float32)
        self._phase1_waypoint_xy = np.asarray(
            K1_SOCCER_CURRICULUM_PHASE1_WAYPOINT_XY, dtype=np.float32
        )
        phase1_delta_xy = (
            np.asarray(K1_SOCCER_CURRICULUM_PHASE1_WAYPOINT_XY, dtype=np.float32)
            - np.asarray(K1_SOCCER_CURRICULUM_START_XY, dtype=np.float32)
        )
        self._phase1_target_dir_xy = phase1_delta_xy / max(float(np.linalg.norm(phase1_delta_xy)), 1e-6)
        self._phase1_reached = np.zeros(self._num_envs, dtype=bool)
        self._phase1_reach_reward_frozen = np.zeros(self._num_envs, dtype=np.float32)
        self._phase1_visual_reached = False
        self._phase1_waypoint_center = np.asarray(
            [
                self._phase1_waypoint_xy[0],
                self._phase1_waypoint_xy[1],
                K1_SOCCER_PHASE1_WAYPOINT_Z,
            ],
            dtype=np.float32,
        )
        self._phase1_waypoint_active_pos = np.asarray(self._phase1_waypoint_center, dtype=np.float64)
        self._phase1_waypoint_hidden_pos = np.asarray(
            [
                self._phase1_waypoint_xy[0],
                self._phase1_waypoint_xy[1],
                K1_SOCCER_PHASE1_WAYPOINT_MARKER_HIDDEN_Z,
            ],
            dtype=np.float64,
        )
        self._waypoint_visual_supported = True
        self._sync_phase1_waypoint_visual(reached=False)

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
        self._phase1_reached[sel] = False
        self._phase1_reach_reward_frozen[sel] = 0.0
        self._sync_phase1_waypoint_visual(reached=bool(np.any(self._phase1_reached)))
        self._sync_prev_foot_pos(env_indices)
        return obs, info

    def _sync_phase1_waypoint_visual(self, *, reached: bool) -> None:
        if not self._waypoint_visual_supported:
            return
        if reached == self._phase1_visual_reached:
            return
        try:
            if reached:
                self._backend.set_world_geom_pos(
                    K1_SOCCER_PHASE1_WAYPOINT_GEOM_PENDING, self._phase1_waypoint_hidden_pos
                )
                self._backend.set_world_geom_pos(
                    K1_SOCCER_PHASE1_WAYPOINT_GEOM_REACHED, self._phase1_waypoint_active_pos
                )
            else:
                self._backend.set_world_geom_pos(
                    K1_SOCCER_PHASE1_WAYPOINT_GEOM_PENDING, self._phase1_waypoint_active_pos
                )
                self._backend.set_world_geom_pos(
                    K1_SOCCER_PHASE1_WAYPOINT_GEOM_REACHED, self._phase1_waypoint_hidden_pos
                )
        except (NotImplementedError, ValueError):
            # Motrix: geom.local_pose is not writable. MuJoCo training model uses
            # discardvisual and drops non-collision waypoint marker geoms.
            self._waypoint_visual_supported = False
            return
        self._phase1_visual_reached = reached

    def _update_phase1_reached(self) -> np.ndarray:
        base_pos = self._backend.get_base_pos()
        dist = np.linalg.norm(base_pos - self._phase1_waypoint_center, axis=1)
        inside = dist <= float(K1_SOCCER_PHASE1_WAYPOINT_SPHERE_RADIUS)
        newly_reached = inside & ~self._phase1_reached
        self._phase1_reached |= inside
        self._sync_phase1_waypoint_visual(reached=bool(self._phase1_reached[0]))
        return newly_reached

    def _phase1_reach_raw_from_distance(self, dist: np.ndarray) -> np.ndarray:
        sigma = max(float(self._reward_cfg.phase1_reach_sigma), 1e-6)
        return np.asarray(1.0 - np.tanh(dist / sigma), dtype=get_global_dtype())

    def _freeze_phase1_reach_reward(self, newly_reached: np.ndarray) -> None:
        if not np.any(newly_reached):
            return
        dist = self._phase1_waypoint_distance()
        reach = self._phase1_reach_raw_from_distance(dist)
        self._phase1_reach_reward_frozen[newly_reached] = reach[newly_reached].astype(
            np.float32, copy=False
        )

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
        return {"obs": self._actor_obs_dim, "critic": self._critic_obs_dim}

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
                "phase1_complete": self._reward_phase1_complete,
                "phase1_reach": self._reward_phase1_reach,
                "penalty_phase1_vel_direction": self._reward_penalty_phase1_vel_direction,
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

    def _ball_obs_for_curriculum(self, env_ids: np.ndarray | None = None) -> np.ndarray:
        """Phase 1: zero ball obs (walk-aligned MDP). Phase 2: true relative ball state."""
        ball_obs = self._ball_obs(env_ids)
        if env_ids is None:
            phase2 = self._phase1_reached
        else:
            phase2 = self._phase1_reached[np.asarray(env_ids, dtype=np.intp)]
        if not np.any(phase2):
            return np.zeros_like(ball_obs)
        if np.all(phase2):
            return ball_obs
        masked = np.zeros_like(ball_obs)
        masked[phase2] = ball_obs[phase2]
        return masked

    def _command_for_curriculum(
        self, info: dict, env_ids: np.ndarray | None = None
    ) -> np.ndarray:
        """Phase 1: episode-fixed walk commands from reset. Phase 2: steer toward ball."""
        cmd = np.array(info["commands"], copy=True)
        if env_ids is None:
            phase2 = self._phase1_reached
        else:
            phase2 = self._phase1_reached[np.asarray(env_ids, dtype=np.intp)]
        if not np.any(phase2):
            return cmd
        cmd_ball = self._command_toward_ball(env_ids)
        cmd[phase2] = cmd_ball[phase2]
        return cmd

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
    ) -> dict[str, np.ndarray]:
        num_rows = linvel.shape[0]
        diff = dof_pos - self.default_angles
        command = self._command_for_curriculum(info, env_ids)
        last_actions = info.get("current_actions", np.zeros((num_rows, self._num_action)))
        gait_phase = info.get(
            "gait_phase", np.zeros((num_rows, 2), dtype=get_global_dtype())
        )
        ball_obs = self._ball_obs_for_curriculum(env_ids)

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

        _, _, rel_pos_b, _ = self._ball_state()
        ball_dist_xy = np.linalg.norm(rel_pos_b[:, :2], axis=1)
        newly_reached = self._update_phase1_reached()
        self._freeze_phase1_reach_reward(newly_reached)
        state.info["phase1_just_reached"] = np.asarray(newly_reached, dtype=get_global_dtype())
        state.info["phase1_reached"] = np.asarray(self._phase1_reached, dtype=get_global_dtype())
        state.info["phase2_mask"] = state.info["phase1_reached"]
        state.info["commands"] = self._command_for_curriculum(state.info)

        max_tilt_rad = np.deg2rad(self._reward_cfg.max_tilt_deg)
        tilt = np.arccos(np.clip(gravity[:, 2], -1, 1))
        term_fall = tilt > max_tilt_rad
        term_low = self._backend.get_base_pos()[:, 2] < self._reward_cfg.min_base_height
        phase1_steps = np.asarray(state.info.get("steps", np.zeros(self._num_envs, dtype=np.uint32)))
        phase1_step_limit = int(
            getattr(self._reward_cfg, "phase1_max_steps", K1_SOCCER_PHASE1_MAX_STEPS)
        )
        term_phase1_timeout = np.logical_and(
            ~self._phase1_reached,
            phase1_steps >= phase1_step_limit,
        )
        term_ball_lost = np.logical_and(
            self._phase1_reached,
            ball_dist_xy > float(self._reward_cfg.ball_lost_distance_hard),
        )
        term_bad = np.logical_or(term_fall, term_low)
        terminated = np.logical_or.reduce([term_bad, term_ball_lost, term_phase1_timeout])

        state.info["ball_dist_xy"] = np.asarray(ball_dist_xy, dtype=get_global_dtype())
        state.info["term_bad"] = np.asarray(term_bad, dtype=get_global_dtype())
        state.info["term_phase1_timeout"] = np.asarray(
            term_phase1_timeout, dtype=get_global_dtype()
        )
        state.info["terminated"] = np.asarray(terminated, dtype=get_global_dtype())
        self._update_feet_step_travel(state.info)
        reward = self._compute_reward(state.info, linvel, gyro, gravity, dof_pos, dof_vel)
        obs = self._compute_obs(state.info, linvel, gyro, gravity, dof_pos, dof_vel)
        state = state.replace(obs=obs, reward=reward, terminated=terminated)

        done = state.terminated | state.truncated
        if self._episode_tracker is None or self._penalty_curriculum is None or not np.any(done):
            return state

        done_indices = np.where(done)[0]
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
        return state

    def _compute_reward(self, info: dict, linvel, gyro, gravity, dof_pos, dof_vel) -> np.ndarray:
        ctx = self._build_reward_context(info, linvel, gyro, gravity, dof_pos, dof_vel)
        phase2_mask = np.asarray(self._phase1_reached, dtype=get_global_dtype())
        dtype = get_global_dtype()
        reward = np.zeros((self._num_envs,), dtype=dtype)
        step_count = info.get("steps", np.zeros((self._num_envs,), dtype=np.uint32))
        should_log = self._enable_reward_log and (int(step_count[0]) % 4 == 0)
        log = {} if should_log else info.get("log", {})

        for name, scale in self._reward_cfg.scales.items():
            if scale == 0 or name not in self._reward_fns:
                continue
            rew = self._reward_fns[name](ctx)
            weighted_rew = rew * scale
            if name in K1_SOCCER_PHASE2_REWARD_KEYS:
                weighted_rew = weighted_rew * phase2_mask
            elif name in K1_SOCCER_PHASE1_REWARD_KEYS:
                weighted_rew = weighted_rew * (1.0 - phase2_mask)
            elif name == "penalty_phase1_vel_direction":
                phase2_factor = float(self._reward_cfg.phase1_vel_direction_phase2_scale)
                multiplier = (1.0 - phase2_mask) + phase2_mask * phase2_factor
                weighted_rew = weighted_rew * multiplier
            reward += weighted_rew
            if should_log:
                log[f"reward/{name}"] = float(np.mean(weighted_rew))

        if should_log:
            log["curriculum/phase2_fraction"] = float(np.mean(phase2_mask))
            if self._penalty_curriculum is not None:
                log["reward/penalty_scale"] = float(self._penalty_curriculum.current_scale)
            info["log"] = log
        return reward * self._cfg.ctrl_dt

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

    def _phase1_waypoint_distance(self) -> np.ndarray:
        base_pos = self._backend.get_base_pos()
        return np.linalg.norm(base_pos - self._phase1_waypoint_center, axis=1)

    def _reward_phase1_reach(self, ctx: RewardContext) -> np.ndarray:
        """Before phase-1 complete: 1 - tanh(dist / std). After: frozen reach at completion."""
        dist = self._phase1_waypoint_distance()
        dist_reward = self._phase1_reach_raw_from_distance(dist)
        reached = self._phase1_reached
        reward = dist_reward.copy()
        if np.any(reached):
            reward[reached] = self._phase1_reach_reward_frozen[reached]
        return np.asarray(reward, dtype=get_global_dtype())

    def _reward_penalty_phase1_vel_direction(self, ctx: RewardContext) -> np.ndarray:
        """Penalize world-frame lateral speed deviating from start -> phase-1 waypoint (+X)."""
        vel_xy = self._backend.get_base_lin_vel()[:, :2]
        speed = np.linalg.norm(vel_xy, axis=1)
        target = self._phase1_target_dir_xy
        parallel = (vel_xy @ target)[:, None] * target[None, :]
        lateral = np.linalg.norm(vel_xy - parallel, axis=1)
        min_speed = float(self._reward_cfg.phase1_vel_direction_min_speed)
        active = speed > min_speed
        return np.asarray(np.where(active, lateral, 0.0), dtype=get_global_dtype())

    def _reward_phase1_complete(self, ctx: RewardContext) -> np.ndarray:
        """Per-step bonus while alive after the phase-1 waypoint is reached."""
        phase1 = np.asarray(ctx.info["phase1_reached"], dtype=get_global_dtype())
        terminated = np.asarray(ctx.info.get("terminated", 0.0), dtype=get_global_dtype())
        return phase1 * (1.0 - terminated)


@registry.envcfg("K1SoccerDribble")
@dataclass
class K1SoccerDribbleFlatCfg(K1SoccerDribbleCfg):
    reward_config: K1SoccerDribbleRewardConfig | None = None


registry.register_env("K1SoccerDribble", K1SoccerDribbleEnv, sim_backend="mujoco")
registry.register_env("K1SoccerDribble", K1SoccerDribbleEnv, sim_backend="motrix")


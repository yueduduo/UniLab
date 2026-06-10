"""K1 velocity-command locomotion with 46000-compatible 78-dim actor obs.

Joystick-style training: episode-fixed ``commands`` in body frame
``[vx, vy, yaw_rate]`` (physical m/s and rad/s). Default sampling range is
``[-1, 1]`` per axis; a fraction of envs receive ``[0, 0, 0]`` for standing.

Actor/critic layout, trunk sensors, position actuators, joint defaults, and
``apply_action`` deploy permutation match ``K1SoccerDribbleCompat`` /
``k1_model_46000`` deployment contract.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.np_env import NpEnvState
from unilab.base.scene import SceneCfg
from unilab.dr import DomainRandomizationManager
from unilab.dtype_config import get_global_dtype
from unilab.envs.locomotion.common.commands import Commands
from unilab.envs.locomotion.common.dr_provider import LocomotionDRProvider
from unilab.envs.locomotion.common.rewards import RewardContext
from unilab.envs.locomotion.g1.joystick import (
    LEFT_FOOT_CONTACT_SENSORS,
    RIGHT_FOOT_CONTACT_SENSORS,
    compute_aggregated_foot_contact,
    compute_command_locomotion_mask,
    compute_feet_phase_contact_targets,
    compute_feet_phase_height_targets,
    compute_standing_command_mask,
)
from unilab.envs.locomotion.k1.constants import (
    K1_ACTUATOR_TO_DEPLOY_PERM,
    K1_DEPLOY_TO_ACTUATOR_PERM,
    K1_NUM_ACTION,
    k1_vel_cmd_compat_actor_obs_dim,
    k1_vel_cmd_compat_critic_obs_dim,
)
from unilab.envs.locomotion.k1.base import ControlConfig
from unilab.envs.locomotion.k1.joystick import (
    InitState,
    K1DomainRandConfig,
    K1RewardConfig,
    K1WalkDomainRandomizationProvider,
    K1WalkEnv,
    K1WalkEnvCfg,
)
from unilab.envs.locomotion.k1.soccer_dribble import K1SoccerAsset, K1SoccerNoiseConfig

# OBS_SCALE mirrors MOS-SIM runtime_config.py / K1SoccerDribbleCompat.
_LIN_VEL_SCALE: float = 1.0
_ANG_VEL_SCALE: float = 0.2
_GRAVITY_SCALE: float = 1.0
_CMD_SCALE: float = 1.0
_JOINT_POS_SCALE: float = 1.0
_JOINT_VEL_SCALE: float = 0.05
_LAST_ACTION_SCALE: float = 1.0

# Stand keyframe base height (scene_vel_cmd_compat.xml / soccer dribble minimal).
_K1_VEL_CMD_STAND_Z: float = 0.5743699226900935


@dataclass
class K1VelCmdInitState(InitState):
    pos = [0.0, 0.0, _K1_VEL_CMD_STAND_Z]


class K1VelCmdCompatDomainRandomizationProvider(K1WalkDomainRandomizationProvider):
    """Walk reset sampling without zeroing small xy commands (preserve fine vel tracking)."""

    def _sample_commands(self, env: Any, num_reset: int) -> np.ndarray:
        commands = LocomotionDRProvider._sample_commands(self, env, num_reset)
        standing_prob = float(getattr(env.cfg.commands, "rel_standing_envs", 0.0))
        if standing_prob > 0.0:
            standing = np.random.uniform(size=(num_reset,)) < min(standing_prob, 1.0)
            commands[standing] = 0.0
        if getattr(env.cfg.commands, "heading_command", False):
            commands[:, 2] = 0.0
        return commands


@dataclass
class K1VelCmdCompatCfg(K1WalkEnvCfg):
    """Velocity-command flat walk; deploy-compatible 78-dim actor obs."""

    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(
            model_file=str(ASSETS_ROOT_PATH / "robots" / "k1" / "scene_vel_cmd_compat.xml")
        )
    )
    init_state: K1VelCmdInitState = field(default_factory=K1VelCmdInitState)
    commands: Commands = field(
        default_factory=lambda: Commands(
            vel_limit=[
                [-1.0, -1.0, -1.0],
                [1.0, 1.0, 1.0],
            ],
            rel_standing_envs=0.2,
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
    control_config: ControlConfig = field(
        default_factory=lambda: ControlConfig(action_scale=0.35)
    )
    reset_base_qvel_limit: float = 0.1
    sim_dt: float = 0.002
    obs_frame_stack: int = 1
    add_body_sensors: bool = True
    reward_config: K1RewardConfig | None = None


class K1VelCmdCompatEnv(K1WalkEnv):
    """Joystick velocity tracking with K1SoccerDribbleCompat obs / action contract."""

    _cfg: K1VelCmdCompatCfg

    def __init__(
        self,
        cfg: K1VelCmdCompatCfg,
        num_envs: int = 1,
        backend_type: str = "mujoco",
    ) -> None:
        super().__init__(cfg, num_envs=num_envs, backend_type=backend_type)
        self._actor_obs_dim = k1_vel_cmd_compat_actor_obs_dim(self._num_action)
        self._critic_obs_dim = k1_vel_cmd_compat_critic_obs_dim(self._num_action)
        self._a2d_perm = np.asarray(K1_ACTUATOR_TO_DEPLOY_PERM, dtype=np.intp)
        self._d2a_perm = np.asarray(K1_DEPLOY_TO_ACTUATOR_PERM, dtype=np.intp)
        if int(self._cfg.obs_frame_stack) != 1:
            raise ValueError("K1VelCmdCompat requires obs_frame_stack=1")
        self._dr_manager = DomainRandomizationManager(
            self, K1VelCmdCompatDomainRandomizationProvider()
        )

    def _init_reward_functions(self) -> None:
        super()._init_reward_functions()
        self._reward_fns["tracking_lin_vel"] = self._reward_tracking_lin_vel_gated
        self._reward_fns["tracking_ang_vel"] = self._reward_tracking_ang_vel_gated
        self._reward_fns["cmd_lin_vel_error"] = self._reward_cmd_lin_vel_error
        self._reward_fns["cmd_ang_vel_error"] = self._reward_cmd_ang_vel_error
        self._reward_fns["cmd_speed_shortfall"] = self._reward_cmd_speed_shortfall
        self._reward_fns["stand_still_drift"] = self._reward_stand_still_drift
        self._reward_fns["stand_still_linvel"] = self._reward_stand_still_linvel
        self._reward_fns["stand_still_dof_vel"] = self._reward_stand_still_dof_vel
        self._reward_fns["stand_still_action"] = self._reward_stand_still_action
        self._reward_fns["stand_still"] = self._reward_stand_still_joints
        self._reward_fns["stand_double_stance"] = self._reward_stand_double_stance
        self._reward_fns["penalty_action_rate"] = self._reward_penalty_action_rate_stand_boosted

    def _cmd_gate_threshold(self) -> float:
        return float(getattr(self._reward_cfg, "stand_still_cmd_threshold", 0.1))

    def _stand_still_cmd_threshold(self) -> float:
        return self._cmd_gate_threshold()

    def _standing_mask(self, ctx: RewardContext) -> np.ndarray:
        return compute_standing_command_mask(ctx.info["commands"], self._cmd_gate_threshold())

    def _moving_mask(self, ctx: RewardContext) -> np.ndarray:
        return compute_command_locomotion_mask(ctx.info["commands"], self._cmd_gate_threshold())

    def _min_command_speed_for_gait(self) -> float:
        return self._cmd_gate_threshold()

    def _vel_cmd_gait_gate(self, ctx: RewardContext) -> np.ndarray:
        """Gait shaping follows command intent (do not wait for body speed to pick up)."""
        return self._moving_mask(ctx)

    def _reward_tracking_lin_vel_gated(self, ctx: RewardContext) -> np.ndarray:
        commands = ctx.info["commands"]
        lin_vel_error = np.sum(np.square(commands[:, :2] - ctx.linvel[:, :2]), axis=1)
        moving_sigma = float(getattr(self._reward_cfg, "moving_tracking_sigma", 0.10))
        stand_sigma = float(self._reward_cfg.tracking_sigma)
        moving = self._moving_mask(ctx) > 0.0
        standing = self._standing_mask(ctx) > 0.0
        moving_rew = np.exp(-lin_vel_error / moving_sigma)
        standing_rew = np.exp(-lin_vel_error / stand_sigma)
        return np.asarray(
            np.where(moving, moving_rew, np.where(standing, standing_rew, moving_rew)),
            dtype=get_global_dtype(),
        )

    def _reward_tracking_ang_vel_gated(self, ctx: RewardContext) -> np.ndarray:
        commands = ctx.info["commands"]
        ang_vel_error = np.square(commands[:, 2] - ctx.gyro[:, 2])
        moving_sigma = float(getattr(self._reward_cfg, "moving_tracking_sigma", 0.10))
        stand_sigma = float(self._reward_cfg.tracking_sigma)
        moving = self._moving_mask(ctx) > 0.0
        standing = self._standing_mask(ctx) > 0.0
        moving_rew = np.exp(-ang_vel_error / moving_sigma)
        standing_rew = np.exp(-ang_vel_error / stand_sigma)
        return np.asarray(
            np.where(moving, moving_rew, np.where(standing, standing_rew, moving_rew)),
            dtype=get_global_dtype(),
        )

    def _reward_cmd_lin_vel_error(self, ctx: RewardContext) -> np.ndarray:
        from unilab.envs.locomotion.common import rewards as locomotion_rewards

        return locomotion_rewards.cmd_lin_vel_error(ctx, cmd_threshold=self._cmd_gate_threshold())

    def _reward_cmd_ang_vel_error(self, ctx: RewardContext) -> np.ndarray:
        from unilab.envs.locomotion.common import rewards as locomotion_rewards

        return locomotion_rewards.cmd_ang_vel_error(ctx, cmd_threshold=self._cmd_gate_threshold())

    def _reward_cmd_speed_shortfall(self, ctx: RewardContext) -> np.ndarray:
        from unilab.envs.locomotion.common import rewards as locomotion_rewards

        return locomotion_rewards.cmd_speed_shortfall(ctx, cmd_threshold=self._cmd_gate_threshold())

    def _reward_stand_still_drift(self, ctx: RewardContext) -> np.ndarray:
        from unilab.envs.locomotion.common import rewards as locomotion_rewards

        return locomotion_rewards.stand_still_drift(
            ctx, cmd_threshold=self._stand_still_cmd_threshold()
        )

    def _reward_stand_still_linvel(self, ctx: RewardContext) -> np.ndarray:
        from unilab.envs.locomotion.common import rewards as locomotion_rewards

        sigma = float(getattr(self._reward_cfg, "stand_still_linvel_sigma", 0.02))
        return locomotion_rewards.stand_still_linvel_exp(
            ctx, cmd_threshold=self._stand_still_cmd_threshold(), sigma=sigma
        )

    def _reward_stand_still_dof_vel(self, ctx: RewardContext) -> np.ndarray:
        from unilab.envs.locomotion.common import rewards as locomotion_rewards

        return locomotion_rewards.stand_still_dof_vel(
            ctx, cmd_threshold=self._stand_still_cmd_threshold()
        )

    def _reward_stand_still_action(self, ctx: RewardContext) -> np.ndarray:
        from unilab.envs.locomotion.common import rewards as locomotion_rewards

        return locomotion_rewards.stand_still_action(
            ctx, cmd_threshold=self._stand_still_cmd_threshold()
        )

    def _reward_stand_still_joints(self, ctx: RewardContext) -> np.ndarray:
        from unilab.envs.locomotion.common import rewards as locomotion_rewards

        return locomotion_rewards.stand_still(
            ctx, command_threshold=self._stand_still_cmd_threshold()
        )

    def _reward_stand_double_stance(self, ctx: RewardContext) -> np.ndarray:
        standing = self._standing_mask(ctx)
        left_contact = compute_aggregated_foot_contact(self._backend, LEFT_FOOT_CONTACT_SENSORS)
        right_contact = compute_aggregated_foot_contact(self._backend, RIGHT_FOOT_CONTACT_SENSORS)
        double_stance = np.asarray(
            np.logical_and(left_contact, right_contact), dtype=get_global_dtype()
        )
        return np.asarray(standing * double_stance, dtype=get_global_dtype())

    def _reward_penalty_action_rate_stand_boosted(self, ctx: RewardContext) -> np.ndarray:
        from unilab.envs.locomotion.common import rewards as locomotion_rewards

        base = locomotion_rewards.action_rate(ctx)
        standing = self._standing_mask(ctx)
        boost = float(getattr(self._reward_cfg, "stand_still_action_rate_scale", 3.0))
        return np.asarray(np.where(standing, base * boost, base), dtype=get_global_dtype())

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
        return np.asarray(reward * self._vel_cmd_gait_gate(ctx), dtype=get_global_dtype())

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
        return np.asarray(reward * self._vel_cmd_gait_gate(ctx), dtype=get_global_dtype())

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
        return np.asarray(reward * self._vel_cmd_gait_gate(ctx), dtype=get_global_dtype())

    def _reward_feet_double_stance(self, ctx: RewardContext):
        commands = ctx.info.get("commands", np.zeros((self._num_envs, 3), dtype=get_global_dtype()))
        left_contact = compute_aggregated_foot_contact(self._backend, LEFT_FOOT_CONTACT_SENSORS)
        right_contact = compute_aggregated_foot_contact(self._backend, RIGHT_FOOT_CONTACT_SENSORS)
        double_stance = np.asarray(
            np.logical_and(left_contact, right_contact), dtype=get_global_dtype()
        )
        moving = compute_command_locomotion_mask(commands, self._min_command_speed_for_gait())
        return np.asarray(double_stance * moving, dtype=get_global_dtype())

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        return {"obs": self._actor_obs_dim, "critic": self._critic_obs_dim}

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
        command = info["commands"]
        last_actions = info.get(
            "current_actions",
            np.zeros((num_rows, self._num_action), dtype=get_global_dtype()),
        )

        diff_d = diff[:, self._a2d_perm]
        dof_vel_d = dof_vel[:, self._a2d_perm]
        last_actions_d = last_actions[:, self._a2d_perm]
        gravity_down = -np.asarray(gravity, dtype=get_global_dtype())

        actor = np.concatenate(
            [
                np.asarray(linvel, dtype=get_global_dtype()) * _LIN_VEL_SCALE,
                np.asarray(gyro, dtype=get_global_dtype()) * _ANG_VEL_SCALE,
                gravity_down * _GRAVITY_SCALE,
                np.asarray(command, dtype=get_global_dtype()) * _CMD_SCALE,
                np.asarray(diff_d, dtype=get_global_dtype()) * _JOINT_POS_SCALE,
                np.asarray(dof_vel_d, dtype=get_global_dtype()) * _JOINT_VEL_SCALE,
                np.asarray(last_actions_d, dtype=get_global_dtype()) * _LAST_ACTION_SCALE,
            ],
            axis=1,
        )

        critic = np.concatenate(
            [actor, np.asarray(linvel, dtype=get_global_dtype())],
            axis=1,
        )
        return {"obs": actor, "critic": critic}

    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> np.ndarray:
        state.info["last_actions"] = state.info.get(
            "current_actions",
            np.zeros_like(actions),
        )
        state.info["current_actions"] = actions

        commands = state.info.get(
            "commands", np.zeros((self._num_envs, 3), dtype=get_global_dtype())
        )
        moving = compute_command_locomotion_mask(commands, self._stand_still_cmd_threshold()) > 0.0
        gait_phase = state.info.get(
            "gait_phase",
            np.zeros((self._num_envs, 2), dtype=get_global_dtype()),
        )
        standing = ~moving
        if np.any(standing):
            # Lock swing targets to double-stance (both feet down) while cmd ≈ 0.
            gait_phase[standing, 0] = 0.0
            gait_phase[standing, 1] = math.pi
        if np.any(moving):
            gait_phase[moving, 0] = (
                gait_phase[moving, 0] + self._gait_phase_delta
            ) % (2.0 * math.pi)
            gait_phase[moving, 1] = (
                gait_phase[moving, 1] + self._gait_phase_delta
            ) % (2.0 * math.pi)
        state.info["gait_phase"] = gait_phase

        actions_actuator = actions[:, self._d2a_perm]
        return (
            np.asarray(actions_actuator, dtype=get_global_dtype())
            * self._cfg.control_config.action_scale
            + self.default_angles
        )


@registry.envcfg("K1VelCmdCompat")
@dataclass
class K1VelCmdCompatFlatCfg(K1VelCmdCompatCfg):
    reward_config: K1RewardConfig | None = None


registry.register_env("K1VelCmdCompat", K1VelCmdCompatEnv, sim_backend="mujoco")
registry.register_env("K1VelCmdCompat", K1VelCmdCompatEnv, sim_backend="motrix")

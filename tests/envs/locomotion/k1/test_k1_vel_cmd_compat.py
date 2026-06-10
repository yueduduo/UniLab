"""Contract tests for K1VelCmdCompat (46000-aligned 78-dim actor obs + joystick cmds)."""

from __future__ import annotations

import numpy as np
import pytest

from unilab.dtype_config import get_global_dtype
from unilab.base import registry
from unilab.envs.locomotion.g1.joystick import (
    compute_command_locomotion_mask,
    compute_forward_speed_gate,
    compute_planar_speed_gate,
)
from unilab.envs.locomotion.common.rewards import (
    RewardContext,
    cmd_lin_vel_error,
    stand_still_drift,
    stand_still_linvel_exp,
)
from unilab.envs.locomotion.k1.constants import (
    K1_NUM_ACTION,
    k1_soccer_dribble_compat_actor_obs_dim,
    k1_vel_cmd_compat_actor_obs_dim,
    k1_vel_cmd_compat_critic_obs_dim,
)
from unilab.envs.locomotion.k1.joystick import K1RewardConfig
from unilab.envs.locomotion.k1.k1_vel_cmd_compat import K1VelCmdCompatCfg, K1VelCmdCompatEnv
from unilab.base.registry import ensure_registries


def test_actor_dim_matches_deploy_contract():
    assert k1_vel_cmd_compat_actor_obs_dim(K1_NUM_ACTION) == 78
    assert k1_vel_cmd_compat_actor_obs_dim(K1_NUM_ACTION) == k1_soccer_dribble_compat_actor_obs_dim(
        K1_NUM_ACTION
    )


def test_critic_dim_is_actor_plus_linvel():
    assert k1_vel_cmd_compat_critic_obs_dim(K1_NUM_ACTION) == 81
    assert k1_vel_cmd_compat_critic_obs_dim(K1_NUM_ACTION) == (
        k1_vel_cmd_compat_actor_obs_dim(K1_NUM_ACTION) + 3
    )


def test_cfg_obs_frame_stack_is_1():
    assert K1VelCmdCompatCfg().obs_frame_stack == 1


def test_cfg_command_limits():
    cfg = K1VelCmdCompatCfg()
    assert cfg.commands.vel_limit[0] == [-1.0, -1.0, -1.0]
    assert cfg.commands.vel_limit[1] == [1.0, 1.0, 1.0]
    assert cfg.commands.rel_standing_envs == 0.2
    assert cfg.reset_base_qvel_limit == 0.1


def test_registry_contains_task():
    ensure_registries()
    assert registry.contains("K1VelCmdCompat")


def test_planar_speed_gate_includes_backward_motion():
    linvel = np.array([[-0.2, 0.0, 0.0], [0.02, 0.0, 0.0]], dtype=get_global_dtype())
    gate = compute_planar_speed_gate(linvel, 0.05)
    assert gate[0] == 1.0
    assert gate[1] == 0.0
    forward_only = compute_forward_speed_gate(linvel, 0.05)
    assert forward_only[0] == 0.0


def test_command_locomotion_mask_zero_for_stand():
    commands = np.array([[0.0, 0.0, 0.0], [-0.5, 0.0, 0.0]], dtype=get_global_dtype())
    mask = compute_command_locomotion_mask(commands, 0.12)
    assert mask[0] == 0.0
    assert mask[1] == 1.0


def test_cmd_lin_vel_error_only_when_commanded_to_move():
    ctx = RewardContext(
        info={"commands": np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=get_global_dtype())},
        linvel=np.array([[0.3, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=get_global_dtype()),
        gyro=np.zeros((2, 3), dtype=get_global_dtype()),
        dof_pos=np.zeros((2, 22), dtype=get_global_dtype()),
        num_envs=2,
    )
    err = cmd_lin_vel_error(ctx, cmd_threshold=0.1)
    assert err[0] == 0.0
    assert err[1] == pytest.approx(0.25)


def test_stand_still_linvel_rewards_only_when_cmd_zero_and_motionless():
    ctx = RewardContext(
        info={"commands": np.array([[0.0, 0.0, 0.0], [-0.5, 0.0, 0.0]], dtype=get_global_dtype())},
        linvel=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=get_global_dtype()),
        gyro=np.zeros((2, 3), dtype=get_global_dtype()),
        dof_pos=np.zeros((2, 22), dtype=get_global_dtype()),
        num_envs=2,
    )
    rew = stand_still_linvel_exp(ctx, cmd_threshold=0.08, sigma=0.015)
    assert rew[0] == pytest.approx(1.0)
    assert rew[1] == 0.0


def test_stand_still_drift_penalizes_motion_only_when_cmd_zero():
    ctx = RewardContext(
        info={"commands": np.array([[0.0, 0.0, 0.0], [-0.5, 0.0, 0.0]], dtype=get_global_dtype())},
        linvel=np.array([[0.3, 0.0, 0.0], [0.3, 0.0, 0.0]], dtype=get_global_dtype()),
        gyro=np.zeros((2, 3), dtype=get_global_dtype()),
        dof_pos=np.zeros((2, 22), dtype=get_global_dtype()),
        num_envs=2,
    )
    drift = stand_still_drift(ctx, cmd_threshold=0.08)
    assert drift[0] > 0.0
    assert drift[1] == 0.0


def test_k1_vel_cmd_compat_mujoco_reset_step():
    pytest.importorskip("mujoco")
    ensure_registries()
    reward_config = K1RewardConfig(
        scales={
            "tracking_lin_vel": 2.0,
            "tracking_ang_vel": 1.5,
            "alive": 1.0,
        },
        tracking_sigma=0.25,
        gait_frequency=1.5,
        feet_phase_swing_height=0.06,
        feet_phase_tracking_sigma=0.008,
        base_height_target=0.5743699226900935,
        min_base_height=0.35,
        max_tilt_deg=65.0,
    )
    env = registry.make(
        "K1VelCmdCompat",
        sim_backend="mujoco",
        num_envs=2,
        env_cfg_override={"reward_config": reward_config},
    )
    assert env.obs_groups_spec == {"obs": 78, "critic": 81}
    env.init_state()
    obs, _ = env.reset(np.arange(2, dtype=np.int32))
    assert obs["obs"].shape == (2, 78)
    assert obs["critic"].shape == (2, 81)
    # Standing command slice should be zero for some envs after reset sampling.
    cmds = env._state.info["commands"]
    assert cmds.shape == (2, 3)
    assert np.all(cmds >= -1.0) and np.all(cmds <= 1.0)
    state = env.step(np.zeros((2, K1_NUM_ACTION), dtype=np.float32))
    assert state.obs["obs"].shape == (2, 78)
    env.close()

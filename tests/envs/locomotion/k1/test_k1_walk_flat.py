"""K1WalkFlat env config and obs contract tests."""

from __future__ import annotations

import pytest
import numpy as np

from unilab.base.registry import ensure_registries
from unilab.envs.locomotion.k1.constants import (
    K1_NUM_ACTION,
    k1_walk_actor_obs_dim,
    k1_walk_critic_obs_dim,
)
from unilab.envs.locomotion.k1.joystick import K1WalkFlatCfg


def test_k1_walk_flat_obs_groups_spec_g1_walk_profile():
    actor_dim = k1_walk_actor_obs_dim(K1_NUM_ACTION)
    assert actor_dim == 6 + 3 * K1_NUM_ACTION + 5
    assert k1_walk_critic_obs_dim(K1_NUM_ACTION) == actor_dim + 3


def test_k1_walk_flat_registry():
    ensure_registries()
    from unilab.base import registry

    assert registry.contains("K1WalkFlat")


@pytest.mark.slow
def test_k1_walk_flat_penalty_curriculum_matches_g1_initial_scale():
    """When curriculum.enabled, K1 applies the same PenaltyCurriculum as G1WalkEnv."""
    pytest.importorskip("mujoco")
    from unilab.base import registry
    from unilab.envs.locomotion.g1.joystick import CurriculumConfig
    from unilab.envs.locomotion.k1.constants import K1_G1_ALIGNED_POSE_WEIGHTS
    from unilab.envs.locomotion.k1.joystick import K1RewardConfig

    ensure_registries()
    reward_config = K1RewardConfig(
        scales={
            "tracking_lin_vel": 1.0,
            "penalty_action_rate": -5.0,
            "pose": -0.5,
            "alive": 0.0,
        },
        tracking_sigma=0.25,
        gait_frequency=1.5,
        feet_phase_swing_height=0.09,
        feet_phase_tracking_sigma=0.008,
        base_height_target=0.55,
        min_base_height=0.35,
        max_tilt_deg=35.0,
        pose_weights=list(K1_G1_ALIGNED_POSE_WEIGHTS),
    )
    curriculum = CurriculumConfig(
        enabled=True,
        initial_scale=0.5,
        min_scale=0.5,
        max_scale=1.0,
    )
    env = registry.make(
        "K1WalkFlat",
        sim_backend="mujoco",
        num_envs=4,
        env_cfg_override={
            "reward_config": reward_config,
            "curriculum": curriculum,
        },
    )
    assert env._penalty_curriculum is not None
    assert env._reward_cfg.scales["penalty_action_rate"] == pytest.approx(-2.5)
    assert env._reward_cfg.scales["pose"] == pytest.approx(-0.25)
    env.close()


@pytest.mark.slow
def test_k1_walk_flat_mujoco_reset_step():
    pytest.importorskip("mujoco")
    from unilab.base import registry
    from unilab.envs.locomotion.k1.joystick import K1RewardConfig
    from unilab.envs.locomotion.k1.constants import K1_G1_ALIGNED_POSE_WEIGHTS

    ensure_registries()
    reward_config = K1RewardConfig(
        scales={"tracking_lin_vel": 1.0, "alive": 0.0},
        tracking_sigma=0.25,
        gait_frequency=1.5,
        feet_phase_swing_height=0.09,
        feet_phase_tracking_sigma=0.008,
        base_height_target=0.55,
        min_base_height=0.35,
        max_tilt_deg=35.0,
        pose_weights=list(K1_G1_ALIGNED_POSE_WEIGHTS),
    )
    env = registry.make(
        "K1WalkFlat",
        sim_backend="mujoco",
        num_envs=4,
        env_cfg_override={"reward_config": reward_config},
    )
    actor_dim = k1_walk_actor_obs_dim(env.action_space.shape[0])
    critic_dim = k1_walk_critic_obs_dim(env.action_space.shape[0])
    env.init_state()
    obs, info = env.reset(np.arange(4, dtype=np.int32))
    assert obs["obs"].shape == (4, actor_dim)
    assert obs["critic"].shape == (4, critic_dim)
    actions = np.zeros((4, env.action_space.shape[0]), dtype=np.float32)
    state = env.step(actions)
    assert state.obs["obs"].shape == (4, actor_dim)
    assert state.reward.shape == (4,)
    env.close()

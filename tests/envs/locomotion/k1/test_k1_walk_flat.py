"""K1WalkFlat env config and obs contract tests."""

from __future__ import annotations

import pytest
import numpy as np

from unilab.base.registry import ensure_registries
from unilab.envs.locomotion.k1.constants import K1_OBS_STACKED_DIM
from unilab.envs.locomotion.k1.joystick import K1WalkFlatCfg, K1WalkEnv


def test_k1_walk_flat_obs_groups_spec():
    cfg = K1WalkFlatCfg()
    assert cfg.obs_frame_stack == 5
    env_cls = K1WalkEnv
    spec = env_cls.obs_groups_spec.fget(env_cls)  # type: ignore[arg-type]
    assert spec["obs"] == K1_OBS_STACKED_DIM
    assert spec["critic"] == K1_OBS_STACKED_DIM + 3


def test_k1_walk_flat_registry():
    ensure_registries()
    from unilab.base import registry

    assert registry.contains("K1WalkFlat")


@pytest.mark.slow
def test_k1_walk_flat_mujoco_reset_step():
    pytest.importorskip("mujoco")
    from unilab.base import registry
    from unilab.envs.locomotion.k1.joystick import K1RewardConfig

    ensure_registries()
    reward_config = K1RewardConfig(
        scales={"tracking_lin_vel": 1.0, "alive": 0.0},
        tracking_sigma=0.25,
        gait_frequency=1.5,
        feet_phase_swing_height=0.06,
        feet_phase_tracking_sigma=0.008,
        base_height_target=0.55,
        min_base_height=0.35,
        max_tilt_deg=25.0,
    )
    env = registry.make(
        "K1WalkFlat",
        sim_backend="mujoco",
        num_envs=4,
        env_cfg_override={"reward_config": reward_config},
    )
    env.init_state()
    obs, info = env.reset(np.arange(4, dtype=np.int32))
    assert obs["obs"].shape == (4, K1_OBS_STACKED_DIM)
    assert obs["critic"].shape == (4, K1_OBS_STACKED_DIM + 3)
    actions = np.zeros((4, env.action_space.shape[0]), dtype=np.float32)
    state = env.step(actions)
    assert state.obs["obs"].shape == (4, K1_OBS_STACKED_DIM)
    assert state.reward.shape == (4,)
    env.close()

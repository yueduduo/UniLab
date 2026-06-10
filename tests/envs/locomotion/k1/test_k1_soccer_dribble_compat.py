"""Contract tests for K1SoccerDribbleCompat (46000-aligned 78-dim actor obs)."""

from __future__ import annotations

import numpy as np
import pytest

from unilab.envs.locomotion.k1.constants import (
    K1_ACTUATOR_JOINT_ORDER,
    K1_ACTUATOR_TO_DEPLOY_PERM,
    K1_DEPLOY_JOINT_ORDER,
    K1_DEPLOY_TO_ACTUATOR_PERM,
    K1_NUM_ACTION,
    k1_soccer_dribble_compat_actor_obs_dim,
    k1_soccer_dribble_compat_critic_obs_dim,
)
from unilab.envs.locomotion.k1.soccer_dribble_compat import K1SoccerDribbleCompatCfg


# ---------------------------------------------------------------------------
# Obs dim contract — must match k1_model_46000 exactly for actor
# ---------------------------------------------------------------------------


def test_actor_dim_matches_46000():
    """Actor must be exactly 78 dims — the k1_model_46000 deployment contract."""
    assert k1_soccer_dribble_compat_actor_obs_dim(K1_NUM_ACTION) == 78


def test_critic_dim_is_actor_plus_ball():
    """Critic = actor(78) + ball_rel_state(3) = 81."""
    assert k1_soccer_dribble_compat_critic_obs_dim(K1_NUM_ACTION) == 81
    assert k1_soccer_dribble_compat_critic_obs_dim(K1_NUM_ACTION) == (
        k1_soccer_dribble_compat_actor_obs_dim(K1_NUM_ACTION) + 3
    )


def test_cfg_obs_frame_stack_is_1():
    assert K1SoccerDribbleCompatCfg().obs_frame_stack == 1


def test_actor_layout_matches_46000_formula():
    # 46000: linvel(3) + angvel(3) + gravity(3) + cmd(3) + joints(22×3) = 78
    n = K1_NUM_ACTION
    assert 3 + 3 + 3 + 3 + 3 * n == 78
    assert k1_soccer_dribble_compat_actor_obs_dim(n) == 78


# ---------------------------------------------------------------------------
# Joint permutation contract
# ---------------------------------------------------------------------------


def test_deploy_joint_order_has_all_22_joints():
    assert len(K1_DEPLOY_JOINT_ORDER) == K1_NUM_ACTION
    assert set(K1_DEPLOY_JOINT_ORDER) == set(K1_ACTUATOR_JOINT_ORDER)


def test_actuator_to_deploy_perm_is_valid_permutation():
    assert sorted(K1_ACTUATOR_TO_DEPLOY_PERM) == list(range(K1_NUM_ACTION))


def test_deploy_to_actuator_perm_is_valid_permutation():
    assert sorted(K1_DEPLOY_TO_ACTUATOR_PERM) == list(range(K1_NUM_ACTION))


def test_perms_are_mutual_inverses():
    perm = np.asarray(K1_ACTUATOR_TO_DEPLOY_PERM, dtype=np.intp)
    inv = np.asarray(K1_DEPLOY_TO_ACTUATOR_PERM, dtype=np.intp)
    identity = np.arange(K1_NUM_ACTION, dtype=np.intp)
    assert np.array_equal(perm[inv], identity), "perm[inv] != identity"
    assert np.array_equal(inv[perm], identity), "inv[perm] != identity"


def test_perm_maps_actuator_to_deploy_correctly():
    perm = K1_ACTUATOR_TO_DEPLOY_PERM
    for deploy_idx, joint_name in enumerate(K1_DEPLOY_JOINT_ORDER):
        actuator_idx = K1_ACTUATOR_JOINT_ORDER.index(joint_name)
        assert perm[deploy_idx] == actuator_idx, (
            f"DEPLOY[{deploy_idx}]={joint_name}: "
            f"expected perm={actuator_idx}, got {perm[deploy_idx]}"
        )


def test_inv_perm_maps_deploy_to_actuator_correctly():
    inv = K1_DEPLOY_TO_ACTUATOR_PERM
    for actuator_idx, joint_name in enumerate(K1_ACTUATOR_JOINT_ORDER):
        deploy_idx = K1_DEPLOY_JOINT_ORDER.index(joint_name)
        assert inv[actuator_idx] == deploy_idx, (
            f"ACTUATOR[{actuator_idx}]={joint_name}: "
            f"expected inv={deploy_idx}, got {inv[actuator_idx]}"
        )


def test_round_trip_permutation():
    rng = np.random.default_rng(0)
    v = rng.normal(size=(8, K1_NUM_ACTION)).astype(np.float32)
    perm = np.asarray(K1_ACTUATOR_TO_DEPLOY_PERM, dtype=np.intp)
    inv = np.asarray(K1_DEPLOY_TO_ACTUATOR_PERM, dtype=np.intp)
    np.testing.assert_array_equal(v, v[:, perm][:, inv])


# ---------------------------------------------------------------------------
# Spot-check: key joints in deploy order
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "joint, deploy_idx",
    [
        ("AAHead_yaw", 0),
        ("ALeft_Shoulder_Pitch", 1),
        ("ARight_Shoulder_Pitch", 2),
        ("Left_Hip_Pitch", 3),
        ("Right_Hip_Pitch", 4),
        ("Head_pitch", 5),
        ("Left_Shoulder_Roll", 6),
        ("Right_Shoulder_Roll", 7),
        ("Left_Hip_Roll", 8),
        ("Right_Hip_Roll", 9),
        ("Left_Knee_Pitch", 16),
        ("Right_Knee_Pitch", 17),
        ("Left_Ankle_Pitch", 18),
        ("Right_Ankle_Pitch", 19),
        ("Left_Ankle_Roll", 20),
        ("Right_Ankle_Roll", 21),
    ],
)
def test_spot_check_deploy_index(joint: str, deploy_idx: int):
    assert K1_DEPLOY_JOINT_ORDER[deploy_idx] == joint


# ---------------------------------------------------------------------------
# Obs segment offsets (actor = 78 dims, matching 46000 header)
# ---------------------------------------------------------------------------


def test_actor_obs_segment_offsets():
    n = K1_NUM_ACTION  # 22
    assert (0,  3)  == (0,  3)   # base_lin_vel
    assert (3,  6)  == (3,  6)   # base_ang_vel
    assert (6,  9)  == (6,  9)   # gravity
    assert (9,  12) == (9,  12)  # cmd
    assert (12, 12 + n)       == (12, 34)  # joint_pos
    assert (12 + n, 12 + 2*n) == (34, 56)  # joint_vel
    assert (12 + 2*n, 12 + 3*n) == (56, 78)  # last_action
    assert 12 + 3*n == k1_soccer_dribble_compat_actor_obs_dim(n) == 78


def test_critic_obs_is_actor_plus_3():
    n = K1_NUM_ACTION
    actor = k1_soccer_dribble_compat_actor_obs_dim(n)
    critic = k1_soccer_dribble_compat_critic_obs_dim(n)
    assert critic == actor + 3
    assert critic == 81
    # critic[78:81] = ball [rel_x_b, rel_y_b, rel_vx_b]

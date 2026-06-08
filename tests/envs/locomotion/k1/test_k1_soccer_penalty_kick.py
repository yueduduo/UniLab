"""K1SoccerPenaltyKick task contract tests."""

from __future__ import annotations

import pytest

from unilab.envs.locomotion.common.commands import Commands
from unilab.envs.locomotion.k1.constants import K1_NUM_ACTION, k1_keyframe_robot_joint_qpos
from unilab.envs.locomotion.k1.soccer_penalty_constants import (
    K1_PENALTY_GOAL_HEIGHT_M,
    K1_PENALTY_GOAL_LINE_X,
    K1_PENALTY_GOAL_WIDTH_M,
    K1_PENALTY_KICK_PHASE_MAX_STEPS,
    K1_PENALTY_KICK_REWARD_KEYS,
    K1_PENALTY_MARKINGS_FIELD_LENGTH_M,
    K1_PENALTY_MARKINGS_FIELD_WIDTH_M,
    K1_PENALTY_MAX_STEPS,
    K1_PENALTY_PITCH_LENGTH_M,
    K1_PENALTY_PITCH_WIDTH_M,
    K1_PENALTY_RESET_Y_OFFSET_M,
    K1_PENALTY_RIGHT_FOOT_STRIKE_DIST_M,
    K1_PENALTY_RIGHT_FOOT_Y_OFFSET_FROM_BASE_M,
    K1_PENALTY_ROBOT_START_XY,
    K1_PENALTY_RUNUP_BALL_DIST_M,
    K1_PENALTY_RUNUP_MAX_STEPS,
    K1_PENALTY_SPOT_DISTANCE_M,
    K1_PENALTY_SPOT_XY,
    k1_soccer_penalty_actor_obs_dim,
    k1_soccer_penalty_critic_obs_dim,
)
from unilab.envs.locomotion.k1.soccer_penalty_kick import (
    K1PenaltyKickDomainRandomizationProvider,
    K1SoccerPenaltyKickCfg,
)


def test_k1_soccer_penalty_obs_dims():
    assert k1_soccer_penalty_actor_obs_dim(K1_NUM_ACTION) == 80
    assert k1_soccer_penalty_critic_obs_dim(K1_NUM_ACTION) == 83
    assert K1SoccerPenaltyKickCfg().obs_frame_stack == 1


def test_penalty_layout_matches_sim_soccer():
    assert K1_PENALTY_PITCH_LENGTH_M == pytest.approx(12.0)
    assert K1_PENALTY_PITCH_WIDTH_M == pytest.approx(9.0)
    assert K1_PENALTY_MARKINGS_FIELD_LENGTH_M == pytest.approx(9.0)
    assert K1_PENALTY_MARKINGS_FIELD_WIDTH_M == pytest.approx(6.0)
    assert K1_PENALTY_GOAL_LINE_X == pytest.approx(4.5)
    assert K1_PENALTY_GOAL_WIDTH_M == pytest.approx(1.9)
    assert K1_PENALTY_GOAL_HEIGHT_M == pytest.approx(1.8)
    assert K1_PENALTY_SPOT_DISTANCE_M == pytest.approx(1.5)
    assert K1_PENALTY_SPOT_XY == pytest.approx((3.0, 0.0))
    assert K1_PENALTY_RIGHT_FOOT_Y_OFFSET_FROM_BASE_M == pytest.approx(0.0962)
    assert K1_PENALTY_ROBOT_START_XY == pytest.approx((1.0, 0.0962))
    assert K1_PENALTY_RESET_Y_OFFSET_M == pytest.approx(0.15)
    assert K1_PENALTY_RUNUP_BALL_DIST_M == pytest.approx(0.35)
    assert K1_PENALTY_RIGHT_FOOT_STRIKE_DIST_M == pytest.approx(0.22)
    assert K1_PENALTY_RUNUP_MAX_STEPS == 180
    assert K1_PENALTY_KICK_PHASE_MAX_STEPS == 120
    assert K1_PENALTY_MAX_STEPS == 400


def test_runup_reach_dual_sigma_has_gradient_at_runup_distance():
    import numpy as np

    from unilab.envs.locomotion.k1.soccer_penalty_kick import (
        K1SoccerPenaltyKickEnv,
        K1SoccerPenaltyKickRewardConfig,
    )

    cfg = K1SoccerPenaltyKickRewardConfig(
        scales={"runup_reach": 1.0},
        tracking_sigma=0.25,
        gait_frequency=1.5,
        feet_phase_swing_height=0.06,
        feet_phase_tracking_sigma=0.008,
        base_height_target=0.55,
        min_base_height=0.25,
        max_tilt_deg=65.0,
        runup_reach_sigma=0.25,
        runup_reach_sigma_2=2.0,
    )
    env = object.__new__(K1SoccerPenaltyKickEnv)
    env._reward_cfg = cfg

    raw_start = float(env._runup_reach_raw_from_distance(np.array([2.0]))[0])
    raw_mid = float(env._runup_reach_raw_from_distance(np.array([1.0]))[0])
    raw_near = float(env._runup_reach_raw_from_distance(np.array([0.3]))[0])
    assert raw_start > 0.10
    assert raw_mid > raw_start
    assert raw_near > raw_mid
    assert raw_near <= 2.0


def test_runup_success_is_dense_after_reach():
    import numpy as np

    from unilab.envs.locomotion.common.rewards import RewardContext
    from unilab.envs.locomotion.k1.soccer_penalty_kick import (
        K1SoccerPenaltyKickEnv,
        K1SoccerPenaltyKickRewardConfig,
    )

    cfg = K1SoccerPenaltyKickRewardConfig(
        scales={"runup_success": 50.0},
        tracking_sigma=0.25,
        gait_frequency=1.5,
        feet_phase_swing_height=0.06,
        feet_phase_tracking_sigma=0.008,
        base_height_target=0.55,
        min_base_height=0.25,
        max_tilt_deg=65.0,
    )
    env = object.__new__(K1SoccerPenaltyKickEnv)
    env._reward_cfg = cfg
    env._runup_reached = np.array([False, True], dtype=bool)

    ctx = RewardContext(
        info={"kick_phase": np.array([0.0, 1.0], dtype=np.float32)},
        linvel=np.zeros((2, 3), dtype=np.float32),
        gyro=np.zeros((2, 3), dtype=np.float32),
        dof_pos=np.zeros((2, 1), dtype=np.float32),
        num_envs=2,
        default_angles=np.zeros(1, dtype=np.float32),
        tracking_sigma=0.25,
        base_height_target=0.55,
        base_height=np.ones(2, dtype=np.float32),
        gravity=np.zeros((2, 3), dtype=np.float32),
        dof_vel=np.zeros((2, 1), dtype=np.float32),
        pose_weights=np.ones(1, dtype=np.float32),
    )
    reward = env._reward_runup_success(ctx)
    assert reward[0] == pytest.approx(0.0)
    assert reward[1] == pytest.approx(1.0)


def test_runup_speed_curriculum_levels_and_commands():
    from unilab.envs.locomotion.k1.soccer_penalty_kick import (
        K1SoccerPenaltyKickEnv,
        K1SoccerPenaltyKickRewardConfig,
    )

    cfg = K1SoccerPenaltyKickRewardConfig(
        scales={"runup_speed": 1.0},
        tracking_sigma=0.25,
        gait_frequency=1.5,
        feet_phase_swing_height=0.06,
        feet_phase_tracking_sigma=0.008,
        base_height_target=0.55,
        min_base_height=0.25,
        max_tilt_deg=65.0,
        runup_min_forward_speed=0.40,
        runup_speed_curriculum_initial=0.40,
        runup_speed_curriculum_increment=0.10,
        runup_speed_curriculum_interval_iterations=1000,
        runup_speed_curriculum_max_speed=0.95,
    )
    env = object.__new__(K1SoccerPenaltyKickEnv)
    env._reward_cfg = cfg
    env._cfg = type("Cfg", (), {"commands": Commands()})()
    env._training_iteration = 0

    env.set_training_iteration(1)
    assert env._runup_speed_curriculum_level() == pytest.approx(0.40)
    assert env._cfg.commands.vel_limit == [[0.4, 0.0, 0.0], [0.95, 0.0, 0.0]]

    env.set_training_iteration(1000)
    assert env._runup_speed_curriculum_level() == pytest.approx(0.40)

    env.set_training_iteration(1001)
    assert env._runup_speed_curriculum_level() == pytest.approx(0.50)
    assert env._cfg.commands.vel_limit == [[0.5, 0.0, 0.0], [0.95, 0.0, 0.0]]

    env.set_training_iteration(6001)
    assert env._runup_speed_curriculum_level() == pytest.approx(0.95)
    assert env._cfg.commands.vel_limit == [[0.95, 0.0, 0.0], [0.95, 0.0, 0.0]]


def test_runup_speed_success_curriculum_advances_on_success_rate():
    import numpy as np

    from unilab.envs.locomotion.k1.soccer_penalty_kick import (
        K1SoccerPenaltyKickEnv,
        K1SoccerPenaltyKickRewardConfig,
    )

    cfg = K1SoccerPenaltyKickRewardConfig(
        scales={"runup_speed": 1.0},
        tracking_sigma=0.25,
        gait_frequency=1.5,
        feet_phase_swing_height=0.06,
        feet_phase_tracking_sigma=0.008,
        base_height_target=0.55,
        min_base_height=0.25,
        max_tilt_deg=65.0,
        runup_speed_curriculum_initial=0.40,
        runup_speed_curriculum_increment=0.10,
        runup_speed_curriculum_max_speed=0.60,
        runup_speed_success_curriculum_enabled=True,
        runup_speed_success_threshold=0.05,
        runup_speed_success_promotion_updates=2,
        runup_success_rate_ema_alpha=1.0,
    )
    env = object.__new__(K1SoccerPenaltyKickEnv)
    env._reward_cfg = cfg
    env._cfg = type("Cfg", (), {"commands": Commands()})()
    env._runup_reached = np.array([True, False, False, False], dtype=bool)
    env._runup_success_rate_ema = 0.0
    env._runup_success_rate_initialized = False
    env._runup_speed_success_level_idx = 0
    env._runup_speed_success_streak = 0
    env._runup_start_distance_phase_idx = 0
    env._runup_start_distance_success_streak = 0

    env._apply_runup_speed_curriculum(env._runup_speed_curriculum_level())
    assert env._runup_speed_curriculum_level() == pytest.approx(0.40)

    done = np.arange(4, dtype=np.int32)
    env._update_success_curricula(done)
    assert env._runup_speed_curriculum_level() == pytest.approx(0.40)
    env._update_success_curricula(done)
    assert env._runup_speed_curriculum_level() == pytest.approx(0.50)
    assert env._cfg.commands.vel_limit == [[0.5, 0.0, 0.0], [0.6, 0.0, 0.0]]


def test_runup_start_distance_curriculum_samples_near_ball_first_phase():
    import numpy as np

    from unilab.envs.locomotion.k1.soccer_penalty_kick import (
        K1SoccerPenaltyKickEnv,
        K1SoccerPenaltyKickRewardConfig,
    )

    cfg = K1SoccerPenaltyKickRewardConfig(
        scales={},
        tracking_sigma=0.25,
        gait_frequency=1.5,
        feet_phase_swing_height=0.06,
        feet_phase_tracking_sigma=0.008,
        base_height_target=0.55,
        min_base_height=0.25,
        max_tilt_deg=65.0,
        runup_start_distance_curriculum_enabled=True,
        runup_start_distance_phase1=[0.20, 0.35],
        runup_start_distance_phase2=[0.35, 0.80],
        runup_start_distance_phase3=[2.0, 2.0],
    )
    env = object.__new__(K1SoccerPenaltyKickEnv)
    env._reward_cfg = cfg
    env._runup_start_distance_phase_idx = 0
    env._init_qpos = np.array(
        [K1_PENALTY_ROBOT_START_XY[0], K1_PENALTY_ROBOT_START_XY[1], 0.55],
        dtype=np.float64,
    )
    provider = K1PenaltyKickDomainRandomizationProvider()

    offset = provider._sample_reset_xy_offset(env, 128)
    reset_x = env._init_qpos[0] + offset[:, 0]
    distance = float(K1_PENALTY_SPOT_XY[0]) - reset_x
    assert np.min(distance) >= 0.20
    assert np.max(distance) <= 0.35


def test_runup_speed_rewards_faster_than_minimum():
    import numpy as np

    from unilab.envs.locomotion.common.rewards import RewardContext
    from unilab.envs.locomotion.k1.soccer_penalty_kick import (
        K1SoccerPenaltyKickEnv,
        K1SoccerPenaltyKickRewardConfig,
    )

    cfg = K1SoccerPenaltyKickRewardConfig(
        scales={"runup_speed": 1.0},
        tracking_sigma=0.25,
        gait_frequency=1.5,
        feet_phase_swing_height=0.06,
        feet_phase_tracking_sigma=0.008,
        base_height_target=0.55,
        min_base_height=0.25,
        max_tilt_deg=65.0,
        runup_speed_curriculum_initial=0.40,
        runup_speed_curriculum_max_speed=0.95,
    )
    env = object.__new__(K1SoccerPenaltyKickEnv)
    env._reward_cfg = cfg
    env._training_iteration = 1

    ctx = RewardContext(
        info={"kick_phase": np.array([0.0, 0.0, 0.0], dtype=np.float32)},
        linvel=np.array([[0.20, 0.0, 0.0], [0.40, 0.0, 0.0], [0.80, 0.0, 0.0]], dtype=np.float32),
        gyro=np.zeros((3, 3), dtype=np.float32),
        dof_pos=np.zeros((3, 1), dtype=np.float32),
        num_envs=3,
        default_angles=np.zeros(1, dtype=np.float32),
        tracking_sigma=0.25,
        base_height_target=0.55,
        base_height=np.ones(3, dtype=np.float32),
        gravity=np.zeros((3, 3), dtype=np.float32),
        dof_vel=np.zeros((3, 1), dtype=np.float32),
        pose_weights=np.ones(1, dtype=np.float32),
    )
    reward = env._reward_runup_speed(ctx)
    assert reward[0] == pytest.approx(0.5)
    assert reward[1] == pytest.approx(1.0)
    assert reward[2] == pytest.approx(2.0)


def test_runup_tracking_and_under_speed_use_minimum_not_sampled_cmd():
    import numpy as np

    from unilab.envs.locomotion.common.rewards import RewardContext
    from unilab.envs.locomotion.k1.soccer_penalty_kick import (
        K1SoccerPenaltyKickEnv,
        K1SoccerPenaltyKickRewardConfig,
    )

    cfg = K1SoccerPenaltyKickRewardConfig(
        scales={"tracking_lin_vel": 1.0, "under_speed": -1.0},
        tracking_sigma=0.25,
        gait_frequency=1.5,
        feet_phase_swing_height=0.06,
        feet_phase_tracking_sigma=0.008,
        base_height_target=0.55,
        min_base_height=0.25,
        max_tilt_deg=65.0,
        runup_speed_curriculum_initial=0.40,
        runup_speed_curriculum_max_speed=0.95,
    )
    env = object.__new__(K1SoccerPenaltyKickEnv)
    env._reward_cfg = cfg
    env._training_iteration = 1

    ctx = RewardContext(
        info={
            "kick_phase": np.array([0.0, 0.0], dtype=np.float32),
            "commands": np.array([[0.95, 0.0, 0.0], [0.95, 0.0, 0.0]], dtype=np.float32),
        },
        linvel=np.array([[0.80, 0.0, 0.0], [0.20, 0.0, 0.0]], dtype=np.float32),
        gyro=np.zeros((2, 3), dtype=np.float32),
        dof_pos=np.zeros((2, 1), dtype=np.float32),
        num_envs=2,
        default_angles=np.zeros(1, dtype=np.float32),
        tracking_sigma=0.25,
        base_height_target=0.55,
        base_height=np.ones(2, dtype=np.float32),
        gravity=np.zeros((2, 3), dtype=np.float32),
        dof_vel=np.zeros((2, 1), dtype=np.float32),
        pose_weights=np.ones(1, dtype=np.float32),
    )
    tracking = env._reward_tracking_lin_vel(ctx)
    under = env._reward_under_speed(ctx)
    assert tracking[0] == pytest.approx(1.0)
    assert tracking[0] > tracking[1]
    assert under[0] == pytest.approx(0.0)
    assert under[1] == pytest.approx(0.5)


def test_runup_timeout_disabled_before_training_iteration_threshold():
    from unilab.envs.locomotion.k1.soccer_penalty_kick import (
        K1SoccerPenaltyKickEnv,
        K1SoccerPenaltyKickRewardConfig,
    )

    cfg = K1SoccerPenaltyKickRewardConfig(
        scales={"runup_reach": 1.0},
        tracking_sigma=0.25,
        gait_frequency=1.5,
        feet_phase_swing_height=0.06,
        feet_phase_tracking_sigma=0.008,
        base_height_target=0.55,
        min_base_height=0.25,
        max_tilt_deg=65.0,
        runup_timeout_enable_after_iteration=1,
    )
    env = object.__new__(K1SoccerPenaltyKickEnv)
    env._reward_cfg = cfg
    env.set_training_iteration(0)
    assert env._runup_timeout_enabled() is False
    env.set_training_iteration(1)
    assert env._runup_timeout_enabled() is True


def test_right_foot_ball_dual_sigma_has_gradient_at_runup_distance():
    import numpy as np

    from unilab.envs.locomotion.k1.soccer_penalty_kick import (
        K1SoccerPenaltyKickEnv,
        K1SoccerPenaltyKickRewardConfig,
    )

    cfg = K1SoccerPenaltyKickRewardConfig(
        scales={"right_foot_to_ball": 1.0},
        tracking_sigma=0.25,
        gait_frequency=1.5,
        feet_phase_swing_height=0.06,
        feet_phase_tracking_sigma=0.008,
        base_height_target=0.55,
        min_base_height=0.25,
        max_tilt_deg=65.0,
        right_foot_ball_sigma=0.30,
        right_foot_ball_sigma_2=2.0,
    )
    env = object.__new__(K1SoccerPenaltyKickEnv)
    env._reward_cfg = cfg

    raw_far = float(env._right_foot_ball_raw_from_distance(np.array([1.5]))[0])
    raw_mid = float(env._right_foot_ball_raw_from_distance(np.array([0.8]))[0])
    raw_near = float(env._right_foot_ball_raw_from_distance(np.array([0.3]))[0])
    assert raw_far > 0.10
    assert raw_mid > raw_far
    assert raw_near > raw_mid
    assert raw_near <= 2.0


def test_right_foot_to_ball_reward_stops_after_kick_phase():
    import numpy as np

    from unilab.envs.locomotion.common.rewards import RewardContext
    from unilab.envs.locomotion.k1.soccer_penalty_kick import (
        K1SoccerPenaltyKickEnv,
        K1SoccerPenaltyKickRewardConfig,
    )

    cfg = K1SoccerPenaltyKickRewardConfig(
        scales={"right_foot_to_ball": 1.0},
        tracking_sigma=0.25,
        gait_frequency=1.5,
        feet_phase_swing_height=0.06,
        feet_phase_tracking_sigma=0.008,
        base_height_target=0.55,
        min_base_height=0.25,
        max_tilt_deg=65.0,
        right_foot_ball_sigma=0.30,
        right_foot_ball_sigma_2=2.0,
    )
    env = object.__new__(K1SoccerPenaltyKickEnv)
    env._reward_cfg = cfg

    ctx = RewardContext(
        info={
            "right_foot_ball_dist_xy": np.array([0.20, 0.20], dtype=np.float32),
            "kick_phase": np.array([0.0, 1.0], dtype=np.float32),
        },
        linvel=np.zeros((2, 3), dtype=np.float32),
        gyro=np.zeros((2, 3), dtype=np.float32),
        dof_pos=np.zeros((2, 1), dtype=np.float32),
        num_envs=2,
        default_angles=np.zeros(1, dtype=np.float32),
        tracking_sigma=0.25,
        base_height_target=0.55,
        base_height=np.ones(2, dtype=np.float32),
        gravity=np.zeros((2, 3), dtype=np.float32),
        dof_vel=np.zeros((2, 1), dtype=np.float32),
        pose_weights=np.ones(1, dtype=np.float32),
    )

    reward = env._reward_right_foot_to_ball(ctx)
    assert reward[0] > 0.0
    assert reward[1] == pytest.approx(0.0)


def test_right_foot_to_ball_progress_reward_uses_distance_delta_rate():
    import numpy as np

    from unilab.envs.locomotion.common.rewards import RewardContext
    from unilab.envs.locomotion.k1.soccer_penalty_kick import (
        K1SoccerPenaltyKickEnv,
        K1SoccerPenaltyKickRewardConfig,
    )

    cfg = K1SoccerPenaltyKickRewardConfig(
        scales={"right_foot_to_ball_progress": 1.0},
        tracking_sigma=0.25,
        gait_frequency=1.5,
        feet_phase_swing_height=0.06,
        feet_phase_tracking_sigma=0.008,
        base_height_target=0.55,
        min_base_height=0.25,
        max_tilt_deg=65.0,
        right_foot_ball_progress_clip=1.0,
    )
    env = object.__new__(K1SoccerPenaltyKickEnv)
    env._reward_cfg = cfg
    env._cfg = type("Cfg", (), {"ctrl_dt": 0.02})()
    ctx = RewardContext(
        info={
            "right_foot_ball_progress": np.array([0.01, 0.04, 0.01], dtype=np.float32),
            "kick_phase": np.array([0.0, 0.0, 1.0], dtype=np.float32),
        },
        linvel=np.zeros((3, 3), dtype=np.float32),
        gyro=np.zeros((3, 3), dtype=np.float32),
        dof_pos=np.zeros((3, 1), dtype=np.float32),
        num_envs=3,
        default_angles=np.zeros(1, dtype=np.float32),
        tracking_sigma=0.25,
        base_height_target=0.55,
        base_height=np.ones(3, dtype=np.float32),
        gravity=np.zeros((3, 3), dtype=np.float32),
        dof_vel=np.zeros((3, 1), dtype=np.float32),
        pose_weights=np.ones(1, dtype=np.float32),
    )

    reward = env._reward_right_foot_to_ball_progress(ctx)
    assert reward[0] == pytest.approx(0.5)
    assert reward[1] == pytest.approx(1.0)
    assert reward[2] == pytest.approx(0.0)


def test_penalty_kick_reward_keys_only_apply_in_kick_phase():
    assert "runup_reach" not in K1_PENALTY_KICK_REWARD_KEYS
    assert "right_foot_to_ball" not in K1_PENALTY_KICK_REWARD_KEYS
    assert "right_foot_to_ball_progress" not in K1_PENALTY_KICK_REWARD_KEYS
    assert "ball_to_goal" in K1_PENALTY_KICK_REWARD_KEYS
    assert "kick_stability" in K1_PENALTY_KICK_REWARD_KEYS
    assert "goal_scored" not in K1_PENALTY_KICK_REWARD_KEYS
    assert "termination_bad" not in K1_PENALTY_KICK_REWARD_KEYS


def test_penalty_default_angles_use_robot_keyframe_not_ball_tail():
    import numpy as np

    qpos = np.zeros(7 + K1_NUM_ACTION + 7, dtype=np.float32)
    qpos[7 : 7 + K1_NUM_ACTION] = np.linspace(0.1, 0.5, K1_NUM_ACTION, dtype=np.float32)
    qpos[-7:] = np.array([3.0, 0.0, 0.09, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    robot = k1_keyframe_robot_joint_qpos(qpos, K1_NUM_ACTION)
    assert robot.shape == (K1_NUM_ACTION,)
    assert not np.any(np.isclose(robot, qpos[-K1_NUM_ACTION :]))


def test_ball_in_goal_matches_sim_soccer_referee():
    import numpy as np

    from unilab.envs.locomotion.k1.soccer_penalty_kick import K1SoccerPenaltyKickEnv

    env = K1SoccerPenaltyKickEnv.__new__(K1SoccerPenaltyKickEnv)
    ball = np.array(
        [
            [4.6, 0.0, 0.5],
            [4.4, 0.0, 0.5],
            [4.6, 1.0, 0.5],
            [4.6, 0.0, 2.0],
        ],
        dtype=np.float32,
    )
    scored = env._ball_in_goal(ball)
    np.testing.assert_array_equal(scored, [True, False, False, False])


def test_penalty_ball_failure_helpers():
    import numpy as np

    from unilab.envs.locomotion.k1.soccer_penalty_kick import (
        K1SoccerPenaltyKickEnv,
        K1SoccerPenaltyKickRewardConfig,
    )

    env = K1SoccerPenaltyKickEnv.__new__(K1SoccerPenaltyKickEnv)
    env._reward_cfg = K1SoccerPenaltyKickRewardConfig(  # pyright: ignore[reportPrivateUsage]
        scales={},
        tracking_sigma=0.25,
        gait_frequency=1.5,
        feet_phase_swing_height=0.06,
        feet_phase_tracking_sigma=0.008,
        base_height_target=0.55,
        min_base_height=0.25,
        max_tilt_deg=55.0,
        pose_weights=[1.0] * K1_NUM_ACTION,
    )
    ball = np.array(
        [
            [4.6, 0.0, 0.5],
            [4.6, 1.1, 0.5],
            [3.1, 0.0, 2.2],
            [2.4, 0.0, 0.1],
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(env._ball_missed_goal(ball), [False, True, False, False])
    np.testing.assert_array_equal(env._ball_flew_high(ball), [False, False, True, False])
    np.testing.assert_array_equal(env._ball_went_backward(ball), [False, False, False, True])


@pytest.mark.parametrize(
    ("owner", "backend", "num_envs"),
    [
        ("flashsac/k1_soccer_penalty_kick/motrix", "motrix", 1024),
        ("flashsac/k1_soccer_penalty_kick/mujoco", "mujoco", 4096),
    ],
)
def test_k1_soccer_penalty_flashsac_owner_compose(owner, backend, num_envs):
    from hydra import compose, initialize_config_dir
    from pathlib import Path

    conf_dir = Path(__file__).resolve().parents[4] / "conf" / "offpolicy"
    with initialize_config_dir(version_base=None, config_dir=str(conf_dir)):
        cfg = compose(config_name="config", overrides=[f"task={owner}"])

    assert cfg.training.task_name == "K1SoccerPenaltyKick"
    assert cfg.training.sim_backend == backend
    assert cfg.algo.num_envs == num_envs
    assert cfg.reward.scales.goal_scored == pytest.approx(200.0)
    assert cfg.reward.scales.ball_to_goal == pytest.approx(2.0)
    assert cfg.reward.scales.ball_goal_progress == pytest.approx(40.0)
    assert cfg.reward.scales.ball_kick_speed == pytest.approx(10.0)
    assert cfg.reward.scales.runup_success == pytest.approx(10.0)
    assert cfg.reward.scales.alive == pytest.approx(0.0)
    assert cfg.reward.scales.right_foot_to_ball == pytest.approx(6.67)
    assert cfg.reward.scales.right_foot_to_ball_progress == pytest.approx(5.0)
    assert cfg.reward.runup_timeout_enable_after_iteration == 1
    assert cfg.reward.scales.term_runup_timeout == pytest.approx(-30.0)
    assert cfg.reward.scales.term_kick_timeout == pytest.approx(-100.0)
    assert cfg.reward.scales.term_ball_missed_goal == pytest.approx(-600.0)
    assert cfg.reward.scales.term_fall == pytest.approx(-1200.0)
    assert cfg.reward.scales.runup_reach == pytest.approx(6.67)
    assert cfg.reward.scales.runup_speed == pytest.approx(8.0)
    assert cfg.reward.scales.tracking_lin_vel == pytest.approx(7.5)
    assert cfg.reward.runup_max_steps == 300
    assert cfg.reward.max_steps == 520
    assert cfg.reward.right_foot_strike_dist == pytest.approx(0.30)
    assert cfg.reward.runup_min_forward_speed == pytest.approx(0.40)
    assert cfg.reward.fixed_cmd_lin_speed == pytest.approx(0.75)
    assert cfg.reward.kick_phase_max_steps == 120
    assert cfg.env.commands.vel_limit == [[0.4, 0.0, 0.0], [0.95, 0.0, 0.0]]
    assert cfg.reward.runup_speed_curriculum_initial == pytest.approx(0.40)
    assert cfg.reward.runup_speed_curriculum_max_speed == pytest.approx(0.95)
    assert cfg.reward.runup_speed_success_curriculum_enabled is True
    assert cfg.reward.runup_speed_success_threshold == pytest.approx(0.15)
    assert cfg.reward.runup_speed_success_promotion_updates == 20
    assert cfg.reward.runup_start_distance_curriculum_enabled is True
    assert cfg.reward.runup_start_distance_phase1 == [0.20, 0.35]
    assert cfg.reward.runup_start_distance_phase2 == [0.35, 0.80]
    assert cfg.reward.runup_start_distance_phase3 == [2.0, 2.0]
    assert cfg.reward.runup_start_distance_success_threshold == pytest.approx(0.15)
    assert cfg.reward.runup_start_distance_promotion_updates == 20


def test_k1_soccer_penalty_mujoco_reset_step_obs_shape():
    pytest.importorskip("mujoco")
    import numpy as np

    from unilab.base import registry
    from unilab.base.registry import ensure_registries
    from unilab.envs.locomotion.k1.soccer_penalty_kick import K1SoccerPenaltyKickRewardConfig

    ensure_registries()
    reward_config = K1SoccerPenaltyKickRewardConfig(
        scales={"runup_reach": 1.0, "alive": 0.0, "goal_scored": 0.0},
        tracking_sigma=0.25,
        gait_frequency=1.5,
        feet_phase_swing_height=0.06,
        feet_phase_tracking_sigma=0.008,
        base_height_target=0.55,
        min_base_height=0.25,
        max_tilt_deg=55.0,
        pose_weights=[1.0] * K1_NUM_ACTION,
    )
    env = registry.make(
        "K1SoccerPenaltyKick",
        sim_backend="mujoco",
        num_envs=2,
        env_cfg_override={"reward_config": reward_config, "obs_frame_stack": 1},
    )
    actor_dim = k1_soccer_penalty_actor_obs_dim(env.action_space.shape[0])
    critic_dim = k1_soccer_penalty_critic_obs_dim(env.action_space.shape[0])
    assert env.obs_groups_spec == {"obs": actor_dim, "critic": critic_dim}
    env.init_state()
    obs, _info = env.reset(np.arange(2, dtype=np.int32))
    assert obs["obs"].shape == (2, actor_dim)
    assert obs["critic"].shape == (2, critic_dim)
    state = env.step(np.zeros((2, env.action_space.shape[0]), dtype=np.float32))
    assert state.obs["obs"].shape[0] == 2
    assert "runup_reached" in state.info

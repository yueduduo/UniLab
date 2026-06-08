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
    K1_PENALTY_STANCE_FOOT_WIDTH_XY_M,
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
    assert K1_PENALTY_STANCE_FOOT_WIDTH_XY_M == pytest.approx(0.1924)
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


def test_linear_runup_start_distance_samples_right_foot_distance_with_small_jitter():
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
        runup_start_distance_linear_enabled=True,
        runup_start_distance_linear_start=0.10,
        runup_start_distance_linear_end=1.10,
        runup_start_distance_linear_jitter=0.02,
        runup_start_distance_linear_y_jitter_max=0.0,
    )
    env = object.__new__(K1SoccerPenaltyKickEnv)
    env._reward_cfg = cfg
    env._training_iteration = 0
    env._runup_start_distance_linear_current_m = 0.10
    env._init_qpos = np.array(
        [K1_PENALTY_ROBOT_START_XY[0], K1_PENALTY_ROBOT_START_XY[1], 0.55],
        dtype=np.float64,
    )
    env._right_foot_offset_from_base_xy = np.array([0.0162, -0.0962], dtype=np.float64)
    provider = K1PenaltyKickDomainRandomizationProvider()

    offset = provider._sample_reset_xy_offset(env, 256)
    reset_base_xy = env._init_qpos[:2] + offset
    reset_right_foot_xy = reset_base_xy + env._right_foot_offset_from_base_xy
    right_foot_ball_x_dist = float(K1_PENALTY_SPOT_XY[0]) - reset_right_foot_xy[:, 0]
    right_foot_ball_y = reset_right_foot_xy[:, 1]
    assert np.min(right_foot_ball_x_dist) >= 0.08
    assert np.max(right_foot_ball_x_dist) <= 0.12
    assert np.max(np.abs(right_foot_ball_y)) <= 1e-9


def test_linear_runup_start_distance_gate_pauses_when_kick_success_low():
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
        runup_start_distance_linear_enabled=True,
        runup_start_distance_linear_start=0.10,
        runup_start_distance_linear_end=1.10,
        runup_start_distance_linear_duration_iterations=10000,
        runup_start_distance_linear_success_gate_enabled=True,
        runup_start_distance_linear_success_threshold=0.20,
    )
    env = object.__new__(K1SoccerPenaltyKickEnv)
    env._reward_cfg = cfg
    env._training_iteration = 0
    env._runup_start_distance_linear_current_m = 0.10
    env._runup_start_distance_linear_last_iteration = 0
    env._kick_success_rate_initialized = True
    env._kick_success_rate_ema = 0.0
    env._goal_success_rate_initialized = False
    env._goal_success_rate_ema = 0.0

    env.set_training_iteration(1000)
    assert env._runup_start_distance_current() == pytest.approx(0.10)
    env._kick_success_rate_ema = 0.50
    env.set_training_iteration(2000)
    assert env._runup_start_distance_current() == pytest.approx(0.20)


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
        info={"kick_phase": np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)},
        linvel=np.array(
            [[0.20, 0.0, 0.0], [0.40, 0.0, 0.0], [0.80, 0.0, 0.0], [0.95, 0.0, 0.0]],
            dtype=np.float32,
        ),
        gyro=np.zeros((4, 3), dtype=np.float32),
        dof_pos=np.zeros((4, 1), dtype=np.float32),
        num_envs=4,
        default_angles=np.zeros(1, dtype=np.float32),
        tracking_sigma=0.25,
        base_height_target=0.55,
        base_height=np.ones(4, dtype=np.float32),
        gravity=np.zeros((4, 3), dtype=np.float32),
        dof_vel=np.zeros((4, 1), dtype=np.float32),
        pose_weights=np.ones(1, dtype=np.float32),
    )
    reward = env._reward_runup_speed(ctx)
    assert reward[0] == pytest.approx(0.5)
    assert reward[1] == pytest.approx(1.0)
    assert reward[2] > reward[1]
    assert reward[3] == pytest.approx(2.0)


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


def test_proximity_shaping_disabled_when_ball_pinned():
    import numpy as np

    from unilab.envs.locomotion.common.rewards import RewardContext
    from unilab.envs.locomotion.k1.soccer_penalty_kick import (
        K1SoccerPenaltyKickEnv,
        K1SoccerPenaltyKickRewardConfig,
    )

    cfg = K1SoccerPenaltyKickRewardConfig(
        scales={
            "right_foot_to_ball": 1.0,
            "runup_reach": 1.0,
            "penalty_ball_pin": -1.0,
        },
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
    env._runup_reached = np.zeros(2, dtype=bool)
    env._runup_reach_reward_frozen = np.zeros(2, dtype=np.float32)
    ctx = RewardContext(
        info={
            "right_foot_ball_dist_xy": np.array([0.05, 0.05], dtype=np.float32),
            "left_foot_runup_reach_dist_xy": np.array([0.05, 0.05], dtype=np.float32),
            "kick_phase": np.array([0.0, 0.0], dtype=np.float32),
            "ball_pin": np.array([1.0, 0.0], dtype=np.float32),
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

    assert env._reward_right_foot_to_ball(ctx)[0] == pytest.approx(0.0)
    assert env._reward_right_foot_to_ball(ctx)[1] > 0.0
    assert env._reward_runup_reach(ctx)[0] == pytest.approx(0.0)
    assert env._reward_runup_reach(ctx)[1] > 0.0
    assert env._reward_penalty_ball_pin(ctx)[0] == pytest.approx(1.0)
    assert env._reward_penalty_ball_pin(ctx)[1] == pytest.approx(0.0)


def test_right_foot_to_ball_active_during_early_linear_curriculum():
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
        runup_start_distance_linear_enabled=True,
    )
    env = object.__new__(K1SoccerPenaltyKickEnv)
    env._reward_cfg = cfg
    env._runup_start_distance_linear_current_m = 0.30
    ctx = RewardContext(
        info={
            "right_foot_ball_dist_xy": np.array([0.20], dtype=np.float32),
            "kick_phase": np.array([0.0], dtype=np.float32),
        },
        linvel=np.zeros((1, 3), dtype=np.float32),
        gyro=np.zeros((1, 3), dtype=np.float32),
        dof_pos=np.zeros((1, 1), dtype=np.float32),
        num_envs=1,
        default_angles=np.zeros(1, dtype=np.float32),
        tracking_sigma=0.25,
        base_height_target=0.55,
        base_height=np.ones(1, dtype=np.float32),
        gravity=np.zeros((1, 3), dtype=np.float32),
        dof_vel=np.zeros((1, 1), dtype=np.float32),
        pose_weights=np.ones(1, dtype=np.float32),
    )

    assert env._reward_right_foot_to_ball(ctx)[0] > 0.0


def test_runup_reach_not_gated_by_close_distance_curriculum():
    import numpy as np

    from unilab.envs.locomotion.common.rewards import RewardContext
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
        runup_start_distance_linear_enabled=True,
    )
    env = object.__new__(K1SoccerPenaltyKickEnv)
    env._reward_cfg = cfg
    env._runup_reached = np.zeros(1, dtype=bool)
    env._runup_reach_reward_frozen = np.zeros(1, dtype=np.float32)
    env._runup_start_distance_linear_current_m = 0.30
    ctx = RewardContext(
        info={
            "left_foot_runup_reach_dist_xy": np.array([0.80], dtype=np.float32),
            "kick_phase": np.array([0.0], dtype=np.float32),
        },
        linvel=np.zeros((1, 3), dtype=np.float32),
        gyro=np.zeros((1, 3), dtype=np.float32),
        dof_pos=np.zeros((1, 1), dtype=np.float32),
        num_envs=1,
        default_angles=np.zeros(1, dtype=np.float32),
        tracking_sigma=0.25,
        base_height_target=0.55,
        base_height=np.ones(1, dtype=np.float32),
        gravity=np.zeros((1, 3), dtype=np.float32),
        dof_vel=np.zeros((1, 1), dtype=np.float32),
        pose_weights=np.ones(1, dtype=np.float32),
    )

    assert env._reward_runup_reach(ctx)[0] > 0.0


def test_runup_reach_uses_left_foot_distance_to_ball_left_target():
    import numpy as np

    from unilab.envs.locomotion.k1.soccer_penalty_kick import K1SoccerPenaltyKickEnv

    env = K1SoccerPenaltyKickEnv.__new__(K1SoccerPenaltyKickEnv)
    env._runup_reach_left_target_offset_from_ball_xy = np.array([0.0, 0.1924], dtype=np.float64)
    env._ball_state = lambda: (  # type: ignore[method-assign]
        np.array([[3.0, 0.0, 0.09]], dtype=np.float64),
        np.zeros((1, 3), dtype=np.float64),
        np.zeros((1, 3), dtype=np.float64),
        np.zeros((1, 3), dtype=np.float64),
    )
    env._foot_positions_w = lambda: (  # type: ignore[method-assign]
        np.array([[3.0, 0.1924, 0.0]], dtype=np.float64),
        np.zeros((1, 3), dtype=np.float64),
    )
    assert env._left_foot_runup_reach_distance_xy()[0] == pytest.approx(0.0)
    target = env._runup_reach_ball_left_target_xy_w()[0]
    ball = np.array([3.0, 0.0], dtype=np.float64)
    assert np.linalg.norm(target - ball) == pytest.approx(0.1924)


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
    assert "ball_goal_progress" in K1_PENALTY_KICK_REWARD_KEYS
    assert "ball_kick_speed" in K1_PENALTY_KICK_REWARD_KEYS
    assert "penalty_ball_vy" in K1_PENALTY_KICK_REWARD_KEYS
    assert "penalty_ball_lateral_progress" in K1_PENALTY_KICK_REWARD_KEYS
    assert "kick_stability" in K1_PENALTY_KICK_REWARD_KEYS
    assert "kick_plant_foot_x" in K1_PENALTY_KICK_REWARD_KEYS
    assert "kick_plant_foot_order" in K1_PENALTY_KICK_REWARD_KEYS
    assert "penalty_kick_plant_foot_order" in K1_PENALTY_KICK_REWARD_KEYS
    assert "kick_pose" in K1_PENALTY_KICK_REWARD_KEYS
    assert "kick_robot_still" in K1_PENALTY_KICK_REWARD_KEYS
    assert "goal_scored" not in K1_PENALTY_KICK_REWARD_KEYS
    assert "termination_bad" not in K1_PENALTY_KICK_REWARD_KEYS


def test_kick_plant_foot_rewards_prefer_left_at_ball_x_and_ahead_of_right():
    import numpy as np

    from unilab.envs.locomotion.common.rewards import RewardContext
    from unilab.envs.locomotion.k1.soccer_penalty_kick import (
        K1SoccerPenaltyKickEnv,
        K1SoccerPenaltyKickRewardConfig,
    )

    cfg = K1SoccerPenaltyKickRewardConfig(
        scales={
            "kick_plant_foot_x": 1.0,
            "kick_plant_foot_order": 1.0,
            "penalty_kick_plant_foot_order": -1.0,
        },
        tracking_sigma=0.25,
        gait_frequency=1.5,
        feet_phase_swing_height=0.06,
        feet_phase_tracking_sigma=0.008,
        base_height_target=0.55,
        min_base_height=0.25,
        max_tilt_deg=65.0,
        kick_plant_foot_x_sigma=0.10,
        kick_plant_foot_min_lead_m=0.04,
    )
    env = object.__new__(K1SoccerPenaltyKickEnv)
    env._reward_cfg = cfg
    ctx = RewardContext(
        info={
            "left_foot_ball_x_error": np.array([0.0, 0.20], dtype=np.float32),
            "kick_plant_foot_lead_x": np.array([0.08, -0.02], dtype=np.float32),
            "kick_phase": np.array([1.0, 1.0], dtype=np.float32),
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

    assert env._reward_kick_plant_foot_x(ctx)[0] == pytest.approx(1.0)
    assert env._reward_kick_plant_foot_x(ctx)[1] < env._reward_kick_plant_foot_x(ctx)[0]
    assert env._reward_kick_plant_foot_order(ctx)[0] == pytest.approx(1.0)
    assert env._reward_kick_plant_foot_order(ctx)[1] == pytest.approx(0.0)
    assert env._reward_penalty_kick_plant_foot_order(ctx)[0] == pytest.approx(0.0)
    assert env._reward_penalty_kick_plant_foot_order(ctx)[1] == pytest.approx(1.0)


def test_ball_kick_speed_zero_when_ball_is_stationary():
    import numpy as np

    from unilab.envs.locomotion.k1.soccer_penalty_kick import (
        K1SoccerPenaltyKickEnv,
        K1SoccerPenaltyKickRewardConfig,
    )

    cfg = K1SoccerPenaltyKickRewardConfig(
        scales={"ball_kick_speed": 1.0},
        tracking_sigma=0.25,
        gait_frequency=1.5,
        feet_phase_swing_height=0.06,
        feet_phase_tracking_sigma=0.008,
        base_height_target=0.55,
        min_base_height=0.25,
        max_tilt_deg=65.0,
        ball_kick_speed_sigma=2.0,
        ball_kick_speed_min_forward=0.08,
    )
    env = object.__new__(K1SoccerPenaltyKickEnv)
    env._reward_cfg = cfg
    env._ball_state = lambda: (  # type: ignore[method-assign]
        np.zeros((2, 3), dtype=np.float64),
        np.array([[0.0, 0.0, 0.0], [0.04, 0.5, 0.0]], dtype=np.float64),
        np.zeros((2, 3), dtype=np.float64),
        np.zeros((2, 3), dtype=np.float64),
    )

    reward = env._reward_ball_kick_speed(type("Ctx", (), {"info": {}})())
    assert reward[0] == pytest.approx(0.0)
    assert reward[1] == pytest.approx(0.0)


def test_ball_kick_speed_increases_with_forward_ball_velocity():
    import numpy as np

    from unilab.envs.locomotion.k1.soccer_penalty_kick import (
        K1SoccerPenaltyKickEnv,
        K1SoccerPenaltyKickRewardConfig,
    )

    cfg = K1SoccerPenaltyKickRewardConfig(
        scales={"ball_kick_speed": 1.0},
        tracking_sigma=0.25,
        gait_frequency=1.5,
        feet_phase_swing_height=0.06,
        feet_phase_tracking_sigma=0.008,
        base_height_target=0.55,
        min_base_height=0.25,
        max_tilt_deg=65.0,
        ball_kick_speed_sigma=2.0,
        ball_kick_speed_min_forward=0.08,
        ball_kick_speed_max_forward=2.5,
    )
    env = object.__new__(K1SoccerPenaltyKickEnv)
    env._reward_cfg = cfg
    env._ball_state = lambda: (  # type: ignore[method-assign]
        np.zeros((3, 3), dtype=np.float64),
        np.array([[0.0, 0.0, 0.0], [1.0, 0.5, 0.0], [2.0, -1.0, 0.0]], dtype=np.float64),
        np.zeros((3, 3), dtype=np.float64),
        np.zeros((3, 3), dtype=np.float64),
    )

    reward = env._reward_ball_kick_speed(type("Ctx", (), {"info": {}})())
    assert reward[0] == pytest.approx(0.0)
    assert reward[1] == pytest.approx(np.expm1(0.5))
    assert reward[2] == pytest.approx(np.expm1(1.0))
    assert reward[2] > reward[1] > reward[0]


def test_ball_kick_speed_caps_forward_speed_at_max():
    import numpy as np

    from unilab.envs.locomotion.k1.soccer_penalty_kick import (
        K1SoccerPenaltyKickEnv,
        K1SoccerPenaltyKickRewardConfig,
    )

    cfg = K1SoccerPenaltyKickRewardConfig(
        scales={"ball_kick_speed": 1.0},
        tracking_sigma=0.25,
        gait_frequency=1.5,
        feet_phase_swing_height=0.06,
        feet_phase_tracking_sigma=0.008,
        base_height_target=0.55,
        min_base_height=0.25,
        max_tilt_deg=65.0,
        ball_kick_speed_sigma=2.0,
        ball_kick_speed_min_forward=0.08,
        ball_kick_speed_max_forward=2.5,
    )
    env = object.__new__(K1SoccerPenaltyKickEnv)
    env._reward_cfg = cfg
    env._ball_state = lambda: (  # type: ignore[method-assign]
        np.zeros((3, 3), dtype=np.float64),
        np.array([[2.0, 0.0, 0.0], [2.5, 0.0, 0.0], [8.0, 0.0, 0.0]], dtype=np.float64),
        np.zeros((3, 3), dtype=np.float64),
        np.zeros((3, 3), dtype=np.float64),
    )

    reward = env._reward_ball_kick_speed(type("Ctx", (), {"info": {}})())
    assert reward[0] == pytest.approx(np.expm1(1.0))
    assert reward[1] == pytest.approx(np.expm1(1.25))
    assert reward[2] == pytest.approx(np.expm1(1.25))
    assert reward[2] == pytest.approx(reward[1])


def test_kick_pose_reward_prefers_default_pose_and_is_kick_phase_masked():
    import numpy as np

    from unilab.envs.locomotion.common.rewards import RewardContext
    from unilab.envs.locomotion.k1.soccer_penalty_kick import (
        K1SoccerPenaltyKickEnv,
        K1SoccerPenaltyKickRewardConfig,
    )

    cfg = K1SoccerPenaltyKickRewardConfig(
        scales={"kick_pose": 1.0},
        tracking_sigma=0.25,
        gait_frequency=1.5,
        feet_phase_swing_height=0.06,
        feet_phase_tracking_sigma=0.008,
        base_height_target=0.55,
        min_base_height=0.25,
        max_tilt_deg=65.0,
        pose_weights=[1.0],
        kick_pose_sigma=0.15,
    )
    env = object.__new__(K1SoccerPenaltyKickEnv)
    env._reward_cfg = cfg
    ctx = RewardContext(
        info={},
        linvel=np.zeros((2, 3), dtype=np.float32),
        gyro=np.zeros((2, 3), dtype=np.float32),
        dof_pos=np.array([[0.0], [0.2]], dtype=np.float32),
        num_envs=2,
        default_angles=np.zeros(1, dtype=np.float32),
        tracking_sigma=0.25,
        base_height_target=0.55,
        base_height=np.ones(2, dtype=np.float32),
        gravity=np.zeros((2, 3), dtype=np.float32),
        dof_vel=np.zeros((2, 1), dtype=np.float32),
        pose_weights=np.ones(1, dtype=np.float32),
    )
    reward = env._reward_kick_pose(ctx)
    assert reward[0] == pytest.approx(1.0)
    assert reward[1] == pytest.approx(np.exp(-0.04 / 0.15))
    assert reward[0] > reward[1]


def test_kick_robot_still_reward_prefers_low_speed():
    import numpy as np

    from unilab.envs.locomotion.common.rewards import RewardContext
    from unilab.envs.locomotion.k1.soccer_penalty_kick import (
        K1SoccerPenaltyKickEnv,
        K1SoccerPenaltyKickRewardConfig,
    )

    cfg = K1SoccerPenaltyKickRewardConfig(
        scales={"kick_robot_still": 1.0},
        tracking_sigma=0.25,
        gait_frequency=1.5,
        feet_phase_swing_height=0.06,
        feet_phase_tracking_sigma=0.008,
        base_height_target=0.55,
        min_base_height=0.25,
        max_tilt_deg=65.0,
        kick_robot_still_sigma=0.15,
    )
    env = object.__new__(K1SoccerPenaltyKickEnv)
    env._reward_cfg = cfg
    ctx = RewardContext(
        info={},
        linvel=np.array([[0.0, 0.0, 0.0], [0.3, 0.0, 0.0]], dtype=np.float32),
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
    reward = env._reward_kick_robot_still(ctx)
    assert reward[0] == pytest.approx(1.0)
    assert reward[1] == pytest.approx(np.exp(-0.3 / 0.15))
    assert reward[0] > reward[1]


def test_kick_phase_pose_and_still_rewards_masked_until_runup_reached():
    import numpy as np

    from unilab.envs.locomotion.k1.soccer_penalty_kick import (
        K1SoccerPenaltyKickEnv,
        K1SoccerPenaltyKickRewardConfig,
    )

    cfg = K1SoccerPenaltyKickRewardConfig(
        scales={"kick_pose": 8.0, "kick_robot_still": 5.0},
        tracking_sigma=0.25,
        gait_frequency=1.5,
        feet_phase_swing_height=0.06,
        feet_phase_tracking_sigma=0.008,
        base_height_target=0.55,
        min_base_height=0.25,
        max_tilt_deg=65.0,
        pose_weights=[1.0],
        kick_pose_sigma=0.15,
        kick_robot_still_sigma=0.15,
    )
    env = object.__new__(K1SoccerPenaltyKickEnv)
    env._reward_cfg = cfg
    env._num_envs = 1
    env._cfg = type("Cfg", (), {"ctrl_dt": 0.02})()
    env._enable_reward_log = True
    env.default_angles = np.zeros(1, dtype=np.float32)
    env._pose_weights = np.ones(1, dtype=np.float32)
    env._backend = type(  # type: ignore[method-assign]
        "Backend",
        (),
        {"get_base_pos": lambda self: np.array([[0.0, 0.0, 0.5]], dtype=np.float64)},
    )()
    env._penalty_curriculum = None
    env._runup_speed_curriculum_level = lambda: 0.4
    env._runup_speed_curriculum_max = lambda: 0.95
    env._runup_start_distance_current = lambda: 0.3
    env._runup_reset_y_jitter_limit = lambda: 0.0
    env._runup_success_rate_ema = 0.0
    env._kick_success_rate_ema = 0.0
    env._goal_success_rate_ema = 0.0
    env._runup_start_distance_phase_idx = 0
    env._ball_state = lambda: (  # type: ignore[method-assign]
        np.zeros((1, 3), dtype=np.float64),
        np.zeros((1, 3), dtype=np.float64),
        np.zeros((1, 3), dtype=np.float64),
        np.zeros((1, 3), dtype=np.float64),
    )
    env._init_reward_functions()

    def _run_case(runup_reached: bool) -> dict:
        env._runup_reached = np.array([runup_reached], dtype=bool)
        info = {
            "steps": np.array([4], dtype=np.uint32),
            "terminated": np.zeros(1, dtype=bool),
            "kick_phase": np.array([float(runup_reached)], dtype=np.float32),
            "runup_reached": np.array([float(runup_reached)], dtype=np.float32),
            "ball_pin": np.zeros(1, dtype=np.float32),
        }
        env._compute_reward(
            info,
            np.array([[0.3, 0.0, 0.0]], dtype=np.float32),
            np.zeros((1, 3), dtype=np.float32),
            np.zeros((1, 3), dtype=np.float32),
            np.array([[0.2]], dtype=np.float32),
            np.zeros((1, 1), dtype=np.float32),
        )
        return info["log"]

    before = _run_case(False)
    after = _run_case(True)
    assert before["reward/kick_pose"] == pytest.approx(0.0)
    assert before["reward/kick_robot_still"] == pytest.approx(0.0)
    assert after["reward/kick_pose"] == pytest.approx(8.0 * np.exp(-0.04 / 0.15))
    assert after["reward/kick_robot_still"] == pytest.approx(5.0 * np.exp(-0.3 / 0.15))


def test_penalty_runup_feet_still_penalizes_stationary_feet_in_runup_phase():
    import numpy as np

    from unilab.envs.locomotion.common.rewards import RewardContext
    from unilab.envs.locomotion.k1.soccer_penalty_kick import (
        K1SoccerPenaltyKickEnv,
        K1SoccerPenaltyKickRewardConfig,
    )

    cfg = K1SoccerPenaltyKickRewardConfig(
        scales={"penalty_runup_feet_still": -1.0},
        tracking_sigma=0.25,
        gait_frequency=1.5,
        feet_phase_swing_height=0.06,
        feet_phase_tracking_sigma=0.008,
        base_height_target=0.55,
        min_base_height=0.25,
        max_tilt_deg=65.0,
        runup_feet_still_sigma=0.015,
    )
    env = object.__new__(K1SoccerPenaltyKickEnv)
    env._reward_cfg = cfg
    ctx = RewardContext(
        info={
            "kick_phase": np.array([0.0, 1.0], dtype=np.float32),
            "feet_step_travel": np.array([0.0, 0.0], dtype=np.float32),
            "ball_pin": np.zeros(2, dtype=np.float32),
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

    reward = env._reward_penalty_runup_feet_still(ctx)
    assert reward[0] == pytest.approx(1.0)
    assert reward[1] == pytest.approx(0.0)


def test_penalty_runup_feet_still_decreases_with_foot_travel():
    import numpy as np

    from unilab.envs.locomotion.common.rewards import RewardContext
    from unilab.envs.locomotion.k1.soccer_penalty_kick import (
        K1SoccerPenaltyKickEnv,
        K1SoccerPenaltyKickRewardConfig,
    )

    cfg = K1SoccerPenaltyKickRewardConfig(
        scales={"penalty_runup_feet_still": -1.0},
        tracking_sigma=0.25,
        gait_frequency=1.5,
        feet_phase_swing_height=0.06,
        feet_phase_tracking_sigma=0.008,
        base_height_target=0.55,
        min_base_height=0.25,
        max_tilt_deg=65.0,
        runup_feet_still_sigma=0.015,
    )
    env = object.__new__(K1SoccerPenaltyKickEnv)
    env._reward_cfg = cfg
    ctx = RewardContext(
        info={
            "kick_phase": np.zeros(2, dtype=np.float32),
            "feet_step_travel": np.array([0.0, 0.03], dtype=np.float32),
            "ball_pin": np.zeros(2, dtype=np.float32),
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

    reward = env._reward_penalty_runup_feet_still(ctx)
    assert reward[0] == pytest.approx(1.0)
    assert reward[1] == pytest.approx(np.exp(-0.03 / 0.015))
    assert reward[1] < reward[0]


def test_ball_goal_progress_zero_when_ball_does_not_move_forward():
    import numpy as np

    from unilab.envs.locomotion.common.rewards import RewardContext
    from unilab.envs.locomotion.k1.soccer_penalty_kick import (
        K1SoccerPenaltyKickEnv,
        K1SoccerPenaltyKickRewardConfig,
    )

    cfg = K1SoccerPenaltyKickRewardConfig(
        scales={"ball_goal_progress": 1.0},
        tracking_sigma=0.25,
        gait_frequency=1.5,
        feet_phase_swing_height=0.06,
        feet_phase_tracking_sigma=0.008,
        base_height_target=0.55,
        min_base_height=0.25,
        max_tilt_deg=65.0,
        ball_goal_progress_sigma=0.05,
    )
    env = object.__new__(K1SoccerPenaltyKickEnv)
    env._reward_cfg = cfg
    ctx = RewardContext(
        info={"ball_forward_progress": np.array([0.0, 0.0], dtype=np.float32)},
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

    reward = env._reward_ball_goal_progress(ctx)
    assert reward[0] == pytest.approx(0.0)
    assert reward[1] == pytest.approx(0.0)


def test_ball_goal_progress_uses_exponential_forward_x_movement():
    import numpy as np

    from unilab.envs.locomotion.common.rewards import RewardContext
    from unilab.envs.locomotion.k1.soccer_penalty_kick import (
        K1SoccerPenaltyKickEnv,
        K1SoccerPenaltyKickRewardConfig,
    )

    cfg = K1SoccerPenaltyKickRewardConfig(
        scales={"ball_goal_progress": 1.0},
        tracking_sigma=0.25,
        gait_frequency=1.5,
        feet_phase_swing_height=0.06,
        feet_phase_tracking_sigma=0.008,
        base_height_target=0.55,
        min_base_height=0.25,
        max_tilt_deg=65.0,
        ball_goal_progress_sigma=0.05,
    )
    env = object.__new__(K1SoccerPenaltyKickEnv)
    env._reward_cfg = cfg
    ctx = RewardContext(
        info={
            "ball_forward_progress": np.array([0.0, 0.05, 0.15], dtype=np.float32),
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

    reward = env._reward_ball_goal_progress(ctx)
    assert reward[0] == pytest.approx(0.0)
    assert reward[1] == pytest.approx(1.0 - np.exp(-0.05 / 0.05))
    assert reward[2] == pytest.approx(1.0 - np.exp(-0.15 / 0.05))
    assert reward[2] > reward[1] > reward[0]


def test_penalty_ball_vy_scales_with_lateral_ball_speed():
    import numpy as np

    from unilab.envs.locomotion.common.rewards import RewardContext
    from unilab.envs.locomotion.k1.soccer_penalty_kick import (
        K1SoccerPenaltyKickEnv,
        K1SoccerPenaltyKickRewardConfig,
    )

    cfg = K1SoccerPenaltyKickRewardConfig(
        scales={"penalty_ball_vy": -1.0},
        tracking_sigma=0.25,
        gait_frequency=1.5,
        feet_phase_swing_height=0.06,
        feet_phase_tracking_sigma=0.008,
        base_height_target=0.55,
        min_base_height=0.25,
        max_tilt_deg=65.0,
        ball_kick_speed_sigma=2.0,
        ball_kick_speed_min_forward=0.08,
    )
    env = object.__new__(K1SoccerPenaltyKickEnv)
    env._reward_cfg = cfg
    ctx = RewardContext(
        info={"ball_vy_lateral": np.array([0.0, 0.5, 2.0], dtype=np.float32)},
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

    reward = env._reward_penalty_ball_vy(ctx)
    assert reward[0] == pytest.approx(0.0)
    assert reward[1] == pytest.approx(1.0 - np.exp(-0.5 / 2.0))
    assert reward[2] == pytest.approx(1.0 - np.exp(-1.0))


def test_penalty_ball_lateral_progress_scales_with_y_movement():
    import numpy as np

    from unilab.envs.locomotion.common.rewards import RewardContext
    from unilab.envs.locomotion.k1.soccer_penalty_kick import (
        K1SoccerPenaltyKickEnv,
        K1SoccerPenaltyKickRewardConfig,
    )

    cfg = K1SoccerPenaltyKickRewardConfig(
        scales={"penalty_ball_lateral_progress": -1.0},
        tracking_sigma=0.25,
        gait_frequency=1.5,
        feet_phase_swing_height=0.06,
        feet_phase_tracking_sigma=0.008,
        base_height_target=0.55,
        min_base_height=0.25,
        max_tilt_deg=65.0,
        ball_goal_progress_sigma=0.05,
    )
    env = object.__new__(K1SoccerPenaltyKickEnv)
    env._reward_cfg = cfg
    ctx = RewardContext(
        info={"ball_lateral_progress": np.array([0.0, 0.05, 0.15], dtype=np.float32)},
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

    reward = env._reward_penalty_ball_lateral_progress(ctx)
    assert reward[0] == pytest.approx(0.0)
    assert reward[1] == pytest.approx(1.0 - np.exp(-0.05 / 0.05))
    assert reward[2] == pytest.approx(1.0 - np.exp(-0.15 / 0.05))


def test_ball_shot_rewards_masked_until_runup_reached():
    import numpy as np

    from unilab.envs.locomotion.k1.soccer_penalty_kick import (
        K1SoccerPenaltyKickEnv,
        K1SoccerPenaltyKickRewardConfig,
    )

    cfg = K1SoccerPenaltyKickRewardConfig(
        scales={
            "ball_goal_progress": 16.0,
            "ball_kick_speed": 120.0,
            "penalty_ball_vy": -40.0,
            "penalty_ball_lateral_progress": -40.0,
        },
        tracking_sigma=0.25,
        gait_frequency=1.5,
        feet_phase_swing_height=0.06,
        feet_phase_tracking_sigma=0.008,
        base_height_target=0.55,
        min_base_height=0.25,
        max_tilt_deg=65.0,
        ball_goal_progress_sigma=0.05,
        ball_kick_speed_sigma=2.0,
        ball_kick_speed_min_forward=0.08,
    )
    env = object.__new__(K1SoccerPenaltyKickEnv)
    env._reward_cfg = cfg
    env._num_envs = 1
    env._cfg = type("Cfg", (), {"ctrl_dt": 0.02})()
    env._enable_reward_log = True
    env.default_angles = np.zeros(1, dtype=np.float32)
    env._pose_weights = np.ones(1, dtype=np.float32)
    env._backend = type(  # type: ignore[method-assign]
        "Backend",
        (),
        {"get_base_pos": lambda self: np.array([[0.0, 0.0, 0.5]], dtype=np.float64)},
    )()
    env._penalty_curriculum = None
    env._runup_speed_curriculum_level = lambda: 0.4
    env._runup_speed_curriculum_max = lambda: 0.95
    env._runup_start_distance_current = lambda: 0.3
    env._runup_reset_y_jitter_limit = lambda: 0.0
    env._runup_success_rate_ema = 0.0
    env._kick_success_rate_ema = 0.0
    env._goal_success_rate_ema = 0.0
    env._runup_start_distance_phase_idx = 0
    env._ball_state = lambda: (  # type: ignore[method-assign]
        np.zeros((1, 3), dtype=np.float64),
        np.array([[2.0, 0.0, 0.0]], dtype=np.float64),
        np.zeros((1, 3), dtype=np.float64),
        np.zeros((1, 3), dtype=np.float64),
    )
    env._init_reward_functions()

    def _run_case(runup_reached: bool) -> dict:
        env._runup_reached = np.array([runup_reached], dtype=bool)
        info = {
            "steps": np.array([4], dtype=np.uint32),
            "terminated": np.zeros(1, dtype=bool),
            "kick_phase": np.array([float(runup_reached)], dtype=np.float32),
            "runup_reached": np.array([float(runup_reached)], dtype=np.float32),
            "ball_forward_progress": np.array([0.1 if runup_reached else 0.0], dtype=np.float32),
            "ball_lateral_progress": np.array([0.0], dtype=np.float32),
            "ball_vy_lateral": np.array([0.0], dtype=np.float32),
            "ball_pin": np.zeros(1, dtype=np.float32),
        }
        env._compute_reward(
            info,
            np.zeros((1, 3), dtype=np.float32),
            np.zeros((1, 3), dtype=np.float32),
            np.zeros((1, 3), dtype=np.float32),
            np.zeros((1, 1), dtype=np.float32),
            np.zeros((1, 1), dtype=np.float32),
        )
        return info["log"]

    before = _run_case(False)
    after = _run_case(True)
    progress_raw = 1.0 - np.exp(-0.1 / 0.05)
    speed_raw = np.expm1(1.0)
    assert before["reward/ball_goal_progress"] == pytest.approx(0.0)
    assert before["reward/ball_kick_speed"] == pytest.approx(0.0)
    assert after["reward/ball_goal_progress"] == pytest.approx(progress_raw * 16.0)
    assert after["reward/ball_kick_speed"] == pytest.approx(speed_raw * 120.0)


def test_forward_progress_reward_only_during_runup_and_scales_with_speed():
    import numpy as np

    from unilab.envs.locomotion.common.rewards import RewardContext
    from unilab.envs.locomotion.k1.soccer_penalty_kick import (
        K1SoccerPenaltyKickEnv,
        K1SoccerPenaltyKickRewardConfig,
    )

    cfg = K1SoccerPenaltyKickRewardConfig(
        scales={"forward_progress": 1.0},
        tracking_sigma=0.25,
        gait_frequency=1.5,
        feet_phase_swing_height=0.06,
        feet_phase_tracking_sigma=0.008,
        base_height_target=0.55,
        min_base_height=0.25,
        max_tilt_deg=65.0,
        runup_speed_curriculum_max_speed=1.0,
    )
    env = object.__new__(K1SoccerPenaltyKickEnv)
    env._reward_cfg = cfg
    env._training_iteration = 1
    ctx = RewardContext(
        info={"kick_phase": np.array([0.0, 1.0], dtype=np.float32)},
        linvel=np.array([[0.50, 0.0, 0.0], [0.50, 0.0, 0.0]], dtype=np.float32),
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

    reward = env._reward_forward_progress(ctx)
    assert reward[0] == pytest.approx(0.5)
    assert reward[1] == pytest.approx(0.0)


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


def test_kick_phase_requires_runup_plant_and_forward_motion():
    import numpy as np

    from unilab.envs.locomotion.k1.soccer_penalty_kick import (
        K1SoccerPenaltyKickEnv,
        K1SoccerPenaltyKickRewardConfig,
    )

    env = K1SoccerPenaltyKickEnv.__new__(K1SoccerPenaltyKickEnv)
    env._num_envs = 5
    env._reward_cfg = K1SoccerPenaltyKickRewardConfig(
        scales={},
        tracking_sigma=0.25,
        gait_frequency=1.5,
        feet_phase_swing_height=0.06,
        feet_phase_tracking_sigma=0.008,
        base_height_target=0.55,
        min_base_height=0.25,
        max_tilt_deg=55.0,
        pose_weights=[1.0] * K1_NUM_ACTION,
        right_foot_strike_dist=0.18,
        kick_phase_min_ball_forward_speed=0.05,
        kick_phase_min_robot_forward_speed=0.15,
        kick_phase_max_left_foot_ball_x_error=0.08,
        kick_plant_foot_min_lead_m=0.02,
        kick_phase_min_base_height=0.40,
    )
    env._runup_reached = np.zeros(5, dtype=bool)
    env._right_foot_ball_distance_xy = lambda: np.array(  # type: ignore[method-assign]
        [0.10, 0.10, 0.10, 0.10, 0.30], dtype=np.float32
    )
    env._ball_state = lambda: (  # type: ignore[method-assign]
        np.array([[3.0, 0.0, 0.09]] * 5, dtype=np.float32),
        np.array(
            [
                [0.10, 0.0, 0.0],
                [0.15, 0.0, 0.0],
                [0.15, 0.0, 0.0],
                [0.15, 0.0, 0.0],
                [0.15, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        np.zeros((5, 3), dtype=np.float32),
        np.zeros((5, 3), dtype=np.float32),
    )
    env.get_local_linvel = lambda: np.array(  # type: ignore[method-assign]
        [
            [0.20, 0.0, 0.0],
            [0.35, 0.0, 0.0],
            [0.35, 0.0, 0.0],
            [0.35, 0.0, 0.0],
            [0.35, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    env._foot_positions_w = lambda: (  # type: ignore[method-assign]
        np.array(
            [
                [3.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [3.2, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        np.array(
            [
                [2.9, 0.0, 0.0],
                [2.9, 0.0, 0.0],
                [2.9, 0.0, 0.0],
                [3.1, 0.0, 0.0],
                [2.9, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )
    env._backend = type(  # type: ignore[method-assign]
        "Backend",
        (),
        {"get_base_pos": lambda self: np.array([[0.45, 0.0, 0.5]] * 5, dtype=np.float32)},
    )()

    newly_reached = env._update_runup_reached()
    # env 0/1 satisfy speed + plant constraints; env 2 bad plant x, env 3 bad foot order, env 4 too far
    np.testing.assert_array_equal(newly_reached, [True, True, False, False, False])
    np.testing.assert_array_equal(env._runup_reached, [True, True, False, False, False])


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
    assert cfg.reward.scales.goal_scored == pytest.approx(100.0)
    assert cfg.reward.scales.ball_to_goal == pytest.approx(10.0)
    assert cfg.reward.scales.ball_goal_progress == pytest.approx(16.0)
    assert cfg.reward.scales.ball_kick_speed == pytest.approx(120.0)
    assert cfg.reward.scales.penalty_ball_vy == pytest.approx(-40.0)
    assert cfg.reward.scales.penalty_ball_lateral_progress == pytest.approx(-40.0)
    assert cfg.reward.scales.runup_success == pytest.approx(10.0)
    assert cfg.reward.scales.alive == pytest.approx(0.0)
    assert cfg.reward.scales.forward_progress == pytest.approx(12.5)
    assert cfg.reward.scales.kick_plant_foot_x == pytest.approx(8.0)
    assert cfg.reward.scales.kick_plant_foot_order == pytest.approx(6.0)
    assert cfg.reward.scales.penalty_kick_plant_foot_order == pytest.approx(-10.0)
    assert cfg.reward.scales.right_foot_to_ball == pytest.approx(8.893)
    assert cfg.reward.scales.right_foot_to_ball_progress == pytest.approx(5.0)
    assert cfg.reward.runup_timeout_enable_after_iteration == 1
    assert cfg.reward.scales.term_runup_timeout == pytest.approx(-30.0)
    assert cfg.reward.scales.term_kick_timeout == pytest.approx(-100.0)
    assert cfg.reward.scales.term_ball_missed_goal == pytest.approx(-600.0)
    assert cfg.reward.scales.term_fall == pytest.approx(-1200.0)
    assert cfg.reward.scales.term_low == pytest.approx(-120.0)
    assert cfg.reward.scales.kick_pose == pytest.approx(8.0)
    assert cfg.reward.scales.kick_robot_still == pytest.approx(5.0)
    assert cfg.reward.scales.penalty_runup_feet_still == pytest.approx(-5.0)
    assert cfg.reward.runup_feet_still_sigma == pytest.approx(0.015)
    assert cfg.reward.ball_kick_speed_max_forward == pytest.approx(2.5)
    assert cfg.reward.scales.runup_reach == pytest.approx(8.893)
    assert cfg.reward.scales.runup_speed == pytest.approx(15.0)
    assert cfg.reward.scales.tracking_lin_vel == pytest.approx(5.0)
    assert cfg.reward.scales.kick_stability == pytest.approx(5.0)
    assert cfg.reward.scales.penalty_orientation == pytest.approx(-15.0)
    assert cfg.reward.runup_max_steps == 300
    assert cfg.reward.max_steps == 520
    assert cfg.reward.right_foot_strike_dist == pytest.approx(0.18)
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
    assert cfg.reward.runup_start_distance_linear_enabled is True
    assert cfg.reward.runup_start_distance_linear_start == pytest.approx(0.10)
    assert cfg.reward.runup_start_distance_linear_end == pytest.approx(1.10)
    assert cfg.reward.runup_start_distance_linear_duration_iterations == 10000
    assert cfg.reward.runup_start_distance_linear_jitter == pytest.approx(0.02)
    assert cfg.reward.runup_start_distance_linear_success_gate_enabled is True
    assert cfg.reward.runup_start_distance_linear_success_threshold == pytest.approx(0.20)
    assert cfg.reward.runup_start_distance_linear_y_jitter_max == pytest.approx(0.05)
    assert cfg.reward.kick_phase_min_ball_forward_speed == pytest.approx(0.05)
    assert cfg.reward.kick_phase_min_robot_forward_speed == pytest.approx(0.15)
    assert cfg.reward.kick_phase_max_left_foot_ball_x_error == pytest.approx(0.08)
    assert cfg.reward.kick_phase_require_plant_foot_lead is True
    assert cfg.reward.kick_phase_min_base_height == pytest.approx(0.40)
    assert cfg.reward.kick_plant_foot_x_sigma == pytest.approx(0.10)
    assert cfg.reward.kick_plant_foot_min_lead_m == pytest.approx(0.02)
    assert cfg.reward.runup_speed_bonus_cap == pytest.approx(1.0)
    assert cfg.reward.runup_reach_sigma == pytest.approx(0.30)


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

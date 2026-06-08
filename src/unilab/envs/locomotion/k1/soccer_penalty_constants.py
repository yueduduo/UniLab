"""K1 penalty-kick layout aligned with sim_soccer2 match_config + world.xml."""

from __future__ import annotations

from unilab.envs.locomotion.k1.constants import K1_NUM_ACTION, k1_walk_actor_obs_dim, k1_walk_critic_obs_dim

# world.xml green pitch plane half-size 6 x 4.5 -> 12 m x 9 m visual ground.
K1_PENALTY_PITCH_LENGTH_M = 12.0
K1_PENALTY_PITCH_WIDTH_M = 9.0

# match_config.json markings / referee playable field (white-line rectangle).
K1_PENALTY_MARKINGS_FIELD_LENGTH_M = 9.0
K1_PENALTY_MARKINGS_FIELD_WIDTH_M = 6.0

# Robot_Goal.obj right-mouth front ≈ +4.49 m; referee uses half of markings length.
K1_PENALTY_GOAL_LINE_X = 0.5 * K1_PENALTY_MARKINGS_FIELD_LENGTH_M
K1_PENALTY_GOAL_WIDTH_M = 1.9
K1_PENALTY_GOAL_HEIGHT_M = 1.8

# multi_robot_sim._add_field_markings: x_spot = half_len - penalty_spot_distance
K1_PENALTY_SPOT_DISTANCE_M = 1.5
K1_PENALTY_SPOT_XY = (
    K1_PENALTY_GOAL_LINE_X - K1_PENALTY_SPOT_DISTANCE_M,
    0.0,
)

# Stand keyframe: right_foot site y ≈ base_y - 0.0962; shift base +Y so foot aligns with ball.
K1_PENALTY_RIGHT_FOOT_Y_OFFSET_FROM_BASE_M = 0.0962

# Run-up start: robot ~2 m behind the penalty spot, facing +X goal; base Y aligns right foot with ball.
K1_PENALTY_ROBOT_START_XY = (
    K1_PENALTY_SPOT_XY[0] - 2.0,
    K1_PENALTY_RIGHT_FOOT_Y_OFFSET_FROM_BASE_M,
)
K1_PENALTY_BALL_RADIUS_M = 0.09167862683534622
K1_PENALTY_BALL_SPAWN_Z = K1_PENALTY_BALL_RADIUS_M
K1_PENALTY_TRUNK_Z = 0.5743699226900935
K1_PENALTY_RESET_Y_OFFSET_M = 0.15
K1_PENALTY_RUNUP_BALL_DIST_M = 0.35
K1_PENALTY_RIGHT_FOOT_STRIKE_DIST_M = 0.22
# Stand keyframe: ||left_foot_xy - right_foot_xy|| ≈ 0.1924 (left target is ball + this offset).
K1_PENALTY_STANCE_FOOT_WIDTH_XY_M = 0.1924
K1_PENALTY_RUNUP_MAX_STEPS = 180
K1_PENALTY_KICK_PHASE_MAX_STEPS = 120
K1_PENALTY_MAX_STEPS = 400

# Kick-phase reward keys (masked until run-up reaches the ball).
K1_PENALTY_KICK_REWARD_KEYS: frozenset[str] = frozenset(
    {
        "ball_to_goal",
        "ball_goal_progress",
        "ball_kick_speed",
        "penalty_ball_vy",
        "penalty_ball_lateral_progress",
        "kick_stability",
        "kick_plant_foot_x",
        "kick_plant_foot_order",
        "penalty_kick_plant_foot_order",
        "kick_pose",
        "kick_robot_still",
    }
)


def k1_soccer_penalty_actor_obs_dim(num_action: int = K1_NUM_ACTION) -> int:
    return k1_walk_actor_obs_dim(num_action) + 3


def k1_soccer_penalty_critic_obs_dim(num_action: int = K1_NUM_ACTION) -> int:
    return k1_soccer_penalty_actor_obs_dim(num_action) + 3

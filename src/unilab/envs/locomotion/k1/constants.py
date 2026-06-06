"""K1 locomotion constants aligned with sim_soccer2 AMP deploy contract."""

from __future__ import annotations

import numpy as np

K1_ACTUATOR_JOINT_ORDER: tuple[str, ...] = (
    "AAHead_yaw",
    "Head_pitch",
    "ALeft_Shoulder_Pitch",
    "Left_Shoulder_Roll",
    "Left_Elbow_Pitch",
    "Left_Elbow_Yaw",
    "ARight_Shoulder_Pitch",
    "Right_Shoulder_Roll",
    "Right_Elbow_Pitch",
    "Right_Elbow_Yaw",
    "Left_Hip_Pitch",
    "Left_Hip_Roll",
    "Left_Hip_Yaw",
    "Left_Knee_Pitch",
    "Left_Ankle_Pitch",
    "Left_Ankle_Roll",
    "Right_Hip_Pitch",
    "Right_Hip_Roll",
    "Right_Hip_Yaw",
    "Right_Knee_Pitch",
    "Right_Ankle_Pitch",
    "Right_Ankle_Roll",
)

K1_DEFAULT_JOINT_ANGLES: dict[str, float] = {
    "AAHead_yaw": 0.0,
    "Head_pitch": 0.0,
    "ALeft_Shoulder_Pitch": 0.0,
    "Left_Shoulder_Roll": -1.3,
    "Left_Elbow_Pitch": 0.0,
    "Left_Elbow_Yaw": -0.5,
    "ARight_Shoulder_Pitch": 0.0,
    "Right_Shoulder_Roll": 1.3,
    "Right_Elbow_Pitch": 0.0,
    "Right_Elbow_Yaw": 0.5,
    "Left_Hip_Pitch": -0.15,
    "Left_Hip_Roll": 0.0,
    "Left_Hip_Yaw": 0.0,
    "Left_Knee_Pitch": 0.3,
    "Left_Ankle_Pitch": -0.15,
    "Left_Ankle_Roll": 0.0,
    "Right_Hip_Pitch": -0.15,
    "Right_Hip_Roll": 0.0,
    "Right_Hip_Yaw": 0.0,
    "Right_Knee_Pitch": 0.3,
    "Right_Ankle_Pitch": -0.15,
    "Right_Ankle_Roll": 0.0,
}

K1_CMD_MAX_LIN_X = 0.5
K1_CMD_MAX_LIN_Y = 0.4
K1_CMD_MAX_YAW = 1.0

K1_OBS_FRAME_STACK = 5
K1_OBS_SINGLE_DIM = 75
K1_OBS_STACKED_DIM = K1_OBS_SINGLE_DIM * K1_OBS_FRAME_STACK
K1_NUM_ACTION = len(K1_ACTUATOR_JOINT_ORDER)
K1_UPPER_BODY_JOINTS = 10
# Task keyframes: free-base qpos prefix (xyz + quat) before actuator joint coordinates.
K1_KEYFRAME_BASE_QPOS_DIM = 7
# G1-walk-aligned pose weights (22 DOF): low on head/arms, high on legs.
K1_G1_ALIGNED_POSE_WEIGHTS: tuple[float, ...] = (
    0.01,
    1.0,
    5.0,
    0.01,
    5.0,
    5.0,
    0.01,
    1.0,
    5.0,
    0.01,
    50.0,
    50.0,
    50.0,
    50.0,
    50.0,
    50.0,
    50.0,
    50.0,
    50.0,
    50.0,
    50.0,
    50.0,
)


def k1_keyframe_robot_joint_qpos(
    keyframe_qpos: np.ndarray, num_action: int = K1_NUM_ACTION
) -> np.ndarray:
    """Robot actuator qpos slice from a task keyframe (excludes trailing ball / object dofs)."""
    start = K1_KEYFRAME_BASE_QPOS_DIM
    stop = start + num_action
    return np.asarray(keyframe_qpos[start:stop], dtype=keyframe_qpos.dtype)


def k1_walk_actor_obs_dim(num_action: int = K1_NUM_ACTION) -> int:
    """G1 walk profile: gyro(3)+gravity(3)+joint(3n)+cmd(3)+phase(2)."""
    return 6 + 3 * num_action + 5


def k1_walk_critic_obs_dim(num_action: int = K1_NUM_ACTION) -> int:
    return k1_walk_actor_obs_dim(num_action) + 3


def k1_soccer_push_actor_obs_dim(num_action: int = K1_NUM_ACTION) -> int:
    # Walk layout (gyro/gravity/proprio/cmd/phase) + ball rel state (3).
    return k1_walk_actor_obs_dim(num_action) + 3


def k1_soccer_push_critic_obs_dim(num_action: int = K1_NUM_ACTION) -> int:
    return k1_soccer_push_actor_obs_dim(num_action) + 3


# Soccer curriculum layout (scene_soccer_dribble_minimal.xml): colinear +X axis.
K1_SOCCER_CURRICULUM_START_XY = (0.0, 0.0)
K1_SOCCER_CURRICULUM_PHASE1_WAYPOINT_XY = (1.0, 0.0)
K1_SOCCER_CURRICULUM_BALL_SPAWN_XY = (1.2, 0.0)
K1_SOCCER_CURRICULUM_PHASE1_DISTANCE_M = 1.0
K1_SOCCER_CURRICULUM_PHASE2_DISTANCE_M = 0.2
# Sphere center ≈ keyframe trunk height; radius matches scene geom size (pass-through volume).
K1_SOCCER_PHASE1_WAYPOINT_Z = 0.5743699226900935
K1_SOCCER_PHASE1_WAYPOINT_SPHERE_RADIUS = 0.08
K1_SOCCER_PHASE1_WAYPOINT_MARKER_HIDDEN_Z = -30.0
K1_SOCCER_PHASE1_WAYPOINT_GEOM_PENDING = "phase1_waypoint_marker_pending"
K1_SOCCER_PHASE1_WAYPOINT_GEOM_REACHED = "phase1_waypoint_marker_reached"

# Reward terms active only after phase-1 waypoint is reached (phase 2 dribble).
K1_SOCCER_PHASE2_REWARD_KEYS: frozenset[str] = frozenset(
    {
        "ball_progress",
        "ball_keep",
        "ball_front",
        "ball_speed_match",
        "ball_approach",
        "ball_moving",
        "ball_still",
        "ball_lost",
        "ball_over_speed",
        "ball_orbit",
        "feet_inward_yaw",
        "feet_step_travel",
        "termination_bad",
    }
)

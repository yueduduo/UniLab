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
# Reset DR: uniform x offset in [-limit, +limit], y fixed at 0.
K1_SOCCER_RESET_X_OFFSET_M = 0.3
# Sphere center ≈ keyframe trunk height; radius matches scene geom size (pass-through volume).
K1_SOCCER_PHASE1_WAYPOINT_Z = 0.5743699226900935
K1_SOCCER_PHASE1_WAYPOINT_SPHERE_RADIUS = 0.18
# Phase-1 fail-safe: terminate if episode steps exceed this without reaching waypoint.
K1_SOCCER_PHASE1_MAX_STEPS = 200
K1_SOCCER_PHASE1_WAYPOINT_MARKER_HIDDEN_Z = -30.0
K1_SOCCER_PHASE1_WAYPOINT_GEOM_PENDING = "phase1_waypoint_marker_pending"
K1_SOCCER_PHASE1_WAYPOINT_GEOM_REACHED = "phase1_waypoint_marker_reached"

# Reward terms masked to phase 1 only (phase1_reach manages its own freeze logic).
K1_SOCCER_PHASE1_REWARD_KEYS: frozenset[str] = frozenset()

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

# ---------------------------------------------------------------------------
# Deploy-compatible joint order — matches MOS-SIM K1_JOINTS_POLICY_ORDER and
# the k1_model_46000.pt deployment contract.  The order differs from
# K1_ACTUATOR_JOINT_ORDER (MJCF actuator order used for backend I/O).
# ---------------------------------------------------------------------------
K1_DEPLOY_JOINT_ORDER: tuple[str, ...] = (
    "AAHead_yaw",           # 0
    "ALeft_Shoulder_Pitch", # 1
    "ARight_Shoulder_Pitch",# 2
    "Left_Hip_Pitch",       # 3
    "Right_Hip_Pitch",      # 4
    "Head_pitch",           # 5
    "Left_Shoulder_Roll",   # 6
    "Right_Shoulder_Roll",  # 7
    "Left_Hip_Roll",        # 8
    "Right_Hip_Roll",       # 9
    "Left_Elbow_Pitch",     # 10
    "Right_Elbow_Pitch",    # 11
    "Left_Hip_Yaw",         # 12
    "Right_Hip_Yaw",        # 13
    "Left_Elbow_Yaw",       # 14
    "Right_Elbow_Yaw",      # 15
    "Left_Knee_Pitch",      # 16
    "Right_Knee_Pitch",     # 17
    "Left_Ankle_Pitch",     # 18
    "Right_Ankle_Pitch",    # 19
    "Left_Ankle_Roll",      # 20
    "Right_Ankle_Roll",     # 21
)

# Permutation: K1_ACTUATOR_JOINT_ORDER → K1_DEPLOY_JOINT_ORDER (for obs/last_action).
# Usage: deploy_vec = actuator_vec[:, K1_ACTUATOR_TO_DEPLOY_PERM]
K1_ACTUATOR_TO_DEPLOY_PERM: tuple[int, ...] = (
    0, 2, 6, 10, 16, 1, 3, 7, 11, 17, 4, 8, 12, 18, 5, 9, 13, 19, 14, 20, 15, 21
)

# Inverse permutation: K1_DEPLOY_JOINT_ORDER → K1_ACTUATOR_JOINT_ORDER (for action control).
# Usage: actuator_actions = deploy_actions[:, K1_DEPLOY_TO_ACTUATOR_PERM]
K1_DEPLOY_TO_ACTUATOR_PERM: tuple[int, ...] = (
    0, 5, 1, 6, 10, 14, 2, 7, 11, 15, 3, 8, 12, 16, 18, 20, 4, 9, 13, 17, 19, 21
)


def k1_soccer_dribble_compat_actor_obs_dim(num_action: int = K1_NUM_ACTION) -> int:
    """46000-compat layout: linvel(3)+angvel(3)+gravity(3)+cmd(3)+joints(3n).

    Exactly 78 dims — identical to k1_model_46000 deployment contract.
    No ball obs in actor; ball direction is encoded via cmd only.
    """
    return 12 + 3 * num_action


def k1_soccer_dribble_compat_critic_obs_dim(num_action: int = K1_NUM_ACTION) -> int:
    """Critic = actor(78) + ball_rel_state(3) privileged for SAC value estimation."""
    return k1_soccer_dribble_compat_actor_obs_dim(num_action) + 3


def k1_vel_cmd_compat_actor_obs_dim(num_action: int = K1_NUM_ACTION) -> int:
    """Same 78-dim actor layout as K1SoccerDribbleCompat / k1_model_46000 deploy contract."""
    return k1_soccer_dribble_compat_actor_obs_dim(num_action)


def k1_vel_cmd_compat_critic_obs_dim(num_action: int = K1_NUM_ACTION) -> int:
    """Critic = actor(78) + privileged base_lin_vel(3) for SAC value estimation."""
    return k1_vel_cmd_compat_actor_obs_dim(num_action) + 3

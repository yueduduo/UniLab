"""K1 locomotion constants aligned with sim_soccer2 AMP deploy contract."""

from __future__ import annotations

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


def k1_walk_actor_obs_dim(num_action: int = K1_NUM_ACTION) -> int:
    """G1 walk profile: gyro(3)+gravity(3)+joint(3n)+cmd(3)+phase(2)."""
    return 6 + 3 * num_action + 5


def k1_walk_critic_obs_dim(num_action: int = K1_NUM_ACTION) -> int:
    return k1_walk_actor_obs_dim(num_action) + 3

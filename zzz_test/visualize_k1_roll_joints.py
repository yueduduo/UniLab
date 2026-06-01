from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
import mujoco
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCENE_XML = REPO_ROOT / "src" / "unilab" / "assets" / "robots" / "k1" / "scene_soccer_dribble_minimal.xml"
OUT_DIR = Path(__file__).resolve().parent


def _joint_qpos_addr(model: mujoco.MjModel, joint_name: str) -> int:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        raise ValueError(f"Joint not found: {joint_name}")
    return int(model.jnt_qposadr[joint_id])


def _render_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    renderer: mujoco.Renderer,
    joint_qpos_addr: dict[str, int],
    joint_values: dict[str, float],
    output_path: Path,
) -> None:
    mujoco.mj_resetDataKeyframe(model, data, 0)  # stand
    for name, value in joint_values.items():
        data.qpos[joint_qpos_addr[name]] = value
    mujoco.mj_forward(model, data)

    cam = mujoco.MjvCamera()
    cam.lookat = np.array([0.0, 0.0, 0.68], dtype=np.float64)
    cam.distance = 2.8
    cam.azimuth = 180.0
    cam.elevation = -14.0

    renderer.update_scene(data, camera=cam)
    rgb = renderer.render()
    iio.imwrite(output_path, rgb)


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, width=640, height=480)

    roll_joints = (
        "Left_Hip_Roll",
        "Right_Hip_Roll",
        "Left_Ankle_Roll",
        "Right_Ankle_Roll",
    )
    qpos_addr = {name: _joint_qpos_addr(model, name) for name in roll_joints}
    qpos_addr["Left_Hip_Yaw"] = _joint_qpos_addr(model, "Left_Hip_Yaw")
    qpos_addr["Right_Hip_Yaw"] = _joint_qpos_addr(model, "Right_Hip_Yaw")

    inward_values = {
        "Left_Hip_Roll": -0.25,
        "Right_Hip_Roll": 0.25,
        "Left_Ankle_Roll": -0.18,
        "Right_Ankle_Roll": 0.18,
    }
    outward_values = {
        "Left_Hip_Roll": 0.25,
        "Right_Hip_Roll": -0.25,
        "Left_Ankle_Roll": 0.18,
        "Right_Ankle_Roll": -0.18,
    }

    inward_img = OUT_DIR / "k1_roll_inward_guess.png"
    outward_img = OUT_DIR / "k1_roll_outward_guess.png"
    left_sideways_img = OUT_DIR / "k1_one_foot_sideways_left.png"
    right_sideways_img = OUT_DIR / "k1_one_foot_sideways_right.png"
    left_toward_other_img = OUT_DIR / "k1_one_foot_toward_other_left.png"
    right_toward_other_img = OUT_DIR / "k1_one_foot_toward_other_right.png"
    both_toward_other_img = OUT_DIR / "k1_both_feet_toward_each_other.png"

    _render_pose(model, data, renderer, qpos_addr, inward_values, inward_img)
    _render_pose(model, data, renderer, qpos_addr, outward_values, outward_img)
    _render_pose(
        model,
        data,
        renderer,
        qpos_addr,
        {"Left_Hip_Yaw": 0.85, "Right_Hip_Yaw": 0.0},
        left_sideways_img,
    )
    _render_pose(
        model,
        data,
        renderer,
        qpos_addr,
        {"Left_Hip_Yaw": 0.0, "Right_Hip_Yaw": -0.85},
        right_sideways_img,
    )
    _render_pose(
        model,
        data,
        renderer,
        qpos_addr,
        {"Left_Hip_Yaw": -0.85, "Right_Hip_Yaw": 0.0},
        left_toward_other_img,
    )
    _render_pose(
        model,
        data,
        renderer,
        qpos_addr,
        {"Left_Hip_Yaw": 0.0, "Right_Hip_Yaw": 0.85},
        right_toward_other_img,
    )
    _render_pose(
        model,
        data,
        renderer,
        qpos_addr,
        {"Left_Hip_Yaw": -0.85, "Right_Hip_Yaw": 0.85},
        both_toward_other_img,
    )

    print("Rendered K1 roll-joint check images:")
    print(f"  inward_guess : {inward_img}")
    print(f"  outward_guess: {outward_img}")
    print(f"  left_sideways: {left_sideways_img}")
    print(f"  right_sideways: {right_sideways_img}")
    print(f"  left_toward_other: {left_toward_other_img}")
    print(f"  right_toward_other: {right_toward_other_img}")
    print(f"  both_toward_other: {both_toward_other_img}")
    print("Joint values used (other joints fixed to stand keyframe):")
    print(f"  inward_guess : {inward_values}")
    print(f"  outward_guess: {outward_values}")
    print("  left_sideways: {'Left_Hip_Yaw': 0.85, 'Right_Hip_Yaw': 0.0}")
    print("  right_sideways: {'Left_Hip_Yaw': 0.0, 'Right_Hip_Yaw': -0.85}")
    print("  left_toward_other: {'Left_Hip_Yaw': -0.85, 'Right_Hip_Yaw': 0.0}")
    print("  right_toward_other: {'Left_Hip_Yaw': 0.0, 'Right_Hip_Yaw': 0.85}")
    print("  both_toward_other: {'Left_Hip_Yaw': -0.85, 'Right_Hip_Yaw': 0.85}")


if __name__ == "__main__":
    main()

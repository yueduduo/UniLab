"""View K1 penalty-kick scene (stand keyframe) in MuJoCo.

Usage (repo root):
    uv run scripts/view_k1_penalty_kick_scene.py
    uv run scripts/view_k1_penalty_kick_scene.py --preview-only
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from unilab.envs.locomotion.k1.soccer_penalty_constants import (
    K1_PENALTY_SPOT_XY,
    K1_PENALTY_STANCE_FOOT_WIDTH_XY_M,
)

SCENE_XML = ROOT / "src/unilab/assets/robots/k1/scene_soccer_penalty_kick.xml"
PREVIEW_PNG = (
    ROOT
    / "conf/offpolicy/task/flashsac/k1_soccer_penalty_kick/figures/scene_preview.png"
)
LEFT_TARGET_PREVIEW_PNG = (
    ROOT
    / "conf/offpolicy/task/flashsac/k1_soccer_penalty_kick/figures/runup_reach_left_target_preview.png"
)


def _apply_stand_keyframe(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    data.qpos[:] = model.key_qpos[kid]
    mujoco.mj_forward(model, data)


def _sensor_pos(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray:
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    adr = model.sensor_adr[sid]
    return np.asarray(data.sensordata[adr : adr + 3], dtype=np.float64)


def _stance_foot_offset_xy(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    rf = _sensor_pos(model, data, "right_foot_pos")
    lf = _sensor_pos(model, data, "left_foot_pos")
    offset = lf[:2] - rf[:2]
    if float(np.linalg.norm(offset)) < 1e-6:
        return np.array([0.0, float(K1_PENALTY_STANCE_FOOT_WIDTH_XY_M)], dtype=np.float64)
    return offset


def _runup_reach_left_target_xy(ball_xy: np.ndarray, foot_offset_xy: np.ndarray) -> np.ndarray:
    return np.asarray(ball_xy[:2], dtype=np.float64).copy() + np.asarray(
        foot_offset_xy[:2], dtype=np.float64
    )


def _print_alignment(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    rf = _sensor_pos(model, data, "right_foot_pos")
    lf = _sensor_pos(model, data, "left_foot_pos")
    ball_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
    ball = np.asarray(data.xpos[ball_bid], dtype=np.float64)
    foot_offset = _stance_foot_offset_xy(model, data)
    target_xy = _runup_reach_left_target_xy(ball[:2], foot_offset)
    target = np.array([target_xy[0], target_xy[1], ball[2]], dtype=np.float64)
    foot_width = float(np.linalg.norm(foot_offset))
    ball_to_target = float(np.linalg.norm(target_xy - ball[:2]))
    print(
        f"[view_penalty] right_foot=({rf[0]:.3f}, {rf[1]:.3f}, {rf[2]:.3f}) "
        f"left_foot=({lf[0]:.3f}, {lf[1]:.3f}, {lf[2]:.3f}) "
        f"ball=({ball[0]:.3f}, {ball[1]:.3f}, {ball[2]:.3f}) "
        f"dy_right={rf[1] - ball[1]:.6f}"
    )
    print(
        f"[view_penalty] stance_foot_width_xy={foot_width:.4f} "
        f"runup_reach left target=({target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f}) "
        f"ball_to_target={ball_to_target:.4f} dist_left_to_target={np.linalg.norm(lf[:2] - target_xy):.3f}"
    )
    return target


def _add_scene_marker(scene: mujoco.MjvScene, pos: np.ndarray, rgba: np.ndarray, *, size: float) -> None:
    if scene.ngeom >= scene.maxgeom:
        raise RuntimeError("MuJoCo scene geom buffer is full.")
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        g,
        type=int(mujoco.mjtGeom.mjGEOM_SPHERE),
        size=np.asarray([size, 0.0, 0.0], dtype=np.float64),
        pos=np.asarray(pos, dtype=np.float64),
        mat=np.eye(3, dtype=np.float64).reshape(-1),
        rgba=np.asarray(rgba, dtype=np.float32),
    )
    scene.ngeom += 1


def _render_preview(model, data, out_path: Path, *, markers: list[tuple[np.ndarray, np.ndarray, float]] | None = None) -> None:
    import imageio.v3 as iio

    out_path.parent.mkdir(parents=True, exist_ok=True)
    renderer = mujoco.Renderer(model, 480, 640)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.lookat[:] = [2.2, 0.0, 0.55]
    cam.distance = 8.5
    cam.elevation = -18.0
    cam.azimuth = 120.0
    renderer.update_scene(data, camera=cam)
    if markers:
        for pos, rgba, size in markers:
            _add_scene_marker(renderer.scene, pos, rgba, size=size)
    iio.imwrite(out_path, renderer.render())
    renderer.close()
    print(f"[view_penalty] preview: {out_path}")


def _run_viewer(model, data) -> None:
    import mujoco.viewer

    print("[view_penalty] Opening MuJoCo viewer — Esc or close window to quit.")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(float(model.opt.timestep))


def main() -> int:
    parser = argparse.ArgumentParser(description="View K1 penalty-kick scene in MuJoCo.")
    parser.add_argument("--preview-only", action="store_true", help="Save PNG and exit.")
    args = parser.parse_args()

    if not SCENE_XML.is_file():
        print(f"Missing {SCENE_XML}", file=sys.stderr)
        return 1

    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    data = mujoco.MjData(model)
    _apply_stand_keyframe(model, data)
    left_target = _print_alignment(model, data)
    ball_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
    ball = np.asarray(data.xpos[ball_bid], dtype=np.float64)
    left_foot = _sensor_pos(model, data, "left_foot_pos")
    markers = [
        (ball, np.array([0.15, 0.85, 0.25, 0.95], dtype=np.float32), 0.045),
        (left_target, np.array([0.95, 0.15, 0.10, 0.95], dtype=np.float32), 0.05),
        (left_foot, np.array([0.20, 0.45, 0.95, 0.95], dtype=np.float32), 0.04),
    ]

    _render_preview(model, data, PREVIEW_PNG)
    _render_preview(model, data, LEFT_TARGET_PREVIEW_PNG, markers=markers)
    print(
        f"[view_penalty] penalty spot ref={K1_PENALTY_SPOT_XY} "
        f"left-target preview={LEFT_TARGET_PREVIEW_PNG}"
    )
    if args.preview_only:
        return 0
    _run_viewer(model, data)
    return 0


if __name__ == "__main__":
    sys.exit(main())

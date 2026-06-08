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

ROOT = Path(__file__).resolve().parents[1]
SCENE_XML = ROOT / "src/unilab/assets/robots/k1/scene_soccer_penalty_kick.xml"
PREVIEW_PNG = (
    ROOT
    / "conf/offpolicy/task/flashsac/k1_soccer_penalty_kick/figures/scene_preview.png"
)


def _apply_stand_keyframe(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    data.qpos[:] = model.key_qpos[kid]
    mujoco.mj_forward(model, data)


def _print_alignment(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "right_foot_pos")
    adr = model.sensor_adr[sid]
    rf = data.sensordata[adr : adr + 3]
    ball_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
    ball = data.xpos[ball_bid]
    print(
        f"[view_penalty] right_foot=({rf[0]:.3f}, {rf[1]:.3f}, {rf[2]:.3f}) "
        f"ball=({ball[0]:.3f}, {ball[1]:.3f}, {ball[2]:.3f}) "
        f"dy={rf[1] - ball[1]:.6f}"
    )


def _render_preview(model, data, out_path: Path) -> None:
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
    _print_alignment(model, data)

    _render_preview(model, data, PREVIEW_PNG)
    if args.preview_only:
        return 0
    _run_viewer(model, data)
    return 0


if __name__ == "__main__":
    sys.exit(main())

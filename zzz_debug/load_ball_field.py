"""仅加载足球 + 球场（无机器人），物理参数与带球任务场景一致。

用法:
  uv run zzz_debug/load_ball_field.py --visual --until-close
  uv run zzz_debug/load_ball_field.py --steps 500
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import motrixsim as mtx
import numpy as np
from motrixsim.render import RenderApp, RenderSettings

DEBUG_DIR = Path(__file__).resolve().parent
SCENE_XML = DEBUG_DIR / "scene_ball_pitch.xml"
SIM_DT = 0.002
CAM_DISTANCE = 6.0
CAM_ELEVATION = -20.0
CAM_AZIMUTH = 90.0
CAM_LOOKAT = (0.0, 0.0, 0.1)


def load_ball_pitch_scene() -> tuple[mtx.SceneModel, mtx.SceneData]:
    if not SCENE_XML.is_file():
        raise FileNotFoundError(SCENE_XML)
    model = mtx.load_model(str(SCENE_XML))
    data = mtx.SceneData(model)
    if model.keyframes:
        data.dof_pos[:] = model.keyframes[0].dof_pos
    else:
        data.dof_pos[:] = model.compute_init_dof_pos()
    data.dof_vel[:] = 0.0
    model.options.timestep = SIM_DT
    model.forward_kinematic(data)
    return model, data


def ball_pos(model: mtx.SceneModel, data: mtx.SceneData) -> np.ndarray:
    ball_link = model.get_link("ball")
    if ball_link is None:
        raise RuntimeError("scene 中缺少 link 'ball'")
    pose = ball_link.get_pose(data)
    return np.asarray(pose[:3], dtype=np.float64)


def print_scene_info(model: mtx.SceneModel, data: mtx.SceneData) -> None:
    pos = ball_pos(model, data)
    print(f"[zzz_debug] scene={SCENE_XML}")
    print(f"[zzz_debug] pitch_geom=COL_Collider ball_pos={pos.tolist()}")


def _launch_renderer(model: mtx.SceneModel) -> RenderApp:
    settings = RenderSettings.performance()
    settings.enable_shadow = True
    render = RenderApp(headless=False)
    render.launch(model, render_settings=settings)
    render.set_main_camera(None)
    render.system_camera.set_view(
        list(CAM_LOOKAT),
        CAM_DISTANCE,
        CAM_ELEVATION,
        CAM_AZIMUTH,
    )
    return render


def run_headless(model: mtx.SceneModel, data: mtx.SceneData, steps: int) -> None:
    print_scene_info(model, data)
    for _ in range(steps):
        model.step(data)
    pos = ball_pos(model, data)
    print(f"[zzz_debug] headless 完成 ({steps} steps) ball_pos={pos.tolist()}")


def run_visual(
    model: mtx.SceneModel,
    data: mtx.SceneData,
    *,
    until_close: bool,
    steps: int,
) -> None:
    render = _launch_renderer(model)
    print_scene_info(model, data)
    play_steps = None if until_close else steps
    sim_s = "∞" if play_steps is None else f"{play_steps * SIM_DT:.1f}"
    print(f"[zzz_debug] Motrix 窗口 — 仅球和球场，关窗退出 (~{sim_s}s sim)")

    steps_run = 0
    frame_dt = 1.0 / 60.0
    last_render = time.perf_counter()
    try:
        while not render.is_closed:
            if play_steps is not None and steps_run >= play_steps:
                break
            model.step(data)
            render.sync(data)
            steps_run += 1
            now = time.perf_counter()
            elapsed = now - last_render
            if elapsed < frame_dt:
                time.sleep(frame_dt - elapsed)
            last_render = time.perf_counter()
    except KeyboardInterrupt:
        pass
    print("[zzz_debug] 渲染结束")


def main() -> int:
    parser = argparse.ArgumentParser(description="Load ball + pitch scene (no robot)")
    parser.add_argument("--visual", action="store_true", help="打开 Motrix 3D 窗口")
    parser.add_argument(
        "--until-close",
        action="store_true",
        help="配合 --visual：关窗后退出",
    )
    parser.add_argument("--steps", type=int, default=2000, help="无窗口模式下的仿真步数")
    args = parser.parse_args()

    if args.until_close and not args.visual:
        raise ValueError("--until-close 需要 --visual")

    model, data = load_ball_pitch_scene()
    if args.visual:
        run_visual(model, data, until_close=args.until_close, steps=args.steps)
    else:
        run_headless(model, data, args.steps)

    print("[zzz_debug] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

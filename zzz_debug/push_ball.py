"""鼠标 / 快捷键对球施力，观察 Motrix 物理反应（无机器人）。

控制（先点击渲染窗口获得焦点）:
  滚轮按下           锁定力的起点（地面位置）
  按住滚轮           力线随鼠标移动：起点固定，终点跟随鼠标
  滚轮松开           沿「起点 → 松开点」方向施加冲量（大小 ∝ 拖动距离）
  左键             仅旋转视角
  R                  复位球到 keyframe
  P                  暂停 / 继续仿真
  J                  相机一次性切到球附近斜视角（不跟踪）

可视化:
  黄色球  力的起点（按下时锁定）
  青色球  当前鼠标终点（拖动中跟随）
  橙/红箭 起点 → 终点 力线
  灰线    同上，辅助线段
  top-left overlay  live ball velocity (vx/vy/vz, |v|, |v_xy|)

用法:
  uv run zzz_debug/push_ball.py
  uv run zzz_debug/push_ball.py --impulse 2.0
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import motrixsim as mtx
import numpy as np
from motrixsim.render import Color, RenderApp
from unilab.visualization.ball_velocity_hud import BallVelocityHud

from load_ball_field import (
    CAM_AZIMUTH,
    CAM_DISTANCE,
    CAM_ELEVATION,
    CAM_LOOKAT,
    SIM_DT,
    _launch_renderer,
    ball_pos,
    load_ball_pitch_scene,
)
from ground_pick import (
    CameraView,
    DEFAULT_PICK_CALIBRATION,
    PickCalibration,
    calibrate_pick_from_ball,
    resolve_ground_target,
)
from mouse_x11 import close_mouse_reader, poll_mouse_pointer, x11_mouse_available

GROUND_Z = 0.0
DEFAULT_IMPULSE_PER_M = 2.0
MIN_DRAG_DIST = 0.05
IMPULSE_VIS_FRAMES = 45
TARGET_MARKER_RADIUS = 0.045
CONTACT_MARKER_RADIUS = 0.035
BALL_VIEW_DISTANCE = 4.5
BALL_VIEW_ELEVATION = -35.0
BALL_VIEW_AZIMUTH = 120.0
_MOTRIX_WHEEL_NAMES = ("wheel", "middle", "button2", "2")
DEFAULT_CAMERA = CameraView(
    lookat=CAM_LOOKAT,
    distance=CAM_DISTANCE,
    elevation=CAM_ELEVATION,
    azimuth=CAM_AZIMUTH,
)
BALL_CAMERA = CameraView(
    lookat=(0.0, 0.0, 0.1),
    distance=BALL_VIEW_DISTANCE,
    elevation=BALL_VIEW_ELEVATION,
    azimuth=BALL_VIEW_AZIMUTH,
)


@dataclass
class ForceVisual:
    show: bool = False
    force_start: np.ndarray | None = None
    force_end: np.ndarray | None = None
    direction: np.ndarray | None = None
    magnitude: float = 0.0
    kind: str = "none"


def ball_lin_vel(model: mtx.SceneModel, data: mtx.SceneData) -> np.ndarray:
    ball_link = model.get_link("ball")
    if ball_link is None:
        raise RuntimeError("scene 中缺少 link 'ball'")
    return np.asarray(ball_link.get_linear_velocity(data), dtype=np.float64).reshape(3)


def segment_horizontal_delta(force_start: np.ndarray, force_end: np.ndarray) -> np.ndarray:
    delta = force_end - force_start
    delta[2] = 0.0
    return delta


def segment_horizontal_dist(force_start: np.ndarray, force_end: np.ndarray) -> float:
    return float(np.linalg.norm(segment_horizontal_delta(force_start, force_end)))


def segment_horizontal_dir(force_start: np.ndarray, force_end: np.ndarray) -> np.ndarray | None:
    dist = segment_horizontal_dist(force_start, force_end)
    if dist < 1e-4:
        return None
    return segment_horizontal_delta(force_start, force_end) / dist


def impulse_from_drag(
    force_start: np.ndarray,
    force_end: np.ndarray,
    *,
    impulse_per_m: float,
) -> float:
    drag_dist = segment_horizontal_dist(force_start, force_end)
    if drag_dist < MIN_DRAG_DIST:
        return 0.0
    return float(impulse_per_m) * drag_dist


def _ground_marker(point: np.ndarray) -> np.ndarray:
    marker = point.copy()
    marker[2] = GROUND_Z + 0.01
    return marker


def build_segment_force_visual(
    *,
    force_start: np.ndarray,
    force_end: np.ndarray,
    magnitude: float,
    kind: str,
) -> ForceVisual:
    start = _ground_marker(force_start)
    end = _ground_marker(force_end)
    direction = segment_horizontal_dir(start, end)
    if direction is None:
        return ForceVisual(show=True, force_start=start, force_end=end, kind=kind)
    return ForceVisual(
        show=True,
        force_start=start,
        force_end=end,
        direction=direction.copy(),
        magnitude=float(magnitude),
        kind=kind,
    )


def reset_ball(model: mtx.SceneModel, data: mtx.SceneData) -> None:
    if model.keyframes:
        data.dof_pos[:] = model.keyframes[0].dof_pos
    else:
        data.dof_pos[:] = model.compute_init_dof_pos()
    data.set_dof_vel(np.zeros_like(np.asarray(data.dof_vel, dtype=np.float32)))
    model.forward_kinematic(data)


def apply_segment_impulse(
    model: mtx.SceneModel,
    data: mtx.SceneData,
    *,
    force_start: np.ndarray,
    force_end: np.ndarray,
    impulse: float,
) -> ForceVisual:
    visual = build_segment_force_visual(
        force_start=force_start,
        force_end=force_end,
        magnitude=impulse,
        kind="impulse",
    )
    if visual.direction is None:
        return visual
    ball_link = model.get_link("ball")
    if ball_link is None:
        raise RuntimeError("scene 中缺少 link 'ball'")
    mass = float(ball_link.mass)
    delta_v = visual.direction * (float(impulse) / mass)
    qvel = np.asarray(data.dof_vel, dtype=np.float32).reshape(-1).copy()
    qvel[:3] += delta_v.astype(np.float32)
    data.set_dof_vel(qvel)
    return visual


def _as_xyz(point: np.ndarray) -> list[float]:
    return [float(point[0]), float(point[1]), float(point[2])]


def _poll_wheel_state(pointer) -> tuple[bool, bool, bool, str]:
    if pointer is not None:
        return pointer.held.middle, pointer.pressed.middle, pointer.released.middle, "x11"
    return False, False, False, "none"


def _poll_wheel_state_with_motrix(
    inp,
    pointer,
    *,
    prev_held: bool,
) -> tuple[bool, bool, bool, str]:
    held, pressed, released, source = _poll_wheel_state(pointer)
    if held or pressed or released:
        return held, pressed, released, source
    motrix_held = any(inp.is_key_pressed(name) for name in _MOTRIX_WHEEL_NAMES)
    motrix_pressed = any(inp.is_mouse_just_pressed(name) for name in _MOTRIX_WHEEL_NAMES) or any(
        inp.is_key_just_pressed(name) for name in _MOTRIX_WHEEL_NAMES
    )
    motrix_released = prev_held and not motrix_held
    return motrix_held, motrix_pressed, motrix_released, "motrix"


def draw_force_visualization(render: RenderApp, visual: ForceVisual) -> None:
    if not visual.show:
        return

    gizmos = render.gizmos
    if visual.force_start is not None:
        gizmos.draw_sphere(
            CONTACT_MARKER_RADIUS,
            _as_xyz(visual.force_start),
            Color.rgb(1.0, 0.9, 0.15),
        )
    if visual.force_end is not None:
        gizmos.draw_sphere(
            TARGET_MARKER_RADIUS,
            _as_xyz(visual.force_end),
            Color.rgb(0.15, 0.85, 1.0),
        )
    if visual.force_start is not None and visual.force_end is not None:
        gizmos.draw_line(
            _as_xyz(visual.force_start),
            _as_xyz(visual.force_end),
            Color.rgb(0.55, 0.55, 0.55),
        )
        if visual.kind == "impulse":
            arrow_color = Color.rgb(1.0, 0.15, 0.15)
        else:
            arrow_color = Color.rgb(0.95, 0.75, 0.1)
        gizmos.draw_arrow(
            _as_xyz(visual.force_start),
            _as_xyz(visual.force_end),
            arrow_color,
        )


def set_camera_on_ball(render: RenderApp, model: mtx.SceneModel, data: mtx.SceneData) -> CameraView:
    pos = ball_pos(model, data)
    view = CameraView(
        lookat=(float(pos[0]), float(pos[1]), float(pos[2])),
        distance=BALL_CAMERA.distance,
        elevation=BALL_CAMERA.elevation,
        azimuth=BALL_CAMERA.azimuth,
    )
    render.set_main_camera(None)
    render.system_camera.set_view(
        list(view.lookat),
        view.distance,
        view.elevation,
        view.azimuth,
    )
    return view


def _print_status(prefix: str, model: mtx.SceneModel, data: mtx.SceneData) -> None:
    pos = ball_pos(model, data)
    vel = ball_lin_vel(model, data)
    speed = float(np.linalg.norm(vel[:2]))
    print(
        f"{prefix} ball_pos=[{pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f}] "
        f"vel=[{vel[0]:+.3f}, {vel[1]:+.3f}, {vel[2]:+.3f}] |v_xy|={speed:.3f}"
    )


def _refresh_pick_calibration(
    render: RenderApp,
    model: mtx.SceneModel,
    data: mtx.SceneData,
    pointer,
) -> PickCalibration:
    if pointer is None or pointer.window_w <= 0 or pointer.window_h <= 0:
        return DEFAULT_PICK_CALIBRATION
    ball_world = ball_pos(model, data)
    return calibrate_pick_from_ball(
        render,
        data,
        ball_world=ball_world,
        window_w=pointer.window_w,
        window_h=pointer.window_h,
    )


def run_interactive(
    model: mtx.SceneModel,
    data: mtx.SceneData,
    *,
    impulse: float,
) -> None:
    render = _launch_renderer(model)
    render.opt.set_left_panel_vis(False)
    paused = False
    steps_run = 0
    impulse_frames_left = 0
    active_visual = ForceVisual()
    preview_visual = ForceVisual()
    camera_view = DEFAULT_CAMERA
    force_start: np.ndarray | None = None
    wheel_dragging = False
    wheel_was_held = False
    frame_dt = 1.0 / 60.0
    pick_calibration = DEFAULT_PICK_CALIBRATION
    last_calib_pose: np.ndarray | None = None

    print(
        f"[push_ball] 控制: 滚轮按下=锁定起点 | 按住=力线预览 | 松开=施力 "
        f"(冲量 = 拖动距离 × {impulse:.2f} N·s/m)"
    )
    if x11_mouse_available():
        print("[push_ball] 鼠标检测: Linux X11 直读滚轮(Button2)，不依赖 Motrix Input")
    print("[push_ball] 可视化: 黄=起点 | 青=当前终点 | 箭=力方向")
    _print_status("[push_ball] init", model, data)

    render.sync(data)
    init_pointer = poll_mouse_pointer()
    pick_calibration = _refresh_pick_calibration(render, model, data, init_pointer)
    last_calib_pose = np.asarray(render.system_camera.pose, dtype=np.float64).copy()
    velocity_hud = BallVelocityHud(render)
    velocity_hud.update(ball_lin_vel(model, data))

    try:
        while not render.is_closed:
            inp = render.input
            loop_start = time.perf_counter()

            if inp.is_key_just_pressed("r"):
                reset_ball(model, data)
                active_visual = ForceVisual()
                preview_visual = ForceVisual()
                impulse_frames_left = 0
                force_start = None
                wheel_dragging = False
                wheel_was_held = False
                pick_calibration = _refresh_pick_calibration(render, model, data, poll_mouse_pointer())
                last_calib_pose = np.asarray(render.system_camera.pose, dtype=np.float64).copy()
                print("[push_ball] reset ball")
                _print_status("[push_ball]", model, data)

            if inp.is_key_just_pressed("p"):
                paused = not paused
                print(f"[push_ball] {'paused' if paused else 'running'}")

            if inp.is_key_just_pressed("j"):
                camera_view = set_camera_on_ball(render, model, data)
                render.sync(data)
                pick_calibration = _refresh_pick_calibration(render, model, data, poll_mouse_pointer())
                last_calib_pose = np.asarray(render.system_camera.pose, dtype=np.float64).copy()
                print(f"[push_ball] 相机切到球附近 lookat={list(camera_view.lookat)}")

            pointer = poll_mouse_pointer()
            wheel_held, wheel_press, wheel_release, wheel_src = _poll_wheel_state_with_motrix(
                inp,
                pointer,
                prev_held=wheel_was_held,
            )
            if wheel_press:
                cam_pose = np.asarray(render.system_camera.pose, dtype=np.float64)
                if last_calib_pose is None or float(np.linalg.norm(cam_pose - last_calib_pose)) > 1e-3:
                    pick_calibration = _refresh_pick_calibration(render, model, data, pointer)
                    last_calib_pose = cam_pose.copy()
            live_target = resolve_ground_target(
                render,
                inp,
                camera_view,
                win_x=None if pointer is None else float(pointer.win_x),
                win_y=None if pointer is None else float(pointer.win_y),
                window_w=0 if pointer is None else pointer.window_w,
                window_h=0 if pointer is None else pointer.window_h,
                calibration=pick_calibration,
            )

            if wheel_press:
                force_start = live_target.copy()
                wheel_dragging = True
                print(f"[push_ball] 锁定起点 via {wheel_src}: {force_start.tolist()}")

            if wheel_dragging and wheel_held and force_start is not None:
                preview_magnitude = impulse_from_drag(
                    force_start,
                    live_target,
                    impulse_per_m=impulse,
                )
                preview_visual = build_segment_force_visual(
                    force_start=force_start,
                    force_end=live_target,
                    magnitude=preview_magnitude,
                    kind="aim",
                )
            else:
                preview_visual = ForceVisual()

            if wheel_release and wheel_dragging and force_start is not None:
                force_end = live_target.copy()
                drag_dist = segment_horizontal_dist(force_start, force_end)
                release_impulse = impulse_from_drag(
                    force_start,
                    force_end,
                    impulse_per_m=impulse,
                )
                if release_impulse > 0.0:
                    impulse_visual = apply_segment_impulse(
                        model,
                        data,
                        force_start=force_start,
                        force_end=force_end,
                        impulse=release_impulse,
                    )
                    active_visual = impulse_visual
                    impulse_frames_left = IMPULSE_VIS_FRAMES
                    print(
                        f"[push_ball] 松开施力 via {wheel_src} "
                        f"drag={drag_dist:.3f}m impulse={release_impulse:.3f}N·s "
                        f"start={force_start.tolist()} end={force_end.tolist()}"
                    )
                    _print_status("[push_ball]", model, data)
                else:
                    print(
                        f"[push_ball] 拖动过短 ({drag_dist:.3f}m < {MIN_DRAG_DIST}m)，未施力 "
                        f"start={force_start.tolist()} end={force_end.tolist()}"
                    )
                wheel_dragging = False
                force_start = None

            wheel_was_held = wheel_held

            if not paused:
                model.step(data)
                steps_run += 1

            if impulse_frames_left > 0:
                impulse_frames_left -= 1
            elif active_visual.kind == "impulse":
                active_visual = ForceVisual()

            if wheel_dragging and wheel_held and preview_visual.show:
                visual_to_draw = preview_visual
            else:
                visual_to_draw = active_visual if active_visual.show else preview_visual
            draw_force_visualization(render, visual_to_draw)
            velocity_hud.update(ball_lin_vel(model, data))
            render.sync(data)
            elapsed = time.perf_counter() - loop_start
            if elapsed < frame_dt:
                time.sleep(frame_dt - elapsed)
    except KeyboardInterrupt:
        pass
    finally:
        close_mouse_reader()

    print("[push_ball] 渲染结束")


def main() -> int:
    parser = argparse.ArgumentParser(description="Mouse/keyboard force on ball (no robot)")
    parser.add_argument(
        "--impulse",
        type=float,
        default=DEFAULT_IMPULSE_PER_M,
        help="冲量系数：松开时冲量 = 水平拖动距离(m) × 该值 (N·s/m)",
    )
    args = parser.parse_args()

    model, data = load_ball_pitch_scene()
    print(f"[push_ball] sim_dt={SIM_DT}")
    run_interactive(model, data, impulse=args.impulse)
    print("[push_ball] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""从 Motrix 相机 + 鼠标位置解析地面施力目标点。"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from motrixsim.render import RenderApp

from unilab.envs.common.rotation import np_quat_apply

GROUND_Z = 0.0
CAMERA_VFOV_DEG = 45.0
# Motrix 实际垂直 FOV 与 pose 推导值的比例（由截屏中足球位置标定）
MOTRIX_VFOV_SCALE = 0.518


@dataclass(frozen=True)
class CameraView:
    lookat: tuple[float, float, float]
    distance: float
    elevation: float
    azimuth: float


@dataclass(frozen=True)
class PickCalibration:
    vfov_deg: float = CAMERA_VFOV_DEG
    scale_h: float = 1.0
    scale_v: float = 1.0
    # Motrix 视口 Y 轴：True 表示 ny=(py/h-0.5)*2，与 Bevy 窗口坐标一致
    origin_top: bool = True


DEFAULT_PICK_CALIBRATION = PickCalibration(
    scale_v=MOTRIX_VFOV_SCALE,
    origin_top=True,
)


@dataclass(frozen=True)
class _CameraBasis:
    cam_pos: np.ndarray
    forward: np.ndarray
    right: np.ndarray
    up: np.ndarray


def ray_ground_hit(ray: list[float] | tuple[float, ...], *, ground_z: float = GROUND_Z) -> np.ndarray | None:
    origin = np.asarray(ray[:3], dtype=np.float64)
    direction = np.asarray(ray[3:6], dtype=np.float64)
    dir_norm = float(np.linalg.norm(direction))
    if dir_norm < 1e-8:
        return None
    direction /= dir_norm
    if abs(direction[2]) < 1e-8:
        return None
    t = (ground_z - origin[2]) / direction[2]
    if t < 0.0:
        return None
    return origin + t * direction


def _normalize(vec: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-8:
        return None
    return vec / norm


def _camera_ray_basis(render: RenderApp) -> _CameraBasis | None:
    pose = np.asarray(render.system_camera.pose, dtype=np.float64).reshape(7)
    cam_pos = pose[:3]
    if float(np.linalg.norm(cam_pos)) < 1e-4:
        return None
    quat_wxyz = np.array([pose[6], pose[3], pose[4], pose[5]], dtype=np.float64)
    forward = _normalize(np_quat_apply(quat_wxyz, np.array([0.0, 0.0, -1.0])))
    right = _normalize(np_quat_apply(quat_wxyz, np.array([1.0, 0.0, 0.0])))
    up = _normalize(np_quat_apply(quat_wxyz, np.array([0.0, 1.0, 0.0])))
    if forward is None or right is None or up is None:
        return None
    return _CameraBasis(
        cam_pos=cam_pos,
        forward=forward,
        right=right,
        up=up,
    )


def _pixel_ndc(
    win_x: float,
    win_y: float,
    *,
    window_w: int,
    window_h: int,
    calibration: PickCalibration,
) -> tuple[float, float]:
    nx = (win_x / float(max(window_w, 1)) - 0.5) * 2.0
    if calibration.origin_top:
        ny = (win_y / float(max(window_h, 1)) - 0.5) * 2.0
    else:
        ny = (0.5 - win_y / float(max(window_h, 1))) * 2.0
    return nx, ny


def _tan_half_fov(calibration: PickCalibration, *, window_w: int, window_h: int) -> tuple[float, float]:
    aspect = float(window_w) / float(max(window_h, 1))
    tan_half_v = float(np.tan(np.deg2rad(calibration.vfov_deg) / 2.0)) * calibration.scale_v
    tan_half_h = float(np.tan(np.deg2rad(calibration.vfov_deg) / 2.0)) * calibration.scale_h * aspect
    return tan_half_h, tan_half_v


def _ray_from_window_coords(
    basis: _CameraBasis,
    *,
    win_x: float,
    win_y: float,
    window_w: int,
    window_h: int,
    calibration: PickCalibration,
) -> list[float] | None:
    nx, ny = _pixel_ndc(
        win_x,
        win_y,
        window_w=window_w,
        window_h=window_h,
        calibration=calibration,
    )
    tan_half_h, tan_half_v = _tan_half_fov(calibration, window_w=window_w, window_h=window_h)
    ray_dir = _normalize(
        basis.forward + nx * tan_half_h * basis.right + ny * tan_half_v * basis.up
    )
    if ray_dir is None:
        return None
    return [
        basis.cam_pos[0],
        basis.cam_pos[1],
        basis.cam_pos[2],
        ray_dir[0],
        ray_dir[1],
        ray_dir[2],
    ]


def _capture_system_camera(render: RenderApp, data, *, max_wait: int = 200):
    task = render.system_camera.capture()
    for _ in range(max_wait):
        render.sync(data, wait=True)
        if task.state == "done":
            return task.take_image()
        time.sleep(0.005)
    return None


def _detect_ball_pixel(pixels: np.ndarray, *, window_w: int, window_h: int) -> tuple[float, float] | None:
    rgb = np.asarray(pixels[:, :, :3], dtype=np.float64)
    lum = rgb.mean(axis=2)
    green_dom = rgb[:, :, 1] - rgb[:, :, 0] > 25.0
    cand = (~green_dom) & (lum > 80.0)
    ys, xs = np.nonzero(cand)
    if len(xs) == 0:
        return None
    cx = float(window_w) * 0.5
    cy = float(window_h) * 0.5
    dist2 = (xs.astype(np.float64) - cx) ** 2 + (ys.astype(np.float64) - cy) ** 2
    keep = np.argsort(dist2)[: min(4000, len(dist2))]
    return float(np.mean(xs[keep])), float(np.mean(ys[keep]))


def _solve_pick_scales(
    basis: _CameraBasis,
    *,
    world_point: np.ndarray,
    target_px: tuple[float, float],
    window_w: int,
    window_h: int,
    vfov_deg: float,
    origin_top: bool,
) -> tuple[float, float] | None:
    rel = world_point - basis.cam_pos
    cz = float(np.dot(rel, basis.forward))
    cx = float(np.dot(rel, basis.right))
    cy = float(np.dot(rel, basis.up))
    if cz <= 1e-6:
        return None
    px, py = target_px
    nx_tgt = (px / float(window_w) - 0.5) * 2.0
    if origin_top:
        ny_tgt = (py / float(window_h) - 0.5) * 2.0
    else:
        ny_tgt = (0.5 - py / float(window_h)) * 2.0
    aspect = float(window_w) / float(window_h)
    base = float(np.tan(np.deg2rad(vfov_deg) / 2.0))
    scale_h = 1.0
    scale_v = 1.0
    if abs(nx_tgt) >= 0.02:
        scale_h = (cx / cz / nx_tgt) / (base * aspect)
    if abs(ny_tgt) >= 0.02:
        scale_v = (cy / cz / ny_tgt) / base
    if scale_h <= 0.0 or scale_v <= 0.0:
        return None
    return scale_h, scale_v


def calibrate_pick_from_ball(
    render: RenderApp,
    data,
    *,
    ball_world: np.ndarray,
    window_w: int,
    window_h: int,
) -> PickCalibration:
    image = _capture_system_camera(render, data)
    if image is None:
        print("[ground_pick] 校准跳过：截屏失败，使用默认投影")
        return DEFAULT_PICK_CALIBRATION

    ball_px = _detect_ball_pixel(np.asarray(image.pixels), window_w=window_w, window_h=window_h)
    if ball_px is None:
        print("[ground_pick] 校准跳过：未检测到球，使用默认投影")
        return DEFAULT_PICK_CALIBRATION

    basis = _camera_ray_basis(render)
    if basis is None:
        print("[ground_pick] 校准跳过：相机 pose 无效，使用默认投影")
        return DEFAULT_PICK_CALIBRATION

    best: PickCalibration | None = None
    best_score = float("inf")
    for origin_top in (True, False):
        scales = _solve_pick_scales(
            basis,
            world_point=ball_world,
            target_px=ball_px,
            window_w=window_w,
            window_h=window_h,
            vfov_deg=CAMERA_VFOV_DEG,
            origin_top=origin_top,
        )
        if scales is None:
            continue
        scale_h, scale_v = scales
        if abs((ball_px[1] / float(window_h)) - 0.5) < 0.05:
            scale_v = MOTRIX_VFOV_SCALE
        if not (0.4 <= scale_h <= 1.6 and 0.4 <= scale_v <= 1.6):
            continue
        score = abs(scale_h - 1.0) + abs(scale_v - MOTRIX_VFOV_SCALE)
        calib = PickCalibration(
            vfov_deg=CAMERA_VFOV_DEG,
            scale_h=float(scale_h),
            scale_v=float(scale_v),
            origin_top=origin_top,
        )
        if score < best_score:
            best_score = score
            best = calib

    if best is None:
        print("[ground_pick] 校准跳过：无法求解 FOV 比例，使用默认投影")
        return DEFAULT_PICK_CALIBRATION

    print(
        "[ground_pick] 已校准拾取: "
        f"ball_px=({ball_px[0]:.1f},{ball_px[1]:.1f}) "
        f"scale_h={best.scale_h:.3f} scale_v={best.scale_v:.3f} "
        f"origin_top={best.origin_top}"
    )
    return best


def orbit_camera_position(view: CameraView) -> np.ndarray:
    lookat = np.asarray(view.lookat, dtype=np.float64)
    el = np.deg2rad(view.elevation)
    az = np.deg2rad(view.azimuth)
    rel_xy = view.distance * np.cos(el)
    rel_z = view.distance * np.sin(el)
    offset = np.array([rel_xy * np.cos(az), rel_xy * np.sin(az), -rel_z], dtype=np.float64)
    return lookat + offset


def ground_target_from_live_camera(
    render: RenderApp,
    *,
    win_x: float,
    win_y: float,
    window_w: int,
    window_h: int,
    calibration: PickCalibration = DEFAULT_PICK_CALIBRATION,
) -> np.ndarray | None:
    basis = _camera_ray_basis(render)
    if basis is None:
        return None
    ray = _ray_from_window_coords(
        basis,
        win_x=win_x,
        win_y=win_y,
        window_w=window_w,
        window_h=window_h,
        calibration=calibration,
    )
    if ray is None:
        return None
    return ray_ground_hit(ray)


def ground_target_from_view(
    view: CameraView,
    *,
    win_x: float,
    win_y: float,
    window_w: int,
    window_h: int,
    calibration: PickCalibration = DEFAULT_PICK_CALIBRATION,
) -> np.ndarray | None:
    cam_pos = orbit_camera_position(view)
    lookat = np.asarray(view.lookat, dtype=np.float64)
    forward = _normalize(lookat - cam_pos)
    if forward is None:
        return None
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    right = _normalize(np.cross(forward, world_up))
    if right is None:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    up = np.cross(right, forward)
    basis = _CameraBasis(cam_pos=cam_pos, forward=forward, right=right, up=up)
    ray = _ray_from_window_coords(
        basis,
        win_x=win_x,
        win_y=win_y,
        window_w=window_w,
        window_h=window_h,
        calibration=calibration,
    )
    if ray is None:
        return None
    return ray_ground_hit(ray)


def ground_target_from_camera_pose(render: RenderApp) -> np.ndarray | None:
    basis = _camera_ray_basis(render)
    if basis is None:
        return None
    ray = [
        basis.cam_pos[0],
        basis.cam_pos[1],
        basis.cam_pos[2],
        basis.forward[0],
        basis.forward[1],
        basis.forward[2],
    ]
    return ray_ground_hit(ray)


def resolve_ground_target(
    render: RenderApp,
    inp,
    view: CameraView,
    *,
    win_x: float | None,
    win_y: float | None,
    window_w: int,
    window_h: int,
    calibration: PickCalibration = DEFAULT_PICK_CALIBRATION,
) -> np.ndarray:
    if win_x is not None and win_y is not None and window_w > 0 and window_h > 0:
        hit = ground_target_from_live_camera(
            render,
            win_x=win_x,
            win_y=win_y,
            window_w=window_w,
            window_h=window_h,
            calibration=calibration,
        )
        if hit is not None:
            return hit

    motrix_ray = inp.mouse_ray()
    if max(abs(float(v)) for v in motrix_ray) > 1e-6:
        hit = ray_ground_hit(motrix_ray)
        if hit is not None:
            return hit

    if win_x is not None and win_y is not None and window_w > 0 and window_h > 0:
        hit = ground_target_from_view(
            view,
            win_x=win_x,
            win_y=win_y,
            window_w=window_w,
            window_h=window_h,
            calibration=calibration,
        )
        if hit is not None:
            return hit

    hit = ground_target_from_camera_pose(render)
    if hit is not None:
        return hit

    return np.array([view.lookat[0], view.lookat[1], GROUND_Z], dtype=np.float64)

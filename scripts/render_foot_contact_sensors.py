#!/usr/bin/env python3
"""Render G1/K1 feet and foot-contact sensor geoms for layout verification."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "logs" / "foot_contact_sensor_compare"
SCENES = {
    "g1": REPO_ROOT / "src/unilab/assets/robots/g1/scene_flat.xml",
    "k1": REPO_ROOT / "src/unilab/assets/robots/k1/scene_flat.xml",
}

CONTACT_GEOM_SUFFIXES = tuple(f"_{i}_geom" for i in range(4))
FOOT_SITE_NAMES = ("left_foot", "right_foot")


def _apply_stand_pose(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    for name in ("stand", "home"):
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, name)
        if key_id >= 0:
            mujoco.mj_resetDataKeyframe(model, data, key_id)
            return
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)


def _highlight_foot_sensors(model: mujoco.MjModel) -> dict[str, list[str]]:
    """Make contact spheres visible; return grouped geom names for legend."""
    groups: dict[str, list[str]] = {"left": [], "right": []}
    palette = [
        [1.0, 0.15, 0.15, 1.0],
        [1.0, 0.55, 0.1, 1.0],
        [0.2, 0.85, 0.25, 1.0],
        [0.2, 0.45, 1.0, 1.0],
    ]
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        body_id = int(model.geom_bodyid[geom_id])
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        if (
            ("foot" in body_name or "ankle_roll" in body_name)
            and "foot_contact_" not in name
        ):
            model.geom_rgba[geom_id] = [0.72, 0.75, 0.8, 0.28]
            continue
        if "foot_contact_" not in name or not name.endswith("_geom"):
            continue
        idx = int(name.split("_")[-2])
        side = "left" if name.startswith("left_") else "right"
        model.geom_rgba[geom_id] = palette[idx % 4]
        model.geom_size[geom_id, 0] = 0.022
        model.geom_type[geom_id] = mujoco.mjtGeom.mjGEOM_SPHERE
        model.geom_group[geom_id] = 1
        model.geom_contype[geom_id] = 0
        model.geom_conaffinity[geom_id] = 0
        groups[side].append(name)
    for site_id in range(model.nsite):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, site_id) or ""
        if name in FOOT_SITE_NAMES:
            model.site_rgba[site_id] = [1.0, 1.0, 0.0, 1.0]
            model.site_size[site_id, 0] = 0.025
    return groups


def _foot_center(model: mujoco.MjModel, data: mujoco.MjData, side: str) -> np.ndarray:
    site_name = f"{side}_foot"
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id < 0:
        raise ValueError(f"missing site {site_name!r}")
    return data.site_xpos[site_id].copy()


def _render_closeup(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    lookat: np.ndarray,
    distance: float = 0.55,
    azimuth: float = 90.0,
    elevation: float = -35.0,
    width: int = 640,
    height: int = 480,
) -> np.ndarray:
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = lookat
    camera.distance = distance
    camera.azimuth = azimuth
    camera.elevation = elevation

    option = mujoco.MjvOption()
    option.geomgroup[:] = 1
    option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
    option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = False

    renderer = mujoco.Renderer(model, height=height, width=width)
    renderer.update_scene(data, camera=camera, scene_option=option)
    return renderer.render()


def _annotate_panel(img: np.ndarray, title: str, geom_names: list[str]) -> np.ndarray:
    """Simple text strip at top using numpy (no cv2 dependency)."""
    from PIL import Image, ImageDraw, ImageFont

    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
        font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except OSError:
        font = ImageFont.load_default()
        font_sm = font
    draw.rectangle((0, 0, img.shape[1], 72), fill=(20, 20, 24))
    draw.text((12, 8), title, fill=(240, 240, 240), font=font)
    legend = "  ".join(
        f"[{i}] {n.replace('_geom', '').split('_')[-1]}" for i, n in enumerate(geom_names[:4])
    )
    draw.text((12, 36), f"contact: {legend}", fill=(200, 200, 200), font=font_sm)
    draw.text((12, 54), "yellow=foot site  R/G/B/C=contact_0..3", fill=(160, 160, 160), font=font_sm)
    return np.asarray(pil)


def render_robot(robot: str, xml_path: Path) -> list[Path]:
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    _apply_stand_pose(model, data)
    mujoco.mj_forward(model, data)
    groups = _highlight_foot_sensors(model)

    saved: list[Path] = []
    for side in ("left", "right"):
        lookat = _foot_center(model, data, side)
        img = _render_closeup(
            model,
            data,
            lookat=lookat,
            distance=0.42,
            azimuth=125.0 if side == "left" else 55.0,
            elevation=-25.0,
        )
        title = f"{robot.upper()} — {side} foot (4 contact sensors + site)"
        img = _annotate_panel(img, title, groups[side])
        out = OUT_DIR / f"{robot}_{side}_foot.png"
        imageio.imwrite(out, img)
        saved.append(out)

    # top-down comparison of both feet
    mid = 0.5 * (_foot_center(model, data, "left") + _foot_center(model, data, "right"))
    img_both = _render_closeup(
        model, data, lookat=mid, distance=1.25, azimuth=135.0, elevation=-85.0
    )
    img_both = _annotate_panel(
        img_both,
        f"{robot.upper()} — both feet (top-down)",
        groups["left"] + groups["right"],
    )
    out_both = OUT_DIR / f"{robot}_feet_topdown.png"
    imageio.imwrite(out_both, img_both)
    saved.append(out_both)
    return saved


def _compose_side_by_side(paths: list[Path], out_path: Path) -> None:
    from PIL import Image

    images = [Image.open(p) for p in paths]
    w = sum(im.width for im in images)
    h = max(im.height for im in images)
    canvas = Image.new("RGB", (w, h), (24, 24, 28))
    x = 0
    for im in images:
        canvas.paste(im, (x, 0))
        x += im.width
    canvas.save(out_path)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_paths: list[Path] = []
    for robot, xml in SCENES.items():
        all_paths.extend(render_robot(robot, xml))

    g1_left = OUT_DIR / "g1_left_foot.png"
    k1_left = OUT_DIR / "k1_left_foot.png"
    g1_right = OUT_DIR / "g1_right_foot.png"
    k1_right = OUT_DIR / "k1_right_foot.png"
    if all(p.exists() for p in (g1_left, k1_left)):
        _compose_side_by_side([g1_left, k1_left], OUT_DIR / "compare_left_foot.png")
    if all(p.exists() for p in (g1_right, k1_right)):
        _compose_side_by_side([g1_right, k1_right], OUT_DIR / "compare_right_foot.png")
    topdown = [OUT_DIR / "g1_feet_topdown.png", OUT_DIR / "k1_feet_topdown.png"]
    if all(p.exists() for p in topdown):
        _compose_side_by_side(topdown, OUT_DIR / "compare_feet_topdown.png")

    print(f"Wrote renders under: {OUT_DIR}")
    for p in sorted(OUT_DIR.glob("*.png")):
        print(f"  {p}")


if __name__ == "__main__":
    main()

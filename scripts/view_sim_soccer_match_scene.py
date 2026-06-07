"""View full sim_soccer2 match scene (world.xml + match_config markings) in MuJoCo.

Usage (repo root):
    uv run scripts/view_sim_soccer_match_scene.py
    uv run scripts/view_sim_soccer_match_scene.py --preview-only
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOCCER_DIR = ROOT / "sim_soccer2" / "simulation" / "motrixsim" / "assets" / "environments" / "soccer"
WORLD_XML = SOCCER_DIR / "world.xml"
MATCH_CONFIG = ROOT / "sim_soccer2" / "simulation" / "motrixsim" / "assets" / "config" / "match_config.json"
PREVIEW_PNG = (
    ROOT
    / "conf/offpolicy/task/flashsac/k1_soccer_penalty_kick/figures/sim_soccer_match_scene.png"
)

# MuJoCo-only attrs that appear in Motrix-exported MJCF.
_MUJOCO_STRIP_ATTRS = ("emissionintensity",)


def _sanitize_for_mujoco(root: ET.Element) -> None:
    for elem in root.iter():
        for attr in _MUJOCO_STRIP_ATTRS:
            elem.attrib.pop(attr, None)
        if elem.tag == "material":
            elem.attrib.pop("emission", None)
            for child in list(elem):
                elem.remove(child)
        if elem.tag == "compiler" and elem.get("angle") == "degree":
            elem.set("angle", "radian")

    worldbody = root.find("worldbody")
    if worldbody is not None:
        for light in list(worldbody.findall("light")):
            worldbody.remove(light)
        ET.SubElement(
            worldbody,
            "light",
            pos="0 0 8",
            dir="0 0 -1",
            directional="true",
            diffuse="0.8 0.8 0.8",
            ambient="0.2 0.2 0.2",
        )


def _resolve_asset_paths(root: ET.Element, asset_dir: Path) -> None:
    for elem in root.iter():
        for attr in ("file", "fileup", "filedown", "filefront", "fileback", "fileleft", "fileright"):
            val = elem.get(attr)
            if val and not Path(val).is_absolute():
                elem.set(attr, (asset_dir / val).as_posix())


def _load_markings_cfg() -> dict:
    data = json.loads(MATCH_CONFIG.read_text(encoding="utf-8"))
    field = data.get("field", {})
    mk = field.get("markings", {})
    if not isinstance(mk, dict):
        return {"enabled": False}
    return mk


def _add_field_markings(worldbody: ET.Element, cfg: dict) -> None:
    if not bool(cfg.get("enabled", False)):
        return

    line_w = max(0.005, float(cfg.get("line_width", 0.05)))
    line_h = max(0.0002, float(cfg.get("line_height", 0.001)))
    rgba = cfg.get("color", [1.0, 1.0, 1.0, 1.0])
    if not isinstance(rgba, (list, tuple)) or len(rgba) != 4:
        rgba = [1.0, 1.0, 1.0, 1.0]
    rgba_str = " ".join(f"{float(v):g}" for v in rgba)

    mark_len = max(0.1, float(cfg.get("field_length", 9.0)))
    mark_wid = max(0.1, float(cfg.get("field_width", 6.0)))
    half_len = 0.5 * mark_len
    half_wid = 0.5 * mark_wid
    half_lw = 0.5 * line_w
    z = line_h
    half_h = 0.5 * line_h

    def add_line_box(name: str, x: float, y: float, sx: float, sy: float) -> None:
        ET.SubElement(
            worldbody,
            "geom",
            name=name,
            type="box",
            pos=f"{x:g} {y:g} {z:g}",
            size=f"{max(half_lw, sx):g} {max(half_lw, sy):g} {half_h:g}",
            rgba=rgba_str,
            contype="0",
            conaffinity="0",
        )

    add_line_box("line-boundary-top", 0.0, half_wid, half_len, half_lw)
    add_line_box("line-boundary-bottom", 0.0, -half_wid, half_len, half_lw)
    add_line_box("line-boundary-left", -half_len, 0.0, half_lw, half_wid)
    add_line_box("line-boundary-right", half_len, 0.0, half_lw, half_wid)
    add_line_box("line-center", 0.0, 0.0, half_lw, half_wid)

    goal_area_depth = max(0.05, float(cfg.get("goal_area_depth", 1.0)))
    goal_area_width = max(line_w, float(cfg.get("goal_area_width", 3.0)))
    penalty_area_depth = max(0.05, float(cfg.get("penalty_area_depth", 2.0)))
    penalty_area_width = max(line_w, float(cfg.get("penalty_area_width", 4.0)))

    for side, sgn in (("left", -1.0), ("right", 1.0)):
        for prefix, depth, box_w in (
            ("goal-area", goal_area_depth, goal_area_width),
            ("penalty-area", penalty_area_depth, penalty_area_width),
        ):
            x_outer = sgn * half_len
            x_inner = sgn * (half_len - depth)
            y_half = min(0.5 * box_w, half_wid)
            add_line_box(f"line-{prefix}-{side}-outer", x_outer, 0.0, half_lw, y_half)
            add_line_box(f"line-{prefix}-{side}-inner", x_inner, 0.0, half_lw, y_half)
            add_line_box(
                f"line-{prefix}-{side}-top",
                0.5 * (x_outer + x_inner),
                y_half,
                0.5 * depth,
                half_lw,
            )
            add_line_box(
                f"line-{prefix}-{side}-bottom",
                0.5 * (x_outer + x_inner),
                -y_half,
                0.5 * depth,
                half_lw,
            )

    spot_dist = max(0.05, float(cfg.get("penalty_spot_distance", 1.5)))
    spot_r = max(0.02, 0.5 * line_w)
    for side, sgn in (("left", -1.0), ("right", 1.0)):
        x_spot = sgn * (half_len - spot_dist)
        ET.SubElement(
            worldbody,
            "geom",
            name=f"line-penalty-spot-{side}",
            type="cylinder",
            pos=f"{x_spot:g} 0 {half_h:g}",
            size=f"{spot_r:g} {line_h:g}",
            rgba=rgba_str,
            contype="0",
            conaffinity="0",
        )

    circle_d = max(0.1, float(cfg.get("center_circle_diameter", 1.5)))
    circle_r = 0.5 * circle_d
    seg_n = 48
    seg_len = (2.0 * math.pi * circle_r / seg_n) * 1.08
    for i in range(seg_n):
        theta = (2.0 * math.pi * i) / seg_n
        x = circle_r * math.cos(theta)
        y = circle_r * math.sin(theta)
        tangent_theta = theta + 0.5 * math.pi
        ET.SubElement(
            worldbody,
            "geom",
            name=f"line-center-circle-{i}",
            type="box",
            pos=f"{x:g} {y:g} {z:g}",
            quat=f"{math.cos(0.5 * tangent_theta):g} 0 0 {math.sin(0.5 * tangent_theta):g}",
            size=f"{0.5 * seg_len:g} {half_lw:g} {half_h:g}",
            rgba=rgba_str,
            contype="0",
            conaffinity="0",
        )


def build_match_scene_xml() -> Path:
    root = ET.parse(WORLD_XML).getroot()
    root = deepcopy(root)
    _sanitize_for_mujoco(root)
    compiler = root.find("compiler")
    asset_dir = SOCCER_DIR / (compiler.get("assetdir") if compiler is not None and compiler.get("assetdir") else "assets")
    _resolve_asset_paths(root, asset_dir)

    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    global_tag = visual.find("global")
    if global_tag is None:
        global_tag = ET.SubElement(visual, "global")
    global_tag.set("offwidth", str(max(int(global_tag.get("offwidth", "0") or 0), 1920)))
    global_tag.set("offheight", str(max(int(global_tag.get("offheight", "0") or 0), 1080)))

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError("world.xml missing worldbody")
    _add_field_markings(worldbody, _load_markings_cfg())

    fd, tmp_path = tempfile.mkstemp(prefix="sim_soccer_match_", suffix=".xml")
    import os

    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(ET.tostring(root, encoding="unicode"))
    return Path(tmp_path)


def _render_preview(model, data, out_path: Path) -> None:
    import imageio.v3 as iio
    import mujoco

    out_path.parent.mkdir(parents=True, exist_ok=True)
    renderer = mujoco.Renderer(model, 720, 1280)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.lookat[:] = [0.0, 0.0, 0.6]
    cam.distance = 14.0
    cam.elevation = -25.0
    cam.azimuth = 135.0
    renderer.update_scene(data, camera=cam)
    iio.imwrite(out_path, renderer.render())
    renderer.close()
    print(f"[view_sim_soccer] preview: {out_path}")


def _run_viewer(model, data) -> None:
    import mujoco
    import mujoco.viewer

    print("[view_sim_soccer] Opening MuJoCo viewer — Esc or close window to quit.")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(float(model.opt.timestep))


def main() -> int:
    parser = argparse.ArgumentParser(description="View sim_soccer2 full match scene in MuJoCo.")
    parser.add_argument("--preview-only", action="store_true", help="Save PNG and exit.")
    args = parser.parse_args()

    if not WORLD_XML.is_file():
        print(f"Missing {WORLD_XML}", file=sys.stderr)
        return 1

    scene_xml = build_match_scene_xml()
    print(f"[view_sim_soccer] scene: {scene_xml}")
    print(f"[view_sim_soccer] source: {WORLD_XML}")
    print(f"[view_sim_soccer] markings: {MATCH_CONFIG}")

    import mujoco

    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    print(
        f"[view_sim_soccer] loaded: nbody={model.nbody}, ngeom={model.ngeom}, "
        f"pitch~12x9m, markings=9x6m"
    )

    _render_preview(model, data, PREVIEW_PNG)
    if args.preview_only:
        return 0
    _run_viewer(model, data)
    return 0


if __name__ == "__main__":
    sys.exit(main())

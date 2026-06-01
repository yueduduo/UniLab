"""将 sim_soccer2 的 K1 与 world.xml 合并为 MotrixSim 可加载场景。

逻辑与 ``sim_soccer2/.../multi_robot_sim._build_multi_robot_soccer_scene_xml`` 对齐
（单台红方机器人、默认 PITCH_SCALE / 外圈地板 / 程序生成球门）。
"""

from __future__ import annotations

import math
import tempfile
import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path

from paths import BUILT_SCENE_XML, K1_ROBOT_XML, SOCCER_WORLD_XML

PITCH_SCALE = 0.45
ROBOT_SPAWN_Z_LIFT_M = 0.04
BASE_JOINT_NAME = "world_joint"
ROBOT_NAME = "robot_rp0"


def _write_temp_xml(xml_text: str) -> Path:
    fd = tempfile.NamedTemporaryFile(prefix="zzz_k1_soccer_", suffix=".xml", delete=False)
    fd.write(xml_text.encode("utf-8"))
    fd.flush()
    fd.close()
    return Path(fd.name)


def _ensure_offscreen_buffer(root: ET.Element, offwidth: int = 1920, offheight: int = 1080) -> None:
    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    global_tag = visual.find("global")
    if global_tag is None:
        global_tag = ET.SubElement(visual, "global")
    cur_w = int(global_tag.get("offwidth", "0") or "0")
    cur_h = int(global_tag.get("offheight", "0") or "0")
    global_tag.set("offwidth", str(max(cur_w, offwidth)))
    global_tag.set("offheight", str(max(cur_h, offheight)))


def _remove_all_plane_geoms(root: ET.Element) -> None:
    for worldbody in list(root.findall("worldbody")):
        for geom in list(worldbody.iter("geom")):
            if geom.get("type") != "plane":
                continue
            parent = next((p for p in worldbody.iter() if geom in list(p)), None)
            if parent is not None:
                parent.remove(geom)


def _find_template_robot_body(worldbody: ET.Element, base_joint_name: str) -> ET.Element:
    for body in list(worldbody.findall("body")):
        if body.find(f"joint[@name='{base_joint_name}']") is not None or body.find(
            f"freejoint[@name='{base_joint_name}']"
        ) is not None:
            return body
    raise RuntimeError(f"Cannot find template robot body with base joint '{base_joint_name}'")


def _prefix_body_tree_names(body: ET.Element, robot_name: str) -> None:
    for elem in body.iter():
        name = elem.get("name")
        if not name:
            continue
        if elem.tag == "body":
            if elem is body:
                elem.set("name", robot_name)
            else:
                elem.set("name", f"{robot_name}__{name}")
        elif elem.tag in ("joint", "freejoint", "site", "geom", "camera", "light"):
            elem.set("name", f"{robot_name}__{name}")


def _spawn_xy_theta(field_size: tuple[float, float] | None) -> tuple[float, float, float]:
    field_len = float(field_size[0]) if field_size is not None else 14.0
    return (-field_len * 0.25, 0.0, 0.0)


def _quat_from_yaw(theta: float) -> list[float]:
    half = 0.5 * float(theta)
    return [float(math.cos(half)), 0.0, 0.0, float(math.sin(half))]


def _add_procedural_goals(
    worldbody: ET.Element,
    field_length: float,
    goal_depth: float = 0.6,
    goal_width: float = 2.6,
    goal_height: float = 1.8,
    post_radius: float = 0.05,
) -> None:
    goal_half_y = 0.5 * float(goal_width)
    field_half_x = 0.5 * float(field_length)
    post_rgba = "0.8 0.8 0.8 1"
    net_rgba = "1 1 1 0.2"
    y_quat = "0 0 0.7071068 0.7071068"

    for side, x_sign in (("left", -1.0), ("right", 1.0)):
        goal_name = f"goal-{side}"
        goal_body = ET.SubElement(
            worldbody, "body", name=goal_name, pos=f"{x_sign * field_half_x:g} 0 0"
        )
        depth = x_sign * float(goal_depth)
        x_half_depth = 0.5 * abs(depth)
        ET.SubElement(
            goal_body,
            "geom",
            name=f"{goal_name}-post-left",
            type="cylinder",
            pos=f"{0.5 * depth:g} {goal_half_y:g} {0.5 * goal_height:g}",
            size=f"{post_radius:g} {0.5 * goal_height:g}",
            quat=y_quat,
            rgba=post_rgba,
            contype="0",
            conaffinity="0",
        )
        ET.SubElement(
            goal_body,
            "geom",
            name=f"{goal_name}-post-right",
            type="cylinder",
            pos=f"{0.5 * depth:g} {-goal_half_y:g} {0.5 * goal_height:g}",
            size=f"{post_radius:g} {0.5 * goal_height:g}",
            quat=y_quat,
            rgba=post_rgba,
            contype="0",
            conaffinity="0",
        )
        ET.SubElement(
            goal_body,
            "geom",
            name=f"{goal_name}-crossbar",
            type="cylinder",
            pos=f"{0.5 * depth:g} 0 {goal_height:g}",
            size=f"{post_radius:g} {goal_half_y:g}",
            rgba=post_rgba,
            contype="0",
            conaffinity="0",
        )


def _add_outer_floor_planes(
    worldbody: ET.Element,
    field_length: float,
    field_width: float,
    cfg: dict[str, object] | None = None,
) -> None:
    c = cfg if isinstance(cfg, dict) else {}
    if not bool(c.get("enabled", True)):
        return
    ratio = float(c.get("margin_ratio", 0.05))
    min_margin = float(c.get("min_margin", 1.0))
    margin_x = max(min_margin, 0.5 * field_length * ratio)
    margin_y = max(min_margin, 0.5 * field_width * ratio)
    field_half_x = 0.5 * float(field_length)
    field_half_y = 0.5 * float(field_width)
    rgba = c.get("color", [0.2, 0.5, 0.2, 1.0])
    if not isinstance(rgba, (list, tuple)) or len(rgba) != 4:
        rgba = [0.2, 0.5, 0.2, 1.0]
    rgba_str = " ".join(f"{float(v):g}" for v in rgba)
    coll = bool(c.get("collision", False))
    contype = "1" if coll else "0"
    conaffinity = "1" if coll else "0"

    for name, pos, size in (
        ("left-floor", f"{-field_half_x - margin_x:g} 0 0", f"{margin_x:g} {field_half_y:g} 1"),
        ("right-floor", f"{field_half_x + margin_x:g} 0 0", f"{margin_x:g} {field_half_y:g} 1"),
        ("top-floor", f"0 {field_half_y + margin_y:g} 0", f"{field_half_x + 2.0 * margin_x:g} {margin_y:g} 1"),
        ("bottom-floor", f"0 {-field_half_y - margin_y:g} 0", f"{field_half_x + 2.0 * margin_x:g} {margin_y:g} 1"),
    ):
        ET.SubElement(
            worldbody,
            "geom",
            name=name,
            type="plane",
            pos=pos,
            size=size,
            rgba=rgba_str,
            contype=contype,
            conaffinity=conaffinity,
        )


def build_k1_soccer_scene_xml(
    *,
    robot_xml: Path = K1_ROBOT_XML,
    soccer_world_xml: Path = SOCCER_WORLD_XML,
    output_path: Path | None = BUILT_SCENE_XML,
    pitch_scale: float = PITCH_SCALE,
) -> Path:
    """合并 K1 + sim_soccer2 足球场，返回写入的 MJCF 路径。"""
    if not robot_xml.is_file():
        raise FileNotFoundError(f"K1 robot XML not found: {robot_xml}")
    if not soccer_world_xml.is_file():
        raise FileNotFoundError(f"Soccer world XML not found: {soccer_world_xml}")

    meshdir = robot_xml.parent / "meshes"
    robot_root = ET.fromstring(robot_xml.read_text(encoding="utf-8"))
    world_root = ET.parse(soccer_world_xml).getroot()

    robot_compiler = robot_root.find("compiler")
    if robot_compiler is not None:
        robot_compiler.set("meshdir", meshdir.as_posix())

    _ensure_offscreen_buffer(robot_root)
    _remove_all_plane_geoms(robot_root)

    worldbody = robot_root.find("worldbody")
    if worldbody is None:
        raise RuntimeError("Robot XML missing worldbody")

    template_body = _find_template_robot_body(worldbody, base_joint_name=BASE_JOINT_NAME)
    template_actuator = robot_root.find("actuator")
    if template_actuator is None:
        raise RuntimeError("Robot XML missing actuator section")
    template_actuators = list(template_actuator)

    worldbody.remove(template_body)
    for child in list(template_actuator):
        template_actuator.remove(child)

    template_sensor = robot_root.find("sensor")
    if template_sensor is not None:
        robot_root.remove(template_sensor)

    template_pos_vals = [float(v) for v in (template_body.get("pos", "0 0 0").split())]
    template_spawn_z = template_pos_vals[2] if len(template_pos_vals) >= 3 else 0.0

    body_copy = deepcopy(template_body)
    _prefix_body_tree_names(body_copy, ROBOT_NAME)
    x, y, theta = _spawn_xy_theta(None)
    spawn_z = float(template_spawn_z) + float(ROBOT_SPAWN_Z_LIFT_M)
    body_copy.set("pos", f"{x:.6f} {y:.6f} {spawn_z:.6f}")
    body_copy.set("quat", " ".join(f"{v:.9g}" for v in _quat_from_yaw(theta)))
    worldbody.append(body_copy)

    for act in template_actuators:
        act_copy = deepcopy(act)
        if act_copy.get("name"):
            act_copy.set("name", f"{ROBOT_NAME}__{act_copy.get('name')}")
        if act_copy.get("joint"):
            act_copy.set("joint", f"{ROBOT_NAME}__{act_copy.get('joint')}")
        template_actuator.append(act_copy)

    world_compiler = world_root.find("compiler")
    world_asset_dir = soccer_world_xml.parent
    if world_compiler is not None and world_compiler.get("assetdir"):
        world_asset_dir = soccer_world_xml.parent / world_compiler.get("assetdir")

    robot_asset = robot_root.find("asset")
    if robot_asset is None:
        robot_asset = ET.SubElement(robot_root, "asset")
    world_asset = world_root.find("asset")
    if world_asset is not None:
        for child in list(world_asset):
            copied = deepcopy(child)
            for attr in (
                "file",
                "fileup",
                "filedown",
                "filefront",
                "fileback",
                "fileleft",
                "fileright",
            ):
                v = copied.get(attr)
                if v and not Path(v).is_absolute():
                    copied.set(attr, (world_asset_dir / v).as_posix())
            robot_asset.append(copied)

    out_field_len = 14.0
    out_field_wid = 9.0
    world_worldbody = world_root.find("worldbody")
    if world_worldbody is not None:
        for child in list(world_worldbody):
            copied = deepcopy(child)
            if copied.tag == "geom" and copied.get("name") == "pitch":
                size_str = copied.get("size")
                if size_str:
                    vals = [float(x) for x in size_str.split()]
                    if len(vals) >= 2:
                        vals[0] *= pitch_scale
                        vals[1] *= pitch_scale
                        copied.set("size", " ".join(f"{v:g}" for v in vals))
                        out_field_len = float(vals[0]) * 2.0
                        out_field_wid = float(vals[1]) * 2.0
                copied.attrib.pop("material", None)
                copied.set("rgba", "0.18 0.45 0.18 1")
            worldbody.append(copied)

    _add_outer_floor_planes(worldbody, field_length=out_field_len, field_width=out_field_wid)
    _add_procedural_goals(worldbody, field_length=out_field_len)

    xml_text = ET.tostring(robot_root, encoding="unicode")
    if output_path is None:
        return _write_temp_xml(xml_text)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(xml_text, encoding="utf-8")
    return output_path

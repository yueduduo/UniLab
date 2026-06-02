#!/usr/bin/env python3
"""Render static PNG assets for the UniLab RL task blog documents.

This script is intentionally cold-path documentation tooling. It compiles
representative MuJoCo scenes from ``src/unilab/assets/robots`` and saves PNG
previews under ``docs/blog/zh_CN/rl_tasks/images``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco

REPO_ROOT = Path(__file__).resolve().parents[5]
DOC_ROOT = REPO_ROOT / "docs" / "blog" / "zh_CN" / "rl_tasks"
ROBOTS_ROOT = REPO_ROOT / "src" / "unilab" / "assets" / "robots"


@dataclass(frozen=True)
class RenderTarget:
    slug: str
    xml_path: Path
    output_path: Path
    keyframe: str | None = None
    camera_distance: float = 4.0
    camera_azimuth: float = 135.0
    camera_elevation: float = -20.0
    lookat_z: float = 0.5


ROBOT_TARGETS: tuple[RenderTarget, ...] = (
    RenderTarget("go1", ROBOTS_ROOT / "go1" / "scene_flat.xml", DOC_ROOT / "images/robots/go1.png"),
    RenderTarget("go2", ROBOTS_ROOT / "go2" / "scene_flat.xml", DOC_ROOT / "images/robots/go2.png"),
    RenderTarget("go2w", ROBOTS_ROOT / "go2w" / "scene_flat.xml", DOC_ROOT / "images/robots/go2w.png"),
    RenderTarget("g1", ROBOTS_ROOT / "g1" / "scene_flat.xml", DOC_ROOT / "images/robots/g1.png", lookat_z=0.9),
    RenderTarget("k1", ROBOTS_ROOT / "k1" / "scene_flat.xml", DOC_ROOT / "images/robots/k1.png", lookat_z=0.8),
    RenderTarget(
        "go2_arm",
        ROBOTS_ROOT / "go2_arm" / "scene_flat.xml",
        DOC_ROOT / "images/robots/go2_arm.png",
    ),
    RenderTarget(
        "allegro_hand",
        ROBOTS_ROOT / "allegro_hand" / "scene.xml",
        DOC_ROOT / "images/robots/allegro_hand.png",
        camera_distance=1.4,
        lookat_z=0.15,
    ),
    RenderTarget(
        "sharpa_wave",
        ROBOTS_ROOT / "sharpa_wave" / "scene.xml",
        DOC_ROOT / "images/robots/sharpa_wave.png",
        camera_distance=1.6,
        lookat_z=0.15,
    ),
)


TASK_SCENES: dict[str, tuple[Path, float, float]] = {
    "go1_joystick_flat": (ROBOTS_ROOT / "go1" / "scene_flat.xml", 4.0, 0.5),
    "go1_joystick_rough": (ROBOTS_ROOT / "go1" / "scene_flat.xml", 4.0, 0.5),
    "go2_joystick_flat": (ROBOTS_ROOT / "go2" / "scene_flat.xml", 4.0, 0.5),
    "go2_joystick_rough": (ROBOTS_ROOT / "go2" / "scene_flat.xml", 4.0, 0.5),
    "go2_handstand": (ROBOTS_ROOT / "go2" / "scene_flat.xml", 4.0, 0.5),
    "go2_footstand": (ROBOTS_ROOT / "go2" / "scene_flat.xml", 4.0, 0.5),
    "go2w_joystick_flat": (ROBOTS_ROOT / "go2w" / "scene_flat.xml", 4.2, 0.5),
    "go2w_joystick_rough": (ROBOTS_ROOT / "go2w" / "scene_flat.xml", 4.2, 0.5),
    "g1_walk_flat": (ROBOTS_ROOT / "g1" / "scene_flat.xml", 4.5, 0.9),
    "g1_walk_rough": (ROBOTS_ROOT / "g1" / "scene_rough.xml", 4.5, 0.9),
    "g1_motion_tracking": (ROBOTS_ROOT / "g1" / "scene_flat.xml", 4.5, 0.9),
    "g1_motion_tracking_deploy": (ROBOTS_ROOT / "g1" / "scene_flat.xml", 4.5, 0.9),
    "g1_flip_tracking": (ROBOTS_ROOT / "g1" / "scene_flat.xml", 4.5, 0.9),
    "g1_wall_flip_tracking": (ROBOTS_ROOT / "g1" / "scene_flat_with_wall.xml", 4.5, 0.9),
    "g1_climb_tracking": (ROBOTS_ROOT / "g1" / "scene_climb_20_z_scale_1.xml", 5.0, 0.9),
    "g1_box_tracking": (ROBOTS_ROOT / "g1" / "scene_flat_with_largebox.xml", 4.5, 0.9),
    "g1_wbt_obs": (ROBOTS_ROOT / "g1" / "scene_flat.xml", 4.5, 0.9),
    "go2_arm_manip_loco": (ROBOTS_ROOT / "go2_arm" / "scene_flat.xml", 4.5, 0.55),
    "k1_walk_flat": (ROBOTS_ROOT / "k1" / "scene_flat.xml", 4.5, 0.8),
    "k1_soccer_dribble": (ROBOTS_ROOT / "k1" / "scene_soccer_dribble_minimal.xml", 4.5, 0.8),
    "allegro_inhand": (ROBOTS_ROOT / "allegro_hand" / "scene.xml", 1.4, 0.15),
    "allegro_inhand_grasp": (ROBOTS_ROOT / "allegro_hand" / "scene.xml", 1.4, 0.15),
    "sharpa_inhand": (ROBOTS_ROOT / "sharpa_wave" / "scene.xml", 1.6, 0.15),
    "sharpa_inhand_grasp": (ROBOTS_ROOT / "sharpa_wave" / "scene.xml", 1.6, 0.15),
}


def _find_keyframe(model: mujoco.MjModel, preferred: str | None) -> int | None:
    if preferred is not None:
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, preferred)
        if key_id >= 0:
            return key_id
    for name in ("stand", "home"):
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, name)
        if key_id >= 0:
            return key_id
    if model.nkey > 0:
        return 0
    return None


def _render_target(target: RenderTarget) -> None:
    model = mujoco.MjModel.from_xml_path(str(target.xml_path))
    data = mujoco.MjData(model)
    key_id = _find_keyframe(model, target.keyframe)
    if key_id is not None:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = target.camera_distance
    camera.azimuth = target.camera_azimuth
    camera.elevation = target.camera_elevation
    camera.lookat[:] = [0.0, 0.0, target.lookat_z]

    target.output_path.parent.mkdir(parents=True, exist_ok=True)
    with mujoco.Renderer(model, height=480, width=640) as renderer:
        renderer.update_scene(data, camera=camera)
        image = renderer.render()
    imageio.imwrite(target.output_path, image)


def _task_targets() -> list[RenderTarget]:
    targets: list[RenderTarget] = []
    for slug, (xml_path, distance, lookat_z) in sorted(TASK_SCENES.items()):
        targets.append(
            RenderTarget(
                slug=slug,
                xml_path=xml_path,
                output_path=DOC_ROOT / "images" / "tasks" / f"{slug}.png",
                camera_distance=distance,
                lookat_z=lookat_z,
            )
        )
    return targets


def main() -> None:
    manifest: dict[str, object] = {"rendered": [], "failed": []}
    for target in (*ROBOT_TARGETS, *_task_targets()):
        try:
            _render_target(target)
        except Exception as exc:
            manifest["failed"].append(
                {
                    "slug": target.slug,
                    "xml": str(target.xml_path.relative_to(REPO_ROOT)),
                    "output": str(target.output_path.relative_to(REPO_ROOT)),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        else:
            manifest["rendered"].append(
                {
                    "slug": target.slug,
                    "xml": str(target.xml_path.relative_to(REPO_ROOT)),
                    "output": str(target.output_path.relative_to(REPO_ROOT)),
                }
            )

    output = DOC_ROOT / "_generated" / "render_manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        "rendered={rendered} failed={failed} manifest={manifest}".format(
            rendered=len(manifest["rendered"]),
            failed=len(manifest["failed"]),
            manifest=output.relative_to(REPO_ROOT),
        )
    )


if __name__ == "__main__":
    main()

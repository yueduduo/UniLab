from __future__ import annotations

from pathlib import Path

from unilab.visualization.soccer_dribble_playback_debug import _load_mujoco_geom_friction


def test_load_soccer_scene_geom_friction() -> None:
    scene = (
        Path(__file__).resolve().parents[2]
        / "src/unilab/assets/robots/k1/scene_soccer_dribble_minimal.xml"
    )
    friction = _load_mujoco_geom_friction(scene)
    assert friction["ball_geom"] == (1.0, 0.02, 0.05)
    assert friction["COL_Collider"] == (1.0, 0.005, 0.02)

"""K1 walk flat asset contract aligned with G1 foot contact layout."""

from __future__ import annotations

import os
from pathlib import Path

import mujoco
import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")


def test_k1_walk_flat_assets_define_contact_sensors_for_gait_rewards():
    repo_root = Path(__file__).parents[4]
    scene_text = (
        repo_root / "src" / "unilab" / "assets" / "robots" / "k1" / "scene_flat.xml"
    ).read_text()
    model_text = (repo_root / "src" / "unilab" / "assets" / "robots" / "k1" / "k1.xml").read_text()

    for name in (
        "left_foot_contact_0",
        "left_foot_contact_1",
        "left_foot_contact_2",
        "left_foot_contact_3",
        "right_foot_contact_0",
        "right_foot_contact_1",
        "right_foot_contact_2",
        "right_foot_contact_3",
    ):
        assert name in scene_text

    for name in (
        "left_foot_contact_0_geom",
        "left_foot_contact_1_geom",
        "left_foot_contact_2_geom",
        "left_foot_contact_3_geom",
        "right_foot_contact_0_geom",
        "right_foot_contact_1_geom",
        "right_foot_contact_2_geom",
        "right_foot_contact_3_geom",
    ):
        assert name in model_text

    assert 'site name="left_foot" pos="0.026 0 -0.038"' in model_text
    assert 'site name="right_foot" pos="0.026 0 -0.038"' in model_text


def test_k1_stand_pose_reports_foot_contact_like_g1():
    repo_root = Path(__file__).parents[4]
    sensor_names = [f"{side}_foot_contact_{i}" for side in ("left", "right") for i in range(4)]

    def stand_contacts(scene_rel: str) -> np.ndarray:
        model = mujoco.MjModel.from_xml_path(str(repo_root / scene_rel))
        data = mujoco.MjData(model)
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        for _ in range(50):
            mujoco.mj_step(model, data)
        values = []
        for name in sensor_names:
            sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
            adr = model.sensor_adr[sensor_id]
            values.append(float(data.sensordata[adr]))
        return np.asarray(values, dtype=np.float64)

    g1 = stand_contacts("src/unilab/assets/robots/g1/scene_flat.xml")
    k1 = stand_contacts("src/unilab/assets/robots/k1/scene_flat.xml")
    assert np.all(g1 >= 0.5)
    assert np.all(k1 >= 0.5)

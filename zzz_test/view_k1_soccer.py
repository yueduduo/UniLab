"""K1 + sim_soccer2 足球场 — MotrixSim 被动查看（无策略/键盘控制）。

用法（仓库根目录）:
    uv run zzz_test/view_k1_soccer.py

首次运行前可生成合并 MJCF（可选）:
    uv run zzz_test/gen_scene_xml.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_ZZZ = Path(__file__).resolve().parent
if str(_ZZZ) not in sys.path:
    sys.path.insert(0, str(_ZZZ))

from scene_builder import build_k1_soccer_scene_xml

RENDER_FPS = 60.0
LOG_LEVEL = "WARN"
CAM_LOOKAT = [0.0, 0.0, 0.8]
CAM_DISTANCE = 12.0
CAM_ELEVATION = -22.0
CAM_AZIMUTH = 135.0


def _render_settings():
    from motrixsim.render import RenderSettings

    settings = RenderSettings.quality()
    settings.enable_shadow = True
    settings.enable_ssgi = True
    settings.simplify_render_mesh = False
    return settings


def main() -> None:
    scene_xml = build_k1_soccer_scene_xml()
    print(f"场景 MJCF: {scene_xml}")
    print(f"足球场资源: sim_soccer2 .../environments/soccer/world.xml")

    import motrixsim as mtx
    from motrixsim.render import RenderApp

    model = mtx.load_model(str(scene_xml))
    data = mtx.SceneData(model)
    model.forward_kinematic(data)

    frame_dt = 1.0 / RENDER_FPS
    print("MotrixSim 查看模式：仅 forward_kinematic，不 step、不发送电机控制。")

    with RenderApp(LOG_LEVEL) as render:
        settings = _render_settings()
        render.launch(model, render_settings=settings)
        render.set_main_camera(None)
        render.system_camera.set_view(CAM_LOOKAT, CAM_DISTANCE, CAM_ELEVATION, CAM_AZIMUTH)

        try:
            while not render.is_closed:
                t0 = time.perf_counter()
                model.forward_kinematic(data)
                render.sync(data)
                sleep_s = frame_dt - (time.perf_counter() - t0)
                if sleep_s > 0.0:
                    time.sleep(sleep_s)
        except KeyboardInterrupt:
            return


if __name__ == "__main__":
    main()

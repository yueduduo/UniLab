"""sim_soccer2 资源路径（与 runtime_config 默认一致）。"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MOTRIXSIM_ASSETS = REPO_ROOT / "sim_soccer2" / "simulation" / "motrixsim" / "assets"

K1_ROBOT_XML = MOTRIXSIM_ASSETS / "robots" / "k1" / "K1_22dof.xml"
SOCCER_WORLD_XML = MOTRIXSIM_ASSETS / "environments" / "soccer" / "world.xml"

BUILT_SCENE_XML = Path(__file__).resolve().parent / "scene_k1_soccer.xml"

"""生成 ``scene_k1_soccer.xml``（K1 + sim_soccer2 world.xml 合并结果）。"""

from __future__ import annotations

import sys
from pathlib import Path

_ZZZ = Path(__file__).resolve().parent
if str(_ZZZ) not in sys.path:
    sys.path.insert(0, str(_ZZZ))

from paths import BUILT_SCENE_XML
from scene_builder import build_k1_soccer_scene_xml


def main() -> None:
    path = build_k1_soccer_scene_xml(output_path=BUILT_SCENE_XML)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()

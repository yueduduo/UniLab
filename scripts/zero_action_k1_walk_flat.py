#!/usr/bin/env python3
"""K1 walk-flat: action=0 (hold default), visualize standing drift.

Examples:
  uv run scripts/zero_action_k1_walk_flat.py --visual
  uv run scripts/zero_action_k1_walk_flat.py --video --steps 400
  uv run scripts/zero_action_k1_walk_flat.py --visual --quiet-stand
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from zero_action_walk_flat_lib import main_for_robot

if __name__ == "__main__":
    raise SystemExit(main_for_robot("k1"))

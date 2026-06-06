"""Auto-play latest APPO checkpoint and switch on new weights.

Watches ``logs/appo/<task-name>/`` for the newest ``model_*.pt`` and
restarts interactive Motrix playback when a newer checkpoint appears.

Usage:
  uv run scripts/play_latest_auto_appo.py
  uv run scripts/play_latest_auto_appo.py --check-interval 1.0
  uv run scripts/play_latest_auto_appo.py --task-name K1SoccerDribble --task-slug k1_soccer_dribble
  uv run scripts/play_latest_auto_appo.py \\
    --checkpoint logs/appo/K1SoccerDribble/2026-06-01_19-34-31_motrix/model_2000.pt
  uv run scripts/play_latest_auto_appo.py --load-run 2026-06-01_19-34-31_motrix --checkpoint 2000
"""

from __future__ import annotations

import sys

from play_latest_auto import run_auto_play


def main() -> int:
    return run_auto_play(
        [
            "--preset",
            "appo",
            "--task-name",
            "K1SoccerDribble",
            "--task-slug",
            "k1_soccer_dribble",
            "--sim",
            "motrix",
            *sys.argv[1:],
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())

"""K1 soccer dribble playback probe — aligned with ``play_latest_auto_appo`` / ``play_interactive``.

Uses the same env overrides (``BackendAdapter``), keyboard teleop defaults, camera, and
raw-env APPO stepping as the auto-play entrypoint.

Usage (headless + debug logs, default ~60 s):
  uv run scripts/probe_soccer_dribble_playback.py --steps 3000

Usage (Motrix 3D window — matches play_latest_auto_appo unless ``--no-keyboard``):
  CUDA_HOME=... LD_LIBRARY_PATH=... uv run scripts/probe_soccer_dribble_playback.py \\
    --visual --steps 3000 --log-every 50

Run until you close the render window:
  uv run scripts/probe_soccer_dribble_playback.py --visual --until-close
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from unilab.base.registry import ensure_registries
from unilab.visualization.soccer_dribble_playback import (
    create_soccer_dribble_env,
    load_soccer_dribble_policy,
    run_headless_soccer_dribble_playback,
    run_motrix_soccer_dribble_playback,
    soccer_debug_from_cfg,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="K1 soccer dribble playback probe")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT
        / "logs/appo/K1SoccerDribble/2026-06-01_19-34-31_motrix/model_1000.pt",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=3000,
        help="Simulation steps for headless mode (3000 ≈ 60 s)",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=50,
        help="Print [soccer-debug] every N steps",
    )
    parser.add_argument(
        "--visual",
        action="store_true",
        help="Open Motrix 3D render window while stepping",
    )
    parser.add_argument(
        "--until-close",
        action="store_true",
        help="With --visual: run until render window is closed",
    )
    parser.add_argument(
        "--no-keyboard",
        action="store_true",
        help="Disable WASD teleop (play_latest_auto_appo enables keyboard by default)",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.until_close and not args.visual:
        raise ValueError("--until-close requires --visual")

    checkpoint_path = args.checkpoint.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    keyboard = args.visual and not args.no_keyboard

    ensure_registries()
    conf_dir = (ROOT / "conf" / "appo").resolve()
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(conf_dir), version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=[
                "task=k1_soccer_dribble/motrix",
                "training.sim_backend=motrix",
                f"training.device={args.device}",
                "interactive.action_mode=policy",
                "interactive.keyboard=true",
                "interactive.soccer_dribble_debug=true",
            ],
        )

    env = create_soccer_dribble_env(cfg, num_envs=1, root_dir=ROOT)
    policy = load_soccer_dribble_policy(
        env,
        checkpoint_path,
        cfg,
        device=args.device,
    )
    env.init_state()
    debug = soccer_debug_from_cfg(env, cfg, log_every_steps=args.log_every)
    if debug is None:
        raise RuntimeError("soccer debug diagnostics required for probe")

    print(f"[probe] checkpoint={checkpoint_path}")
    sim_seconds = args.steps * float(env.cfg.ctrl_dt)
    mode = "visual" if args.visual else "headless"
    print(f"[probe] mode={mode} keyboard={keyboard} ctrl_dt={env.cfg.ctrl_dt}")

    if args.visual:
        if args.until_close:
            print("[probe] running until render window closes.")
        else:
            print(f"[probe] visual play_steps={args.steps} (~{sim_seconds:.1f}s if finite).")
        run_motrix_soccer_dribble_playback(
            env,
            policy,
            cfg,
            device=args.device,
            debug=debug,
            keyboard=keyboard,
            keyboard_step_lin=float(OmegaConf.select(cfg, "interactive.keyboard_step_lin", default=0.1)),
            keyboard_step_ang=float(OmegaConf.select(cfg, "interactive.keyboard_step_ang", default=0.2)),
            until_close=args.until_close,
            play_steps=args.steps,
            log_prefix="[probe]",
        )
    else:
        debug.print_setup_once()
        print(f"[probe] running {args.steps} headless steps (~{sim_seconds:.1f}s sim time).")
        run_headless_soccer_dribble_playback(
            env,
            policy,
            debug,
            device=args.device,
            steps=args.steps,
        )

    print("[probe] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

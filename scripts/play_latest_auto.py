"""Auto-play latest checkpoint and switch on new weights.

Usage (FlashSAC / off-policy):
  uv run scripts/play_latest_auto.py \\
    --task-name K1SoccerDribble --task-slug k1_soccer_dribble --check-interval 1.0

Usage (APPO):
  uv run scripts/play_latest_auto_appo.py --check-interval 1.0
"""

from __future__ import annotations

import argparse
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


MODEL_PATTERN = re.compile(r"^model_(\d+)\.pt$")


@dataclass(frozen=True)
class PlayLatestPreset:
    config_dir: str
    algo_log_name: str
    task_override: str  # format: {algo}, {task_slug}, {sim}
    hydra_algo_override: str | None = None
    interactive_overrides: tuple[str, ...] = (
        "+interactive.action_mode=policy",
        "+interactive.keyboard=true",
    )


PRESETS: dict[str, PlayLatestPreset] = {
    "flashsac": PlayLatestPreset(
        config_dir="offpolicy",
        algo_log_name="flash_sac",
        task_override="{algo}/{task_slug}/{sim}",
        hydra_algo_override="{algo}",
    ),
    "appo": PlayLatestPreset(
        config_dir="appo",
        algo_log_name="appo",
        task_override="{task_slug}/{sim}",
        interactive_overrides=(
            "interactive.action_mode=policy",
            "interactive.keyboard=true",
            "interactive.soccer_dribble_debug=true",
        ),
    ),
}


def _latest_run_dir(task_log_root: Path) -> Path | None:
    if not task_log_root.exists():
        return None
    runs = [p for p in task_log_root.iterdir() if p.is_dir()]
    if not runs:
        return None
    return max(runs, key=lambda p: p.name)


def _latest_checkpoint(run_dir: Path) -> Path | None:
    best_iter = -1
    best_path: Path | None = None
    for path in run_dir.iterdir():
        if not path.is_file():
            continue
        m = MODEL_PATTERN.match(path.name)
        if m is None:
            continue
        it = int(m.group(1))
        if it > best_iter:
            best_iter = it
            best_path = path
    return best_path


def _resolve_checkpoint_arg(
    *,
    repo_root: Path,
    task_log_root: Path,
    checkpoint: str,
    load_run: str | None,
) -> Path | None:
    """Resolve --checkpoint to an existing model_*.pt path."""
    raw = checkpoint.strip()
    if not raw:
        return None

    candidate = Path(raw)
    if candidate.is_file():
        return candidate.resolve()
    repo_candidate = repo_root / raw
    if repo_candidate.is_file():
        return repo_candidate.resolve()

    if raw.isdigit():
        filename = f"model_{raw}.pt"
    elif raw.endswith(".pt"):
        filename = raw
    elif raw.startswith("model_"):
        filename = raw if raw.endswith(".pt") else f"{raw}.pt"
    else:
        filename = raw

    if load_run is not None:
        run_dir = task_log_root / load_run
        if not run_dir.is_dir():
            run_dir = Path(load_run)
            if not run_dir.is_dir():
                return None
        ckpt = run_dir / filename
        return ckpt.resolve() if ckpt.is_file() else None

    run_dir = _latest_run_dir(task_log_root)
    if run_dir is None:
        return None
    ckpt = run_dir / filename
    return ckpt.resolve() if ckpt.is_file() else None


def _build_play_cmd(
    *,
    repo_root: Path,
    preset: PlayLatestPreset,
    algo: str,
    task_slug: str,
    sim: str,
    checkpoint_path: Path,
) -> list[str]:
    task = preset.task_override.format(algo=algo, task_slug=task_slug, sim=sim)
    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "play_interactive.py"),
        "--config-path",
        str(repo_root / "conf" / preset.config_dir),
        "--config-name",
        "config",
    ]
    if preset.hydra_algo_override is not None:
        cmd.append(f"algo={preset.hydra_algo_override.format(algo=algo)}")
    cmd.extend(
        [
            f"task={task}",
            f"training.sim_backend={sim}",
            f"algo.algo_log_name={preset.algo_log_name}",
            f"algo.load_run={checkpoint_path}",
            *preset.interactive_overrides,
        ]
    )
    return cmd


def _terminate_process(proc: subprocess.Popen[bytes] | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def run_auto_play(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Auto-play latest checkpoint")
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        default="flashsac",
        help="Training stack preset (flashsac=off-policy, appo=on-policy APPO)",
    )
    parser.add_argument("--algo", default=None, help="Hydra algo name (flashsac preset only)")
    parser.add_argument("--task-name", default="K1SoccerDribble")
    parser.add_argument("--task-slug", default="k1_soccer_dribble")
    parser.add_argument("--sim", default="motrix")
    parser.add_argument("--logs-root", default="logs")
    parser.add_argument("--check-interval", type=float, default=2.0)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "Fixed checkpoint: full/relative path, model_2000.pt, or iteration 2000. "
            "With --load-run, resolve under that run directory."
        ),
    )
    parser.add_argument(
        "--load-run",
        default=None,
        help="Training run folder name (e.g. 2026-06-01_19-34-31_motrix) for --checkpoint",
    )
    args = parser.parse_args(argv)

    preset = PRESETS[args.preset]
    algo = args.algo or args.preset
    repo_root = Path(__file__).resolve().parents[1]
    task_log_root = repo_root / args.logs_root / preset.algo_log_name / args.task_name

    fixed_ckpt: Path | None = None
    if args.checkpoint is not None:
        fixed_ckpt = _resolve_checkpoint_arg(
            repo_root=repo_root,
            task_log_root=task_log_root,
            checkpoint=args.checkpoint,
            load_run=args.load_run,
        )
        if fixed_ckpt is None:
            print(f"[play-latest] checkpoint not found: {args.checkpoint}", file=sys.stderr)
            return 1
        print(f"[play-latest] preset={args.preset} fixed checkpoint: {fixed_ckpt}")
    else:
        print(f"[play-latest] preset={args.preset} watching: {task_log_root}")

    current_ckpt: Path | None = None
    play_proc: subprocess.Popen[bytes] | None = None

    def _handle_stop(_sig, _frame) -> None:
        nonlocal play_proc
        print("\n[play-latest] stopping...")
        _terminate_process(play_proc)
        play_proc = None
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)

    try:
        while True:
            if fixed_ckpt is not None:
                latest_ckpt = fixed_ckpt
            else:
                run_dir = _latest_run_dir(task_log_root)
                if run_dir is None:
                    time.sleep(args.check_interval)
                    continue
                latest_ckpt = _latest_checkpoint(run_dir)
                if latest_ckpt is None:
                    time.sleep(args.check_interval)
                    continue

            if current_ckpt != latest_ckpt:
                print(f"[play-latest] switching to: {latest_ckpt}")
                _terminate_process(play_proc)
                play_cmd = _build_play_cmd(
                    repo_root=repo_root,
                    preset=preset,
                    algo=algo,
                    task_slug=args.task_slug,
                    sim=args.sim,
                    checkpoint_path=latest_ckpt,
                )
                play_proc = subprocess.Popen(play_cmd, cwd=str(repo_root), env=os.environ.copy())
                current_ckpt = latest_ckpt

            time.sleep(args.check_interval)
    except KeyboardInterrupt:
        return 0


def main() -> int:
    return run_auto_play()


if __name__ == "__main__":
    raise SystemExit(main())

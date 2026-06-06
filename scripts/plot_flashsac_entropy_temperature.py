"""Plot FlashSAC actor_entropy and temperature from TensorBoard event files."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = (
    ROOT / "conf/offpolicy/task/flashsac/k1_soccer_dribble/figures"
)


def load_scalars(run_dir: Path, tag: str) -> tuple[list[int], list[float]]:
    ea = EventAccumulator(str(run_dir))
    ea.Reload()
    events = ea.Scalars(tag)
    return [int(e.step) for e in events], [float(e.value) for e in events]


def target_entropy_from_config(run_dir: Path, action_dim: int = 22) -> float | None:
    cfg_path = run_dir / "run_config.json"
    if not cfg_path.exists():
        return None
    cfg = json.loads(cfg_path.read_text())
    params = cfg.get("algo", {}).get("algo_params", {})
    if params.get("temp_target_entropy") is not None:
        return float(params["temp_target_entropy"])
    sigma = float(params.get("temp_target_sigma", 0.15))
    per_dim = 0.5 * math.log(2.0 * math.pi * math.e * sigma * sigma)
    return float(action_dim) * per_dim


def plot_combined(runs: list[tuple[str, Path]], out_path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    colors = ["#2563eb", "#7c3aed", "#dc2626", "#059669"]
    target_plotted = False

    for i, (label, run_dir) in enumerate(runs):
        if not run_dir.is_dir():
            continue
        c = colors[i % len(colors)]
        s_ent, v_ent = load_scalars(run_dir, "train/actor_entropy")
        s_temp, v_temp = load_scalars(run_dir, "train/temperature")
        axes[0].plot(s_ent, v_ent, label=label, color=c, alpha=0.85, linewidth=1.2)
        axes[1].plot(s_temp, v_temp, label=label, color=c, alpha=0.85, linewidth=1.2)
        tgt = target_entropy_from_config(run_dir)
        if tgt is not None and not target_plotted:
            axes[0].axhline(
                tgt,
                color="#64748b",
                linestyle="--",
                linewidth=1,
                label=f"target_entropy (22-d, σ from cfg) ≈ {tgt:.2f}",
            )
            target_plotted = True

    axes[0].set_ylabel("actor_entropy  (−E[log π])")
    axes[0].set_title("FlashSAC: actor_entropy and temperature vs iteration")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[1].set_ylabel("temperature  (α)")
    axes[1].set_xlabel("iteration（TensorBoard step）")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_dual_axis(label: str, run_dir: Path, out_path: Path) -> None:
    s_ent, v_ent = load_scalars(run_dir, "train/actor_entropy")
    s_temp, v_temp = load_scalars(run_dir, "train/temperature")
    tgt = target_entropy_from_config(run_dir)

    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(s_ent, v_ent, color="#2563eb", label="actor_entropy")
    if tgt is not None:
        ax1.axhline(tgt, color="#94a3b8", linestyle="--", label=f"target ≈ {tgt:.2f}")
    ax1.set_ylabel("entropy", color="#2563eb")
    ax1.tick_params(axis="y", labelcolor="#2563eb")

    ax2 = ax1.twinx()
    ax2.plot(s_temp, v_temp, color="#dc2626", alpha=0.85, label="temperature")
    ax2.set_ylabel("temperature α", color="#dc2626")
    ax2.tick_params(axis="y", labelcolor="#dc2626")

    ax1.set_xlabel("iteration")
    ax1.set_title(label)
    ax1.grid(True, alpha=0.3)
    lines1, lab1 = ax1.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lab1 + lab2, loc="upper right", fontsize=8)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    log_root = ROOT / "logs" / "flash_sac"
    runs: list[tuple[str, Path]] = [
        ("K1WalkFlat mujoco 20-53", log_root / "K1WalkFlat/2026-06-04_20-53-01_mujoco"),
        ("K1WalkFlat mujoco 20-11", log_root / "K1WalkFlat/2026-06-04_20-11-39_mujoco"),
        ("K1SoccerDribble motrix", log_root / "K1SoccerDribble/2026-06-05_00-37-56_motrix"),
    ]

    plot_combined(runs, args.out_dir / "entropy_temperature_curves.png")
    print(f"wrote {args.out_dir / 'entropy_temperature_curves.png'}")

    for label, run_dir in runs:
        if run_dir.is_dir():
            safe = run_dir.name
            plot_dual_axis(label, run_dir, args.out_dir / f"entropy_temp_{safe}.png")
            print(f"wrote {args.out_dir / f'entropy_temp_{safe}.png'}")


if __name__ == "__main__":
    main()

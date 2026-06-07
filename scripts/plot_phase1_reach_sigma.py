"""Visualize phase1_reach = 1 - tanh(dist / sigma) for different sigma values."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = (
    ROOT / "conf/offpolicy/task/flashsac/k1_soccer_dribble/figures/phase1_reach_sigma.png"
)
PHASE1_DISTANCE_M = 1.0
PHASE1_RADIUS_M = 0.18


def phase1_reach(dist: np.ndarray, sigma: float) -> np.ndarray:
    return 1.0 - np.tanh(dist / max(sigma, 1e-6))


def plot(sigmas: list[float], out_path: Path) -> None:
    dist = np.linspace(0.0, 1.5, 600)
    colors = ["#dc2626", "#ea580c", "#2563eb", "#059669"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    for sigma, color in zip(sigmas, colors, strict=True):
        rew = phase1_reach(dist, sigma)
        axes[0].plot(dist, rew, label=f"sigma = {sigma:g} m", color=color, linewidth=2)
        axes[1].plot(dist, rew * 10.0, label=f"sigma = {sigma:g} m (x10 scale)", color=color, linewidth=2)

    for ax in axes:
        ax.axvline(PHASE1_DISTANCE_M, color="#64748b", linestyle=":", linewidth=1.2, alpha=0.8)
        ax.axvline(PHASE1_RADIUS_M, color="#94a3b8", linestyle="--", linewidth=1.0, alpha=0.8)
        ax.set_xlabel("base-to-waypoint distance (m)")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0.0, 1.5)
    axes[0].set_ylim(0.0, 1.05)
    axes[1].set_ylim(0.0, 10.5)

    axes[0].set_ylabel("raw reward  1 - tanh(d / sigma)")
    axes[0].set_title("phase1_reach shaping for different sigma")
    axes[1].set_ylabel("weighted reward  (scale = 10.0)")
    axes[1].set_title("per-step reward after scale=10.0")

    legend_extra = (
        f"dashed: reach radius {PHASE1_RADIUS_M} m\n"
        f"dotted: phase1 distance {PHASE1_DISTANCE_M} m"
    )
    axes[0].text(
        0.98,
        0.02,
        legend_extra,
        transform=axes[0].transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85},
    )
    axes[0].legend(loc="upper right", fontsize=9)
    axes[1].legend(loc="upper right", fontsize=9)

    table_rows = []
    for sigma in sigmas:
        r0 = float(phase1_reach(np.array([0.0]), sigma)[0])
        r_start = float(phase1_reach(np.array([PHASE1_DISTANCE_M]), sigma)[0])
        r_radius = float(phase1_reach(np.array([PHASE1_RADIUS_M]), sigma)[0])
        table_rows.append([f"{sigma:g}", f"{r0:.3f}", f"{r_radius:.3f}", f"{r_start:.3f}"])

    table = axes[1].table(
        cellText=table_rows,
        colLabels=["sigma (m)", "d=0", f"d={PHASE1_RADIUS_M}", f"d={PHASE1_DISTANCE_M}"],
        loc="lower center",
        bbox=[0.08, -0.42, 0.84, 0.28],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sigmas",
        type=float,
        nargs="+",
        default=[0.1, 0.3, 0.6, 1.0],
        help="phase1_reach_sigma values to compare",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    plot(args.sigmas, args.out)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()

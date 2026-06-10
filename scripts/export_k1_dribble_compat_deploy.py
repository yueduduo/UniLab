#!/usr/bin/env python3
"""Export FlashSAC K1SoccerDribbleCompat checkpoint to TorchScript deploy policy (78->22)."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from unilab.algos.torch.flash_sac.network import FlashSACActor
from unilab.envs.locomotion.k1.constants import (
    K1_NUM_ACTION,
    k1_soccer_dribble_compat_actor_obs_dim,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = (
    REPO_ROOT
    / "MOS-SIM-perf-pipelined-capture"
    / "simulation"
    / "motrixsim"
    / "assets"
    / "policies"
    / "k1_model_46000.pt"
)


class _DeterministicDeploy(nn.Module):
    """Deploy wrapper: deterministic tanh(mean), matching eval explore(deterministic=True)."""

    def __init__(self, actor: FlashSACActor) -> None:
        super().__init__()
        self._actor = actor

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        mean, _ = self._actor.get_mean_and_std(obs, training=False)
        return torch.tanh(mean)


def export_torchscript(
    checkpoint: Path,
    output: Path,
    *,
    actor_num_blocks: int = 2,
    actor_hidden_dim: int = 128,
    actor_noise_zeta_mu: float = 2.0,
    actor_noise_zeta_max: int = 16,
) -> None:
    obs_dim = k1_soccer_dribble_compat_actor_obs_dim(K1_NUM_ACTION)
    action_dim = K1_NUM_ACTION
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "actor" not in ckpt:
        raise RuntimeError(f"Expected FlashSAC checkpoint with top-level 'actor' key: {checkpoint}")

    actor = FlashSACActor(
        num_blocks=actor_num_blocks,
        input_dim=obs_dim,
        hidden_dim=actor_hidden_dim,
        action_dim=action_dim,
        noise_zeta_mu=actor_noise_zeta_mu,
        noise_zeta_max=actor_noise_zeta_max,
        device="cpu",
    )
    actor.load_state_dict(ckpt["actor"])
    actor.eval()

    deploy = _DeterministicDeploy(actor).eval()
    with torch.inference_mode():
        example = torch.zeros(1, obs_dim, dtype=torch.float32)
        traced = torch.jit.trace(deploy, example)
    traced.eval()

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(traced, str(output))
    with torch.inference_mode():
        sanity = traced(example)
    if sanity.shape != (1, action_dim):
        raise RuntimeError(f"Unexpected deploy output shape {tuple(sanity.shape)}")
    print(f"Exported TorchScript deploy policy: {output} ({obs_dim}->{action_dim})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="FlashSAC .pt checkpoint (e.g. model_9000.pt)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output path (default: {DEFAULT_OUT})",
    )
    parser.add_argument("--actor-num-blocks", type=int, default=2)
    parser.add_argument("--actor-hidden-dim", type=int, default=128)
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")
    export_torchscript(
        args.checkpoint.resolve(),
        args.output.resolve(),
        actor_num_blocks=args.actor_num_blocks,
        actor_hidden_dim=args.actor_hidden_dim,
    )


if __name__ == "__main__":
    main()

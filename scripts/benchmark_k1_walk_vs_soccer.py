"""Benchmark K1WalkFlat vs K1SoccerDribble env.step throughput (Motrix)."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
CONF_OFFPOLICY = ROOT / "conf" / "offpolicy"


def _make_env(cfg, algo_name: str = "flashsac"):
    from unilab.base.registry import ensure_registries
    from unilab.training.backend_adapter import BackendAdapter
    from unilab.training.common import create_env

    ensure_registries()
    env_cfg_override = BackendAdapter(cfg, root_dir=ROOT, algo_name=algo_name).build_task_env_cfg_override()
    return create_env(
        cfg,
        num_envs=int(cfg.algo.num_envs),
        env_cfg_override=env_cfg_override,
    )


def _bench_env(env, num_warmup: int, num_steps: int) -> dict:
    n = env._num_envs
    act_dim = env.action_space.shape[0]
    actions = np.zeros((n, act_dim), dtype=np.float32)
    env.init_state()
    env.reset(np.arange(n, dtype=np.int32))
    for _ in range(num_warmup):
        env.step(actions)
    t0 = time.perf_counter()
    for _ in range(num_steps):
        env.step(actions)
    elapsed = time.perf_counter() - t0
    env.close()
    total_env_steps = num_steps * n
    return {
        "num_envs": n,
        "num_steps": num_steps,
        "sim_dt": float(env._cfg.sim_dt),
        "ctrl_dt": float(env._cfg.ctrl_dt),
        "sim_substeps": int(env._cfg.sim_substeps),
        "elapsed_s": elapsed,
        "s_per_ctrl_step": elapsed / num_steps,
        "env_steps_per_s": total_env_steps / elapsed,
        "physics_substeps_per_s": total_env_steps * int(env._cfg.sim_substeps) / elapsed,
    }


def main() -> None:
    GlobalHydra.instance().clear()
    num_envs = 1024
    num_warmup = 20
    num_steps = 200
    tasks: list[tuple[str, str]] = [
        ("K1WalkFlat/motrix", "flashsac/k1_walk_flat/motrix"),
        ("K1WalkFlat/mujoco", "flashsac/k1_walk_flat/mujoco"),
        ("K1SoccerDribble/motrix", "flashsac/k1_soccer_dribble/motrix"),
    ]
    rows: list[tuple[str, dict]] = []
    with initialize_config_dir(config_dir=str(CONF_OFFPOLICY), version_base="1.3"):
        for label, task in tasks:
            cfg = compose("config", overrides=[f"task={task}", f"algo.num_envs={num_envs}"])
            try:
                env = _make_env(cfg)
            except Exception as exc:
                print(f"SKIP {label}: {exc}")
                continue
            stats = _bench_env(env, num_warmup, num_steps)
            stats["task"] = label
            stats["reward_terms"] = len(env._reward_cfg.scales)
            stats["updates_per_step"] = int(OmegaConf.select(cfg, "algo.updates_per_step") or 1)
            rows.append((label, stats))

    walk = next(s for name, s in rows if name == "K1WalkFlat/motrix")
    soccer = next((s for name, s in rows if name == "K1SoccerDribble/motrix"), None)
    print(f"Benchmark: num_envs={num_envs}, warmup={num_warmup}, steps={num_steps}\n")
    print(
        f"{'task':<22} {'sim_dt':>7} {'sub':>4} {'s/step':>10} "
        f"{'env-steps/s':>14} {'#rew':>5} {'upd':>4}"
    )
    for label, s in rows:
        print(
            f"{label:<22} {s['sim_dt']:>7.4f} {s['sim_substeps']:>4} "
            f"{s['s_per_ctrl_step']:>10.4f} {s['env_steps_per_s']:>14.0f} "
            f"{s['reward_terms']:>5} {s['updates_per_step']:>4}"
        )
    if soccer is not None:
        ratio_step = soccer["s_per_ctrl_step"] / walk["s_per_ctrl_step"]
        print()
        print(f"Soccer / Walk wall-time per ctrl step: {ratio_step:.2f}x")
        print(
            f"Theory from sim_substeps only: "
            f"{soccer['sim_substeps'] / walk['sim_substeps']:.2f}x"
        )


if __name__ == "__main__":
    main()

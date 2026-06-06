"""Headless benchmark: compare sim_dt/ctrl_dt combinations for box+mesh ball on Motrix."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import motrixsim as mtx
import numpy as np

DEBUG_DIR = Path(__file__).resolve().parent
SCENE_XML = DEBUG_DIR / "scene_ball_pitch.xml"
SIM_SECONDS = 4.0
INIT_VX = 1.5


@dataclass(frozen=True)
class BallRollResult:
    sim_dt: float
    steps: int
    ok: bool
    err: str | None
    vx_samples: dict[float, float]
    z_min: float
    z_max: float
    z_p95_p5_mm: float
    vx_final: float


def _ball_state(model: mtx.SceneModel, data: mtx.SceneData) -> tuple[np.ndarray, np.ndarray]:
    link = model.get_link("ball")
    if link is None:
        raise RuntimeError("missing link 'ball'")
    pos = np.asarray(link.get_pose(data)[:3], dtype=np.float64)
    qvel = np.asarray(data.dof_vel, dtype=np.float64).reshape(-1)
    vel = qvel[:3].copy()
    return pos, vel


def run_ball_roll(sim_dt: float, init_vx: float = INIT_VX, sim_seconds: float = SIM_SECONDS) -> BallRollResult:
    model = mtx.load_model(str(SCENE_XML))
    data = mtx.SceneData(model)
    model.options.timestep = float(sim_dt)
    if model.keyframes:
        data.dof_pos[:] = model.keyframes[0].dof_pos
    else:
        data.dof_pos[:] = model.compute_init_dof_pos()
    qvel = np.zeros_like(np.asarray(data.dof_vel, dtype=np.float32))
    qvel = np.asarray(qvel, dtype=np.float32).reshape(-1)
    qvel[:3] = np.array([init_vx, 0.0, 0.0], dtype=np.float32)
    data.set_dof_vel(qvel)
    model.forward_kinematic(data)

    steps = int(round(sim_seconds / sim_dt))
    sample_times = [0.0, 1.0, 2.0, 3.0, sim_seconds]
    sample_steps = {int(round(t / sim_dt)): t for t in sample_times}
    vx_samples: dict[float, float] = {}
    zs: list[float] = []
    ok = True
    err: str | None = None

    try:
        for i in range(steps + 1):
            pos, vel = _ball_state(model, data)
            zs.append(float(pos[2]))
            if i in sample_steps:
                vx_samples[sample_steps[i]] = float(vel[0])
            if i < steps:
                model.step(data)
    except BaseException as exc:
        ok = False
        err = f"{type(exc).__name__}: {exc}"

    if zs:
        z_arr = np.asarray(zs, dtype=np.float64)
        z_min = float(z_arr.min())
        z_max = float(z_arr.max())
        z_p95_p5_mm = float((np.percentile(z_arr, 95) - np.percentile(z_arr, 5)) * 1000.0)
    else:
        z_min = z_max = z_p95_p5_mm = float("nan")

    vx_final = vx_samples.get(sim_seconds, float("nan"))
    return BallRollResult(
        sim_dt=sim_dt,
        steps=steps,
        ok=ok,
        err=err,
        vx_samples=vx_samples,
        z_min=z_min,
        z_max=z_max,
        z_p95_p5_mm=z_p95_p5_mm,
        vx_final=vx_final,
    )


ENV_SIM_SECONDS = 5.0


def run_env_standing(sim_dt: float, ctrl_dt: float, sim_seconds: float = ENV_SIM_SECONDS) -> dict[str, object]:
    from hydra import compose, initialize_config_dir

    from unilab.envs.locomotion.k1.soccer_dribble import K1SoccerDribbleEnv, K1SoccerDribbleFlatCfg
    from unilab.training.reward import resolve_reward_dict

    conf_dir = Path(__file__).resolve().parents[1] / "conf" / "appo"
    with initialize_config_dir(version_base=None, config_dir=str(conf_dir)):
        cfg = compose(config_name="config", overrides=["task=k1_soccer_dribble/motrix"])
    reward_dict = resolve_reward_dict(cfg)
    from unilab.envs.locomotion.k1.soccer_dribble import K1SoccerDribbleRewardConfig

    env_cfg = K1SoccerDribbleFlatCfg(
        sim_dt=sim_dt,
        ctrl_dt=ctrl_dt,
        reward_config=K1SoccerDribbleRewardConfig(**reward_dict),
    )
    env_cfg.validate()
    decimation = env_cfg.sim_substeps
    ctrl_steps = int(round(sim_seconds / ctrl_dt))
    env = K1SoccerDribbleEnv(env_cfg, num_envs=1, backend_type="motrix")
    obs, info = env.reset(np.array([0], dtype=np.intp))
    del obs, info

    ball_z: list[float] = []
    ball_vx: list[float] = []
    ok = True
    err: str | None = None
    num_action = int(env.action_space.shape[0])
    zero_action = np.zeros((1, num_action), dtype=np.float32)

    try:
        for _ in range(ctrl_steps):
            ball_pos_w, ball_vel_w, _, _ = env._ball_state()
            ball_z.append(float(ball_pos_w[0, 2]))
            ball_vx.append(float(ball_vel_w[0, 0]))
            state = env.step(zero_action)
            del state
    except BaseException as exc:
        ok = False
        err = f"{type(exc).__name__}: {exc}"

    if ball_z:
        z_arr = np.asarray(ball_z, dtype=np.float64)
        vx_arr = np.asarray(ball_vx, dtype=np.float64)
        summary = {
            "sim_dt": sim_dt,
            "ctrl_dt": ctrl_dt,
            "decimation": decimation,
            "ctrl_steps": ctrl_steps,
            "sim_seconds": ctrl_steps * ctrl_dt,
            "ok": ok,
            "err": err,
            "ball_z_min": float(z_arr.min()),
            "ball_z_max": float(z_arr.max()),
            "ball_z_p95_p5_mm": float((np.percentile(z_arr, 95) - np.percentile(z_arr, 5)) * 1000.0),
            "ball_z_mean": float(z_arr.mean()),
            "ball_vx_abs_max": float(np.max(np.abs(vx_arr))),
        }
    else:
        summary = {
            "sim_dt": sim_dt,
            "ctrl_dt": ctrl_dt,
            "decimation": decimation,
            "ctrl_steps": ctrl_steps,
            "ok": ok,
            "err": err,
        }
    env.close()
    return summary


def _print_ball(r: BallRollResult) -> None:
    status = "OK" if r.ok else f"FAIL ({r.err})"
    print(f"\n=== ball roll sim_dt={r.sim_dt:.4f}s ({r.steps} steps) [{status}] ===")
    for t in sorted(r.vx_samples):
        print(f"  t={t:.1f}s  vx={r.vx_samples[t]:.3f} m/s")
    print(f"  z: min={r.z_min:.5f} max={r.z_max:.5f} p95-p5={r.z_p95_p5_mm:.2f} mm")


def _print_env(s: dict[str, object]) -> None:
    status = "OK" if s.get("ok") else f"FAIL ({s.get('err')})"
    print(
        f"\n=== env standing sim_dt={s['sim_dt']:.4f} ctrl_dt={s['ctrl_dt']:.4f} "
        f"decimation={s['decimation']} ({s.get('sim_seconds', '?')}s sim) [{status}] ==="
    )
    if "ball_z_min" in s:
        print(
            f"  ball_z: min={s['ball_z_min']:.5f} max={s['ball_z_max']:.5f} "
            f"p95-p5={s['ball_z_p95_p5_mm']:.2f} mm mean={s['ball_z_mean']:.5f}"
        )
        print(f"  ball |vx| max={s['ball_vx_abs_max']:.3f} m/s")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-standing", action="store_true", help="run K1SoccerDribble standing env test")
    args = parser.parse_args()

    print(f"[benchmark] scene={SCENE_XML.name} init_vx={INIT_VX} sim_seconds={SIM_SECONDS}")

    configs = [
        (0.001, 0.004),
        (0.002, 0.020),
    ]
    for sim_dt, _ctrl in configs:
        _print_ball(run_ball_roll(sim_dt))

    # High-speed penetration stress (sim_soccer2 decider can push ball hard).
    print("\n--- high-speed stress init_vx=3.0 ---")
    for sim_dt, _ctrl in configs:
        _print_ball(run_ball_roll(sim_dt, init_vx=3.0))

    if args.env_standing:
        for sim_dt, ctrl_dt in configs:
            _print_env(run_env_standing(sim_dt, ctrl_dt))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"


def _load_play_latest_auto():
    path = _SCRIPTS_DIR / "play_latest_auto.py"
    module_name = "play_latest_auto_test"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_build_play_cmd_flashsac_preset() -> None:
    mod = _load_play_latest_auto()
    cmd = mod._build_play_cmd(
        repo_root=_SCRIPTS_DIR.parent,
        preset=mod.PRESETS["flashsac"],
        algo="flashsac",
        task_slug="k1_soccer_dribble",
        sim="motrix",
        checkpoint_path=Path("/tmp/model_9.pt"),
    )
    assert "conf/offpolicy" in cmd[3]
    assert "algo=flashsac" in cmd
    assert "task=flashsac/k1_soccer_dribble/motrix" in cmd
    assert "algo.load_run=/tmp/model_9.pt" in cmd


def test_build_play_cmd_appo_preset() -> None:
    mod = _load_play_latest_auto()
    cmd = mod._build_play_cmd(
        repo_root=_SCRIPTS_DIR.parent,
        preset=mod.PRESETS["appo"],
        algo="appo",
        task_slug="k1_soccer_dribble",
        sim="motrix",
        checkpoint_path=Path("/tmp/model_9.pt"),
    )
    assert "conf/appo" in cmd[3]
    assert "algo=appo" not in cmd
    assert "task=k1_soccer_dribble/motrix" in cmd
    assert "algo.algo_log_name=appo" in cmd
    assert "interactive.action_mode=policy" in cmd
    assert "interactive.soccer_dribble_debug=true" in cmd
    assert "+interactive.action_mode=policy" not in cmd


def test_resolve_checkpoint_arg_full_path() -> None:
    mod = _load_play_latest_auto()
    repo = _SCRIPTS_DIR.parent
    ckpt = repo / "logs/appo/K1SoccerDribble/2026-06-01_19-34-31_motrix/model_2000.pt"
    if not ckpt.is_file():
        return
    resolved = mod._resolve_checkpoint_arg(
        repo_root=repo,
        task_log_root=repo / "logs/appo/K1SoccerDribble",
        checkpoint=str(ckpt),
        load_run=None,
    )
    assert resolved == ckpt.resolve()


def test_resolve_checkpoint_arg_iteration_with_load_run() -> None:
    mod = _load_play_latest_auto()
    repo = _SCRIPTS_DIR.parent
    task_log_root = repo / "logs/appo/K1SoccerDribble"
    run_dir = task_log_root / "2026-06-01_19-34-31_motrix"
    if not (run_dir / "model_2000.pt").is_file():
        return
    resolved = mod._resolve_checkpoint_arg(
        repo_root=repo,
        task_log_root=task_log_root,
        checkpoint="2000",
        load_run="2026-06-01_19-34-31_motrix",
    )
    assert resolved == (run_dir / "model_2000.pt").resolve()

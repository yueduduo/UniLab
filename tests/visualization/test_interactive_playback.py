from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from unilab.visualization.interactive_playback import (
    KeyboardCommander,
    PlaybackControls,
    RslRlPlaybackConfig,
    RslRlPlaybackSession,
    create_rsl_rl_playback_session,
    infer_actor_input_dim_from_checkpoint_payload,
    is_appo_learner_checkpoint,
    prepare_motion_overlay_selection,
)

_VEL_LIMIT = [[-0.6, -0.4, -0.8], [1.0, 0.4, 0.8]]


class _FakeWrappedEnv:
    def __init__(self, env: Any):
        self.env = env
        self.reset_calls = 0
        self.step_calls = 0
        self.last_actions = None

    def reset(self):
        self.reset_calls += 1
        return "obs", {}

    def step(self, actions):
        self.step_calls += 1
        self.last_actions = actions
        return f"obs_{self.step_calls}", 0.0, False, {}


def _fake_env(num_envs: int = 1) -> SimpleNamespace:
    return SimpleNamespace(
        action_space=SimpleNamespace(
            shape=(2,),
            low=np.full((2,), -1.0),
            high=np.full((2,), 1.0),
        ),
        get_physics_state_snapshot=lambda: np.zeros((num_envs, 4), dtype=np.float32),
        state=SimpleNamespace(info={"motion_data": object()}),
    )


def test_playback_controls_gate_single_step_and_speed() -> None:
    controls = PlaybackControls(paused=True, speed=2.0)

    assert controls.consume_step_permission() is False
    controls.request_single_step()
    assert controls.consume_step_permission() is True
    assert controls.consume_step_permission() is False

    controls.resume()
    assert controls.consume_step_permission() is True

    controls.set_speed(0.0)
    assert controls.speed > 0.0
    controls.set_speed(4.0)
    assert controls.target_dt(0.02) == pytest.approx(0.005)


def test_playback_session_advance_respects_pause_and_single_step() -> None:
    env = _fake_env()
    wrapped = _FakeWrappedEnv(env)
    session = RslRlPlaybackSession(
        env=env,
        wrapped_env=wrapped,
        device="cpu",
        action_mode="zero",
        policy=None,
        num_envs=1,
    )
    controls = PlaybackControls(paused=True)

    session.reset()
    assert session.advance(controls) is False
    assert wrapped.step_calls == 0

    controls.request_single_step()
    assert session.advance(controls) is True
    assert wrapped.step_calls == 1
    assert torch.equal(wrapped.last_actions, torch.zeros((1, 2)))
    assert session.advance(controls) is False


def test_create_rsl_rl_playback_session_loads_checkpoint_and_runner_log_dir(tmp_path: Path) -> None:
    env = SimpleNamespace(
        obs_groups_spec={"obs": 5},
        action_space=SimpleNamespace(
            shape=(2,),
            low=np.full((2,), -1.0),
            high=np.full((2,), 1.0),
        ),
        get_physics_state_snapshot=lambda: np.zeros((1, 4), dtype=np.float32),
    )
    captured: dict[str, Any] = {}
    checkpoint_path = tmp_path / "model_10.pt"
    torch.save({"actor_state_dict": {"mlp.0.weight": torch.zeros((4, 5))}}, checkpoint_path)

    class Wrapper:
        def __init__(self, wrapped_env, *, device, policy_obs_mode):
            captured["wrapper_env"] = wrapped_env
            captured["device"] = device
            captured["policy_obs_mode"] = policy_obs_mode

        def reset(self):
            return "obs", {}

        def step(self, actions):
            return "obs", 0.0, False, {}

    class Runner:
        def __init__(self, wrapped_env, train_cfg, log_dir, device):
            captured["runner_log_dir"] = log_dir
            captured["train_cfg"] = train_cfg
            captured["runner_device"] = device

        def load(self, checkpoint, load_cfg):
            captured["checkpoint"] = checkpoint
            captured["load_cfg"] = load_cfg

        def get_inference_policy(self, *, device):
            captured["policy_device"] = device
            return lambda obs: torch.ones((1, 2))

    session, policy_obs_mode, checkpoint = create_rsl_rl_playback_session(
        playback_cfg=RslRlPlaybackConfig(
            task="MyTask",
            load_run="-1",
            checkpoint=None,
            action_mode="policy",
            policy_obs_mode="auto",
            algo_log_name="custom_ppo",
            log_root=None,
            num_envs=1,
        ),
        env_factory=lambda num_envs: env,
        algo_config={"runner": {"logger": "tensorboard"}},
        root_dir=Path("/repo"),
        device="cpu",
        checkpoint_resolver=lambda *args: str(checkpoint_path),
        checkpoint_input_dim_reader=lambda path: 5,
        entrypoint_log_root=lambda root_dir, *, algo_log_name, log_root=None: (
            Path("/tmp") / algo_log_name
        ),
        wrapper_cls=Wrapper,
        runner_cls=Runner,
        policy_obs_dims_getter=lambda spec: (5, 7),
        train_cfg_normalizer=lambda cfg: cfg,
        log=lambda message: None,
    )

    assert session.env is env
    assert policy_obs_mode == "actor"
    assert checkpoint == str(checkpoint_path)
    assert captured["runner_log_dir"] == "/tmp/custom_ppo/MyTask/play_temp"
    assert captured["checkpoint"] == str(checkpoint_path)
    assert captured["train_cfg"]["runner"]["logger"] == "none"


def test_create_rsl_rl_playback_session_rejects_missing_env() -> None:
    with pytest.raises(RuntimeError, match="Playback env factory"):
        create_rsl_rl_playback_session(
            playback_cfg=RslRlPlaybackConfig(
                task="MyTask",
                load_run="-1",
                checkpoint=None,
                action_mode="zero",
                policy_obs_mode="auto",
                algo_log_name="custom_ppo",
                log_root=None,
                num_envs=1,
            ),
            env_factory=lambda num_envs: None,
            algo_config={},
            root_dir=Path("/repo"),
            device="cpu",
            checkpoint_resolver=lambda *args: None,
            checkpoint_input_dim_reader=lambda path: None,
            entrypoint_log_root=lambda root_dir, *, algo_log_name, log_root=None: Path("/tmp"),
            wrapper_cls=object,
            runner_cls=object,
            policy_obs_dims_getter=lambda spec: (0, 0),
            train_cfg_normalizer=lambda cfg: cfg,
            log=lambda message: None,
        )


def test_keyboard_commander_nudges_stack_and_clamp_to_vel_limit() -> None:
    commander = KeyboardCommander.from_vel_limit(_VEL_LIMIT, step_lin=0.1, step_ang=0.2)
    assert commander.command.tolist() == [0.0, 0.0, 0.0]

    # Linear and angular axes stack independently.
    commander.nudge(KeyboardCommander.AXIS_VX, +1.0)
    commander.nudge(KeyboardCommander.AXIS_VYAW, +1.0)
    assert commander.command == pytest.approx([0.1, 0.0, 0.2])

    # Repeated nudges saturate at the configured velocity limits.
    for _ in range(50):
        commander.nudge(KeyboardCommander.AXIS_VX, +1.0)
        commander.nudge(KeyboardCommander.AXIS_VY, -1.0)
    assert commander.command[0] == pytest.approx(1.0)
    assert commander.command[1] == pytest.approx(-0.4)

    commander.zero()
    assert commander.command.tolist() == [0.0, 0.0, 0.0]


def test_keyboard_commander_rejects_bad_vel_limit_shape() -> None:
    with pytest.raises(ValueError, match=r"shape \(2, 3\)"):
        KeyboardCommander.from_vel_limit([[0.0, 0.0], [1.0, 1.0]])


def test_poll_motrix_nudge_keyboard_applies_arrow_and_enter() -> None:
    from unilab.visualization.interactive_playback import poll_motrix_nudge_keyboard

    class _Input:
        def __init__(self, just_pressed: set[str]):
            self._just_pressed = just_pressed

        def is_key_just_pressed(self, key: str) -> bool:
            return key in self._just_pressed

    commander = KeyboardCommander.from_vel_limit(_VEL_LIMIT, step_lin=0.1, step_ang=0.2)

    assert poll_motrix_nudge_keyboard(commander, _Input({"up"})) is True
    assert commander.command.tolist() == pytest.approx([0.1, 0.0, 0.0])

    assert poll_motrix_nudge_keyboard(commander, _Input({"down"})) is True
    assert commander.command.tolist() == pytest.approx([0.0, 0.0, 0.0])

    assert poll_motrix_nudge_keyboard(commander, _Input({"left"})) is True
    assert commander.command.tolist() == pytest.approx([0.0, 0.0, 0.2])

    assert poll_motrix_nudge_keyboard(commander, _Input({"right"})) is True
    assert commander.command.tolist() == pytest.approx([0.0, 0.0, 0.0])

    commander.nudge(KeyboardCommander.AXIS_VX, +1.0)
    assert poll_motrix_nudge_keyboard(commander, _Input({"enter"})) is True
    assert commander.command.tolist() == [0.0, 0.0, 0.0]

    assert poll_motrix_nudge_keyboard(commander, _Input(set())) is False


def test_prepare_motion_overlay_selection_filters_body_names() -> None:
    env = SimpleNamespace(
        motion_loader=object(),
        motion_sampler=object(),
        cfg=SimpleNamespace(body_names=("base", "left_foot", "right_foot")),
    )
    messages: list[str] = []

    selection = prepare_motion_overlay_selection(
        env,
        show_target_bodies=True,
        show_reward_debug=False,
        target_body_names="right_foot,missing,base",
        target_max_bodies=1,
        log=messages.append,
    )

    assert selection.enabled is True
    assert selection.selected_indices.tolist() == [2]
    assert messages == ["WARNING: body name not found in task body list: missing"]


def test_infer_actor_input_dim_supports_appo_checkpoint_payload() -> None:
    payload = {
        "actor": {"mlp.0.weight": torch.zeros((128, 98))},
        "critic": {},
        "optimizer": {},
    }
    assert is_appo_learner_checkpoint(payload) is True
    assert infer_actor_input_dim_from_checkpoint_payload(payload) == 98


def test_create_rsl_rl_playback_session_loads_appo_checkpoint(tmp_path: Path) -> None:
    env = SimpleNamespace(
        obs_groups_spec={"obs": 98, "critic": 101},
        action_space=SimpleNamespace(
            shape=(29,),
            low=np.full((29,), -1.0),
            high=np.full((29,), 1.0),
        ),
        get_physics_state_snapshot=lambda: np.zeros((1, 4), dtype=np.float32),
    )
    captured: dict[str, Any] = {}

    class Wrapper:
        def __init__(self, wrapped_env, *, device, policy_obs_mode):
            captured["policy_obs_mode"] = policy_obs_mode

        def reset(self):
            return TensorDict({"policy": torch.zeros((1, 98))}, batch_size=1), {}

        def step(self, actions):
            return TensorDict({"policy": torch.zeros((1, 98))}, batch_size=1), 0.0, False, {}

    class Runner:
        def __init__(self, *args, **kwargs):
            raise AssertionError("OnPolicyRunner should not be used for APPO checkpoints")

        def load(self, *args, **kwargs):
            raise AssertionError("OnPolicyRunner.load should not be called")

        def get_inference_policy(self, *, device):
            raise AssertionError("OnPolicyRunner policy should not be requested")

    checkpoint_path = tmp_path / "model_0.pt"
    run_config = {
        "config": {
            "algo": {
                "algo_log_name": "appo",
                "obs_groups": {"actor": {"policy": 98}},
                "actor": {
                    "class_name": "rsl_rl.models.MLPModel",
                    "hidden_dims": [128],
                    "activation": "elu",
                    "obs_normalization": False,
                    "distribution_cfg": {
                        "class_name": "rsl_rl.modules.distribution.GaussianDistribution",
                        "init_std": 1.0,
                        "std_type": "scalar",
                    },
                },
            }
        }
    }
    (tmp_path / "run_config.json").write_text(json.dumps(run_config), encoding="utf-8")
    from unilab.visualization.interactive_playback import build_appo_actor

    actor = build_appo_actor(
        run_config["config"]["algo"],
        obs_dim=98,
        critic_dim=101,
        action_dim=29,
        num_envs=1,
        device="cpu",
    )
    torch.save({"actor": actor.state_dict(), "critic": {}, "optimizer": {}}, checkpoint_path)

    session, policy_obs_mode, resolved_checkpoint = create_rsl_rl_playback_session(
        playback_cfg=RslRlPlaybackConfig(
            task="G1WalkFlat",
            load_run=str(checkpoint_path),
            checkpoint=None,
            action_mode="policy",
            policy_obs_mode="auto",
            algo_log_name="appo",
            log_root=None,
            num_envs=1,
        ),
        env_factory=lambda num_envs: env,
        algo_config={},
        root_dir=Path("/repo"),
        device="cpu",
        checkpoint_resolver=lambda *args: str(checkpoint_path),
        checkpoint_input_dim_reader=lambda path: 98,
        entrypoint_log_root=lambda root_dir, *, algo_log_name, log_root=None: Path("/tmp") / algo_log_name,
        wrapper_cls=Wrapper,
        runner_cls=Runner,
        policy_obs_dims_getter=lambda spec: (98, 98),
        train_cfg_normalizer=lambda cfg: cfg,
        log=lambda message: captured.setdefault("logs", []).append(message),
    )

    assert policy_obs_mode == "actor"
    assert resolved_checkpoint == str(checkpoint_path)
    assert session.policy is not None
    actions = session.policy(TensorDict({"policy": torch.zeros((1, 98))}, batch_size=1))
    assert actions.shape == (1, 29)
    assert any("APPO learner checkpoint" in message for message in captured["logs"])

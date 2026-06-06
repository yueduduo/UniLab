"""K1SoccerDribble (push) obs contract and curriculum reward tests."""

from __future__ import annotations

import pytest

from unilab.envs.locomotion.k1.constants import (
    K1_NUM_ACTION,
    K1_SOCCER_CURRICULUM_BALL_SPAWN_XY,
    K1_SOCCER_CURRICULUM_PHASE1_DISTANCE_M,
    K1_SOCCER_CURRICULUM_PHASE1_WAYPOINT_XY,
    K1_SOCCER_CURRICULUM_PHASE2_DISTANCE_M,
    K1_SOCCER_PHASE2_REWARD_KEYS,
    k1_keyframe_robot_joint_qpos,
    k1_soccer_push_actor_obs_dim,
    k1_soccer_push_critic_obs_dim,
)
from unilab.envs.locomotion.k1.soccer_dribble import K1SoccerDribbleCfg


def test_k1_soccer_push_obs_dims():
    actor_dim = k1_soccer_push_actor_obs_dim(K1_NUM_ACTION)
    assert actor_dim == 80
    assert k1_soccer_push_critic_obs_dim(K1_NUM_ACTION) == 83
    assert K1SoccerDribbleCfg().obs_frame_stack == 1


def test_k1_soccer_curriculum_layout_matches_scene():
    assert K1_SOCCER_CURRICULUM_PHASE1_WAYPOINT_XY == (1.0, 0.0)
    assert K1_SOCCER_CURRICULUM_BALL_SPAWN_XY == (1.2, 0.0)
    assert K1_SOCCER_CURRICULUM_PHASE1_DISTANCE_M == 1.0
    assert K1_SOCCER_CURRICULUM_PHASE2_DISTANCE_M == 0.2


def test_k1_soccer_phase2_reward_keys_are_ball_or_dribble_only():
    assert "tracking_lin_vel" not in K1_SOCCER_PHASE2_REWARD_KEYS
    assert "phase1_complete" not in K1_SOCCER_PHASE2_REWARD_KEYS
    assert "ball_progress" in K1_SOCCER_PHASE2_REWARD_KEYS
    assert "feet_step_travel" in K1_SOCCER_PHASE2_REWARD_KEYS


def test_soccer_default_angles_use_robot_keyframe_not_ball_tail():
    import numpy as np

    qpos = np.zeros(7 + K1_NUM_ACTION + 7, dtype=np.float32)
    qpos[7 : 7 + K1_NUM_ACTION] = np.linspace(0.1, 0.5, K1_NUM_ACTION, dtype=np.float32)
    qpos[-7:] = np.array([1.2, 0.0, 0.09, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    robot = k1_keyframe_robot_joint_qpos(qpos, K1_NUM_ACTION)
    assert robot.shape == (K1_NUM_ACTION,)
    assert not np.any(np.isclose(robot, qpos[-K1_NUM_ACTION :]))


def test_phase1_keeps_walk_reset_commands():
    import numpy as np

    from unilab.envs.locomotion.k1.soccer_dribble import K1SoccerDribbleEnv

    env = K1SoccerDribbleEnv.__new__(K1SoccerDribbleEnv)
    env._phase1_reached = np.array([False, False], dtype=bool)
    walk_cmd = np.array([[0.4, 0.0, 0.0], [0.5, 0.1, 0.0]], dtype=np.float32)
    info = {"commands": walk_cmd.copy()}
    cmd = env._command_for_curriculum(info)
    np.testing.assert_allclose(cmd, walk_cmd)


def test_soccer_reset_does_not_randomize_yaw():
    import numpy as np

    from unilab.envs.locomotion.k1.soccer_dribble import K1SoccerDomainRandomizationProvider

    provider = K1SoccerDomainRandomizationProvider()
    yaw = provider._sample_reset_yaw(None, 8)
    np.testing.assert_allclose(yaw, 0.0)


def test_soccer_reset_randomizes_x_only():
    import numpy as np

    from unilab.envs.locomotion.k1.soccer_dribble import K1SoccerDomainRandomizationProvider

    provider = K1SoccerDomainRandomizationProvider()
    np.random.seed(0)
    offset = provider._sample_reset_xy_offset(None, 512)
    assert offset.shape == (512, 2)
    np.testing.assert_allclose(offset[:, 1], 0.0)
    assert float(np.min(offset[:, 0])) >= -0.5
    assert float(np.max(offset[:, 0])) <= 0.5
    assert float(np.std(offset[:, 0])) > 0.1


@pytest.mark.parametrize(
    "task_owner,expected_sim_backend,expected_num_envs",
    [
        ("flashsac/k1_soccer_dribble/motrix", "motrix", 1024),
        ("flashsac/k1_soccer_dribble/mujoco", "mujoco", 4096),
    ],
)
def test_k1_soccer_dribble_flashsac_owner_uses_walk_flashsac_reward_base(
    task_owner, expected_sim_backend, expected_num_envs
):
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    conf_dir = pytest.importorskip("pathlib").Path(__file__).resolve().parents[4] / "conf" / "offpolicy"
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(conf_dir), version_base="1.3"):
        cfg = compose("config", overrides=[f"task={task_owner}"])
    assert cfg.training.sim_backend == expected_sim_backend
    assert cfg.algo.num_envs == expected_num_envs
    assert cfg.reward.scales.tracking_lin_vel == pytest.approx(2.0)
    assert cfg.reward.scales.feet_phase == pytest.approx(5.0)
    assert cfg.reward.scales.alive == pytest.approx(10.0)
    assert cfg.reward.scales.ball_progress == pytest.approx(1.7)
    assert cfg.reward.scales.phase1_complete == pytest.approx(1.0)
    assert cfg.reward.scales.penalty_orientation == pytest.approx(-15.0)
    assert cfg.env.curriculum.enabled is True
    assert cfg.env.control_config.action_scale == pytest.approx(0.35)
    assert cfg.env.reset_base_qvel_limit == pytest.approx(0.5)
    assert cfg.algo.warm_start.enabled is False
    assert cfg.env.noise_config.scale_joint_vel == pytest.approx(0.1)
    assert cfg.env.commands.vel_limit[0][0] == pytest.approx(0.4)
    assert cfg.env.commands.vel_limit[1][0] == pytest.approx(0.7)
    assert cfg.reward.min_forward_speed_for_gait_reward == pytest.approx(0.05)


def test_phase1_complete_reward_while_alive_in_phase2():
    import numpy as np

    from unilab.envs.locomotion.common.rewards import RewardContext
    from unilab.envs.locomotion.k1.soccer_dribble import K1SoccerDribbleEnv

    env = K1SoccerDribbleEnv.__new__(K1SoccerDribbleEnv)
    ctx = RewardContext(
        info={
            "phase1_reached": np.array([0.0, 1.0, 1.0], dtype=np.float32),
            "terminated": np.array([0.0, 0.0, 1.0], dtype=np.float32),
        },
        linvel=np.zeros((3, 3), dtype=np.float32),
        gyro=np.zeros((3, 3), dtype=np.float32),
        dof_pos=np.zeros((3, 1), dtype=np.float32),
        num_envs=3,
    )
    rew = env._reward_phase1_complete(ctx)
    np.testing.assert_allclose(rew, [0.0, 1.0, 0.0])


def test_phase1_complete_triggers_once_when_entering_waypoint():
    import numpy as np

    from unilab.envs.locomotion.k1.constants import (
        K1_SOCCER_PHASE1_WAYPOINT_Z,
    )
    from unilab.envs.locomotion.k1.soccer_dribble import K1SoccerDribbleEnv

    env = K1SoccerDribbleEnv.__new__(K1SoccerDribbleEnv)
    env._phase1_reached = np.array([False, True], dtype=bool)
    env._phase1_waypoint_center = np.array([1.0, 0.0, K1_SOCCER_PHASE1_WAYPOINT_Z], dtype=np.float32)
    env._phase1_visual_reached = False
    env._waypoint_visual_supported = False

    class _Backend:
        def get_base_pos(self) -> np.ndarray:
            return np.array(
                [[1.0, 0.0, K1_SOCCER_PHASE1_WAYPOINT_Z], [1.0, 0.0, K1_SOCCER_PHASE1_WAYPOINT_Z]],
                dtype=np.float32,
            )

    env._backend = _Backend()
    newly = env._update_phase1_reached()
    np.testing.assert_array_equal(newly, [True, False])
    np.testing.assert_array_equal(env._phase1_reached, [True, True])

    newly_again = env._update_phase1_reached()
    np.testing.assert_array_equal(newly_again, [False, False])


def test_phase1_masks_ball_obs_until_waypoint_reached():
    import numpy as np

    from unilab.envs.locomotion.k1.soccer_dribble import K1SoccerDribbleEnv

    env = K1SoccerDribbleEnv.__new__(K1SoccerDribbleEnv)
    env._phase1_reached = np.array([False, True], dtype=bool)

    def fake_ball(env_ids=None):
        full = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        if env_ids is None:
            return full
        return full[np.asarray(env_ids, dtype=np.intp)]

    env._ball_obs = fake_ball  # type: ignore[method-assign]
    masked = env._ball_obs_for_curriculum()
    np.testing.assert_allclose(masked[0], 0.0)
    np.testing.assert_allclose(masked[1], [4.0, 5.0, 6.0])
    np.testing.assert_allclose(env._ball_obs_for_curriculum(np.array([0], dtype=np.int32)), [[0.0, 0.0, 0.0]])


def test_soccer_waypoint_visual_sync_skips_when_backend_unsupported():
    import numpy as np

    from unilab.envs.locomotion.k1.soccer_dribble import K1SoccerDribbleEnv

    class _Backend:
        def set_world_geom_pos(self, geom_name: str, pos: np.ndarray) -> None:
            raise NotImplementedError

    env = K1SoccerDribbleEnv.__new__(K1SoccerDribbleEnv)
    env._waypoint_visual_supported = True
    env._phase1_visual_reached = False
    env._phase1_waypoint_hidden_pos = np.zeros(3)
    env._phase1_waypoint_active_pos = np.ones(3)
    env._backend = _Backend()
    env._sync_phase1_waypoint_visual(reached=True)
    assert env._waypoint_visual_supported is False


def test_soccer_waypoint_visual_sync_skips_when_geom_missing_in_physics_model():
    import numpy as np

    from unilab.envs.locomotion.k1.soccer_dribble import K1SoccerDribbleEnv

    class _Backend:
        def set_world_geom_pos(self, geom_name: str, pos: np.ndarray) -> None:
            raise ValueError(f"Geom '{geom_name}' not found in MuJoCo model")

    env = K1SoccerDribbleEnv.__new__(K1SoccerDribbleEnv)
    env._waypoint_visual_supported = True
    env._phase1_visual_reached = True
    env._phase1_waypoint_hidden_pos = np.zeros(3)
    env._phase1_waypoint_active_pos = np.ones(3)
    env._backend = _Backend()
    env._sync_phase1_waypoint_visual(reached=False)
    assert env._waypoint_visual_supported is False


@pytest.mark.slow
def test_k1_soccer_dribble_motrix_reset_step_obs_shape():
    pytest.importorskip("motrix")
    import numpy as np

    from unilab.base import registry
    from unilab.base.registry import ensure_registries
    from unilab.envs.locomotion.k1.soccer_dribble import K1SoccerDribbleRewardConfig

    ensure_registries()
    reward_config = K1SoccerDribbleRewardConfig(
        scales={"ball_progress": 1.0, "alive": 0.0},
        tracking_sigma=0.25,
        gait_frequency=1.5,
        feet_phase_swing_height=0.06,
        feet_phase_tracking_sigma=0.008,
        base_height_target=0.55,
        min_base_height=0.25,
        max_tilt_deg=55.0,
        pose_weights=[1.0] * K1_NUM_ACTION,
        ball_keep_distance=0.45,
        ball_keep_sigma=0.08,
        ball_front_lateral_sigma=0.16,
        ball_speed_sigma=0.12,
        ball_lost_distance=1.35,
        ball_lost_distance_hard=2.8,
        ball_speed_cap=0.35,
        ball_move_speed_target=0.20,
        ball_move_speed_sigma=0.15,
        ball_approach_speed_cap=1.0,
        ball_still_speed_threshold=0.06,
        ball_cmd_deadzone=0.12,
        ball_cmd_kp=0.9,
        fixed_cmd_lin_speed=0.3,
        feet_step_travel_cap=0.10,
    )
    env = registry.make(
        "K1SoccerDribble",
        sim_backend="motrix",
        num_envs=2,
        env_cfg_override={"reward_config": reward_config, "obs_frame_stack": 1},
    )
    actor_dim = k1_soccer_push_actor_obs_dim(env.action_space.shape[0])
    critic_dim = k1_soccer_push_critic_obs_dim(env.action_space.shape[0])
    assert env.obs_groups_spec == {"obs": actor_dim, "critic": critic_dim}
    env.init_state()
    obs, _info = env.reset(np.arange(2, dtype=np.int32))
    assert obs["obs"].shape == (2, actor_dim)
    assert obs["critic"].shape == (2, critic_dim)
    actions = np.zeros((2, env.action_space.shape[0]), dtype=np.float32)
    state = env.step(actions)
    assert state.obs["obs"].shape == (2, actor_dim)
    env.close()

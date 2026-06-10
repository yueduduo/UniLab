"""K1 soccer dribble — deploy-compatible observation layout (46000-aligned, 78-dim actor).

Actor obs (78 dims) — identical to MOS-SIM ``k1_model_46000.pt`` deployment contract:

  [0:3]   base_lin_vel  (body frame)        × 1.0   ← OBS_SCALE["base_lin_vel"]
  [3:6]   base_ang_vel  (body frame)        × 0.2   ← OBS_SCALE["base_ang_vel"]
  [6:9]   gravity_orientation (body frame)  × 1.0   downward gravity direction
  [9:12]  cmd [vx, vy, yaw_rate]            × 1.0   ball-directed command from env
  [12:34] joint_pos − default_angles        × 1.0   (K1_DEPLOY_JOINT_ORDER)
  [34:56] joint_vel                         × 0.05  (K1_DEPLOY_JOINT_ORDER)
  [56:78] last_action                       × 1.0   (K1_DEPLOY_JOINT_ORDER)

No ``gait_phase`` (absent in 46000). No ``ball_obs`` in actor.

Critic obs (81 dims) — actor + privileged ball state for SAC value estimation:

  [0:78]  actor obs (as above)
  [78:81] ball: [rel_x_b, rel_y_b, rel_vx_b]  (body frame, always real values)

Ball signal in actor
--------------------
The policy does **not** directly observe the ball.  Ball information is encoded
in ``cmd``:

* Phase 1: cmd is a sampled forward walking command (pure locomotion).
* Phase 2: cmd is ``fixed_cmd_lin_speed × unit_direction_toward_ball`` (clipped to
  vel_limit), matching the single-robot Decider logic in MOS-SIM.  The policy
  learns to follow ball-directed velocity commands, achieving dribbling behaviour
  without requiring an explicit ball feature in the obs.

Deployment compatibility
------------------------
The 78-dim actor obs can be fed directly to the network using the same adapter
code that builds ``_obs_k1_fullbody_for_hybrid`` in MOS-SIM, provided:

1. ``include_base_lin_vel_obs=True`` (already true for the 46000 slot).
2. ``policy_joint_names = K1_DEPLOY_JOINT_ORDER`` (same as ``K1_JOINTS_POLICY_ORDER``
   in MOS-SIM ``runtime_config.py``).
3. The cmd buffer is populated with ball-directed velocity from the Decider.

Joint order contract
--------------------
* Backend I/O (MuJoCo) uses ``K1_ACTUATOR_JOINT_ORDER``.
* Policy obs/action use ``K1_DEPLOY_JOINT_ORDER``.
* ``K1_ACTUATOR_TO_DEPLOY_PERM``:  obs_d  = obs_a [:, perm]
* ``K1_DEPLOY_TO_ACTUATOR_PERM``:  act_a  = act_d [:, inv_perm]
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from unilab.base import registry
from unilab.base.np_env import NpEnvState
from unilab.dtype_config import get_global_dtype
from unilab.envs.locomotion.k1.constants import (
    K1_ACTUATOR_TO_DEPLOY_PERM,
    K1_DEPLOY_TO_ACTUATOR_PERM,
    K1_NUM_ACTION,
    k1_soccer_dribble_compat_actor_obs_dim,
    k1_soccer_dribble_compat_critic_obs_dim,
)
from unilab.envs.locomotion.k1.soccer_dribble import (
    K1SoccerDribbleCfg,
    K1SoccerDribbleEnv,
    K1SoccerDribbleRewardConfig,
)

# OBS_SCALE values mirror MOS-SIM runtime_config.py OBS_SCALE exactly.
_LIN_VEL_SCALE: float = 1.0
_ANG_VEL_SCALE: float = 0.2
_GRAVITY_SCALE: float = 1.0
_CMD_SCALE: float = 1.0
_JOINT_POS_SCALE: float = 1.0
_JOINT_VEL_SCALE: float = 0.05
_LAST_ACTION_SCALE: float = 1.0


@dataclass
class K1SoccerDribbleCompatCfg(K1SoccerDribbleCfg):
    """Soccer-dribble cfg with 46000-aligned actor obs (78 dims).

    Inherits scene, DR, reward, and curriculum from ``K1SoccerDribbleCfg``.
    ``obs_frame_stack`` is locked to 1.
    """

    obs_frame_stack: int = 1


class K1SoccerDribbleCompatEnv(K1SoccerDribbleEnv):
    """Soccer dribble with k1_model_46000-compatible 78-dim actor obs.

    Only ``obs_groups_spec``, ``_compute_obs``, and ``apply_action`` are
    overridden.  All reward functions, curriculum logic, and termination
    conditions are identical to ``K1SoccerDribbleEnv``.
    """

    _cfg: K1SoccerDribbleCompatCfg

    def __init__(
        self,
        cfg: K1SoccerDribbleCompatCfg,
        num_envs: int = 1,
        backend_type: str = "mujoco",
    ) -> None:
        super().__init__(cfg, num_envs=num_envs, backend_type=backend_type)
        self._actor_obs_dim = k1_soccer_dribble_compat_actor_obs_dim(self._num_action)
        self._critic_obs_dim = k1_soccer_dribble_compat_critic_obs_dim(self._num_action)
        self._a2d_perm = np.asarray(K1_ACTUATOR_TO_DEPLOY_PERM, dtype=np.intp)
        self._d2a_perm = np.asarray(K1_DEPLOY_TO_ACTUATOR_PERM, dtype=np.intp)
        if int(self._cfg.obs_frame_stack) != 1:
            raise ValueError("K1SoccerDribbleCompat requires obs_frame_stack=1")

    # ------------------------------------------------------------------
    # Contract
    # ------------------------------------------------------------------

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        return {"obs": self._actor_obs_dim, "critic": self._critic_obs_dim}

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _compute_obs(
        self,
        info: dict,
        linvel,
        gyro,
        gravity,
        dof_pos,
        dof_vel,
        *,
        env_ids: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Build 78-dim actor obs (46000-identical) + 81-dim critic obs.

        Actor:  [linvel | angvel×0.2 | gravity | cmd | jp | jv×0.05 | la] — 78 dims.
        Critic: actor + [ball_rel_x_b, ball_rel_y_b, ball_rel_vx_b]         — 81 dims.

        ``dof_pos`` / ``dof_vel`` arrive in ``K1_ACTUATOR_JOINT_ORDER``;
        they are reordered to ``K1_DEPLOY_JOINT_ORDER`` before concatenation.

        ``gravity`` is the upvector sensor (≈[0,0,1] when upright); we negate
        it to get the downward gravity direction matching 46000's convention.

        Ball state in critic uses real values at all times (no curriculum masking)
        so the value network always has privileged access to ball position.
        """
        num_rows = linvel.shape[0]
        diff = dof_pos - self.default_angles   # actuator order, (N, 22)
        command = self._command_for_curriculum(info, env_ids)
        last_actions = info.get(
            "current_actions",
            np.zeros((num_rows, self._num_action), dtype=get_global_dtype()),
        )

        # Reorder: actuator → deploy
        diff_d = diff[:, self._a2d_perm]
        dof_vel_d = dof_vel[:, self._a2d_perm]
        last_actions_d = last_actions[:, self._a2d_perm]

        # upvector → downward gravity direction (matches 46000)
        gravity_down = -np.asarray(gravity, dtype=get_global_dtype())

        actor = np.concatenate(
            [
                np.asarray(linvel,        dtype=get_global_dtype()) * _LIN_VEL_SCALE,
                np.asarray(gyro,          dtype=get_global_dtype()) * _ANG_VEL_SCALE,
                gravity_down                                        * _GRAVITY_SCALE,
                np.asarray(command,       dtype=get_global_dtype()) * _CMD_SCALE,
                np.asarray(diff_d,        dtype=get_global_dtype()) * _JOINT_POS_SCALE,
                np.asarray(dof_vel_d,     dtype=get_global_dtype()) * _JOINT_VEL_SCALE,
                np.asarray(last_actions_d,dtype=get_global_dtype()) * _LAST_ACTION_SCALE,
            ],
            axis=1,
        )  # (N, 78)

        # Critic: actor + privileged ball state (real values, no curriculum mask)
        ball_priv = np.asarray(self._ball_obs(env_ids), dtype=get_global_dtype())
        critic = np.concatenate([actor, ball_priv], axis=1)  # (N, 81)

        return {"obs": actor, "critic": critic}

    # ------------------------------------------------------------------
    # Action
    # ------------------------------------------------------------------

    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> np.ndarray:
        """Accept policy actions in ``K1_DEPLOY_JOINT_ORDER``, map to actuator order.

        ``state.info["current_actions"]`` stores actions in deploy order so that
        ``last_actions_d`` in ``_compute_obs`` reads them back without extra permutation.
        Gait phase is still updated for the feet-phase reward functions.
        """
        state.info["last_actions"] = state.info.get(
            "current_actions",
            np.zeros_like(actions),
        )
        state.info["current_actions"] = actions  # deploy order

        gait_phase = state.info.get(
            "gait_phase",
            np.zeros((self._num_envs, 2), dtype=get_global_dtype()),
        )
        gait_phase[:, 0] = (gait_phase[:, 0] + self._gait_phase_delta) % (2.0 * math.pi)
        gait_phase[:, 1] = (gait_phase[:, 1] + self._gait_phase_delta) % (2.0 * math.pi)
        state.info["gait_phase"] = gait_phase

        # permute deploy → actuator order for PD control
        actions_actuator = actions[:, self._d2a_perm]
        return (
            np.asarray(actions_actuator, dtype=get_global_dtype())
            * self._cfg.control_config.action_scale
            + self.default_angles
        )


@registry.envcfg("K1SoccerDribbleCompat")
@dataclass
class K1SoccerDribbleCompatFlatCfg(K1SoccerDribbleCompatCfg):
    reward_config: K1SoccerDribbleRewardConfig | None = None


registry.register_env("K1SoccerDribbleCompat", K1SoccerDribbleCompatEnv, sim_backend="mujoco")
registry.register_env("K1SoccerDribbleCompat", K1SoccerDribbleCompatEnv, sim_backend="motrix")

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from unilab.envs.locomotion.common.base import (
    BaseNoiseConfig,
    ControlConfigBase,
    LocomotionBaseCfg,
    LocomotionBaseEnv,
)
from unilab.envs.locomotion.common.base import (
    Sensor as LocomotionSensor,
)


@dataclass
class NoiseConfig(BaseNoiseConfig):
    scale_joint_angle: float = 0.01
    scale_joint_vel: float = 1.5
    scale_gyro: float = 0.2


@dataclass
class ControlConfig(ControlConfigBase):
    action_scale: float | np.ndarray = 0.25  # type: ignore[assignment]


@dataclass
class Sensor(LocomotionSensor):
    local_linvel: str = "trunk_local_linvel"
    gyro: str = "trunk_gyro"
    upvector: str = "trunk_upvector"


@dataclass
class Asset:
    base_name = "Trunk"
    foot_name = "left_foot_link"
    ground = "floor"


@dataclass
class K1BaseCfg(LocomotionBaseCfg):
    noise_config: NoiseConfig = field(default_factory=NoiseConfig)  # type: ignore[assignment]
    control_config: ControlConfig = field(default_factory=ControlConfig)  # type: ignore[assignment]
    sensor: Sensor = field(default_factory=Sensor)
    asset: Asset = field(default_factory=Asset)
    sim_dt: float = 0.005
    ctrl_dt: float = 0.02


class K1BaseEnv(LocomotionBaseEnv):
    _cfg: K1BaseCfg
    _keyframe_name = "stand"
    _use_global_dtype = False

    def _obs_noise(self, data: np.ndarray, scale: float) -> np.ndarray:
        return np.asarray(super()._obs_noise(data, scale), dtype=data.dtype)

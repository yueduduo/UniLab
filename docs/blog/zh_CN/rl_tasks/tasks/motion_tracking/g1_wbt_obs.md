---
title: "G1 WBT Obs"
slug: "g1_wbt_obs"
category: "motion_tracking"
robot: "g1"
registered_envs: [G1WBTObs]
---

# G1 WBT Obs

## 任务目标

提供 Whole-Body Tracking 观测口径的 SAC 任务，用于对齐全身跟踪训练所需的严格 obs 结构。

![G1 WBT Obs 场景渲染](../../images/tasks/g1_wbt_obs.png)

> 图片由 `_tools/render_static_images.py` 使用 `src/unilab/assets/robots/g1/scene_flat.xml` 的 MuJoCo scene 离屏渲染生成；如果当前环境无法启用 EGL/OSMesa，生成 manifest 会记录失败原因。

## 证据索引

| 项目 | 内容 |
| --- | --- |
| 任务类别 | 动作追踪 |
| 机器人 | g1 |
| 代表 scene | src/unilab/assets/robots/g1/scene_flat.xml |
| 代表源码 | src/unilab/envs/motion_tracking/g1/tracking_obs.py |
| 代表 owner YAML | conf/offpolicy/task/sac/g1_wbt_obs/mujoco.yaml |
| 动作维度 | 29 |

### Registry 与 Owner

| 注册名 | 后端 | 配置类 | 配置源码 |
| --- | --- | --- | --- |
| G1WBTObs | motrix, mujoco | G1WBTObsCfg | src/unilab/envs/motion_tracking/g1/tracking_obs.py |

| 算法组 | 后端 | training.task_name | owner YAML |
| --- | --- | --- | --- |
| sac | mujoco | G1WBTObs | conf/offpolicy/task/sac/g1_wbt_obs/mujoco.yaml |

## 代码级执行流

这部分不再用通用关键词套模板，而是为 `g1_wbt_obs` 单独写源码导读。源码片段只负责提供证据；每节说明都围绕本任务自己的配置、状态、观测、动作和 reward 语义展开。

| 阶段 | 证据位置 | 代码级说明 |
| --- | --- | --- |
| 1. Hydra owner | conf/offpolicy/task/sac/g1_wbt_obs/mujoco.yaml | 训练命令先 compose owner YAML。这里确定 `training.task_name`、后端身份、obs_groups、reward scale 和 env override。 |
| 2. Registry 构造 Env | src/unilab/envs/motion_tracking/g1/tracking_obs.py | `@registry.envcfg` 注册配置类，`@registry.env` 注册后端实现类；训练脚本只调用 registry，不直接 new 任务类。 |
| 3. Reset/初始状态 | src/unilab/assets/robots/g1/scene_flat.xml | env 从 scene keyframe 或 motion/grasp cache 取初态，再通过 reset randomization 改写 qpos/qvel/info。 |
| 4. Obs dict | src/unilab/envs/motion_tracking/g1/tracking_obs.py | env 读取 backend 传感器和 info，返回 dict；learner 根据 `obs_groups_spec` 和 owner `algo.obs_groups` 选择 actor/critic 输入。 |
| 5. Action 到控制目标 | src/unilab/envs/motion_tracking/g1/tracking_obs.py | policy 输出先按 action scale / 控制顺序映射到 actuator，再由 backend step 推进仿真。 |
| 6. Reward dispatch | src/unilab/envs/motion_tracking/g1/tracking_obs.py | 源码把 reward key 映射到函数，owner YAML 的 scale 决定每项奖励/惩罚是否启用以及权重。 |
| 7. Termination | src/unilab/envs/motion_tracking/g1/tracking_obs.py | env 根据高度、姿态、接触、参考轨迹误差、物体状态或能量阈值生成 done/terminated。 |

### 本任务阅读重点

| 阅读重点 |
| --- |
| G1WBTObs 的价值在观测口径，不是新动作或新机器人：它为 SAC/WBT 对齐 deploy-like obs/history。 |
| 要读 tracking_obs.py 中的 cfg、reset provider 和 `_compute_obs`，因为它重写了观测结构。 |
| reward 在 tracking 基础上扩展，obs contract 是这个任务文档的重点。 |

### 本任务注册入口

| 注册名 | 配置类 | 可用后端 |
| --- | --- | --- |
| G1WBTObs | G1WBTObsCfg | motrix, mujoco |

### 本任务源码锚点

| 小节 | 源码文件 | 明确锚点 |
| --- | --- | --- |
| 配置：WBT obs | src/unilab/envs/motion_tracking/g1/tracking_obs.py | G1WBTObsCfg, G1WBTObs |
| Reset：WBT 随机化 | src/unilab/envs/motion_tracking/g1/tracking_obs.py | build_reset_plan, WBT |
| Obs：严格观测结构 | src/unilab/envs/motion_tracking/g1/tracking_obs.py | _compute_obs |
| Action：SAC tracking action | src/unilab/envs/motion_tracking/g1/tracking.py | apply_action, simulate_action_latency |
| Reward/Termination | src/unilab/envs/motion_tracking/g1/tracking_obs.py | _init_reward_functions |

## 关键源码逐段解释（按本任务手写）

### 配置：WBT obs

| 本任务人工导读 |
| --- |
| cfg 单独注册 WBT 观测任务，并设置 obs/history/domain randomization 字段。 |
| 它继承 SAC tracking cfg，但不是普通 PPO tracking 页面。 |

```python
# src/unilab/envs/motion_tracking/g1/tracking_obs.py
# WBT obs 是 SAC tracking 的部署观测变体，单独注册为 G1WBTObs。
@registry.envcfg("G1WBTObs")
@dataclass
class G1WBTObsCfg(G1MotionTrackingSACCfg):
    """SAC whole-body tracking with sim2real obs / DR / reward extensions."""

    # noise_config 控制 actor 观测裁剪、anchor 噪声和 proprio history。
    noise_config: ObsNoiseConfig = field(default_factory=ObsNoiseConfig)  # type: ignore[assignment]
    # domain_rand 扩展 encoder bias、足底摩擦和 y/z COM 随机化。
    domain_rand: ObsDomainRand = field(default_factory=ObsDomainRand)  # type: ignore[assignment]


# --------------------------------------------------------------------------- #
# DR provider extension
# --------------------------------------------------------------------------- #


```
### Reset：WBT 随机化

| 本任务人工导读 |
| --- |
| reset 在 motion reference 上叠加 WBT 专用随机化。 |
| 默认关节偏置、摩擦等扰动写入 ResetPlan/info。 |

```python
# src/unilab/envs/motion_tracking/g1/tracking_obs.py
    def build_reset_plan(self, env: Any, env_ids: np.ndarray) -> ResetPlan:
        num_reset = len(env_ids)
        # WBT reset 仍以 motion sampler 采样的参考帧作为初始状态。
        motion_frames = env.motion_sampler.sample_frames(env_ids)
        # motion_data 提供参考 root/body/joint 状态。
        motion_data = env.motion_loader.get_motion_at_frame(motion_frames)
        qpos, qvel = _build_motion_reference_state(env, env_ids, motion_data)

        info_updates: dict[str, Any] = {
            # 动作历史清零，保证 obs history 和 action_rate 从新 episode 开始。
            "current_actions": zero_actions(num_reset, env._num_action),
            "last_actions": zero_actions(num_reset, env._num_action),
            # Seed prev_dof_vel with the post-reset joint velocity so the first
            # joint_acc_l2 sample is physically meaningful (Δv from the new
            # starting velocity, not a spurious step from pre-termination).
            # 用 reset 后 qvel 初始化 prev_dof_vel，首帧 joint_acc_l2 不会被上个 episode 污染。
            "prev_dof_vel": qvel[:, 6:].astype(get_global_dtype()),
        }

        dr_cfg = env.cfg.domain_rand
        if getattr(dr_cfg, "enable_encoder_bias", False):
            low, high = dr_cfg.encoder_bias_range
            # encoder bias 只写入 actor 观测的 joint_pos_rel，模拟实机编码器零点偏差。
            info_updates["joint_pos_obs_bias"] = np.random.uniform(
                low, high, size=(num_reset, env._num_action)
            ).astype(get_global_dtype())

        # 复用通用 reset DR 生成质量、COM x、kp/kd 等 payload。
        randomization = build_common_reset_randomization(
            env, num_reset, base_kp=self._base_kp, base_kd=self._base_kd
        )

        # Foot-geom friction.
        if getattr(dr_cfg, "randomize_geom_friction", False):
            assert self._base_geom_friction is not None
            assert self._foot_geom_ids is not None
            payload = randomization or ResetRandomizationPayload()
            low, high = dr_cfg.friction_range
            # 每个 env 采样一个足底摩擦缩放，广播到所有匹配 foot geoms。
            scale = np.random.uniform(low, high, size=(num_reset, 1)).astype(np.float64)
            geom_friction = np.broadcast_to(
                self._base_geom_friction,
                (num_reset, *self._base_geom_friction.shape),
            ).copy()
            geom_friction[:, self._foot_geom_ids, 0] = scale * np.ones(
                (1, self._foot_geom_ids.size)
            )
            payload.geom_friction = geom_friction
            randomization = payload

        # y / z COM offsets, layered on top of parent's x-only common build.
        has_com_y = getattr(dr_cfg, "randomize_com_y", False)
        has_com_z = getattr(dr_cfg, "randomize_com_z", False)
        if has_com_y or has_com_z:
            payload = randomization or ResetRandomizationPayload()
            com_offset = payload.base_com_offset
            if com_offset is None:
                com_offset = np.zeros((num_reset, 3), dtype=np.float64)
            if has_com_y:
                low, high = dr_cfg.com_offset_y
                # 在通用 x COM 随机化之外，WBT 额外允许 y 方向质心偏移。
                com_offset[:, 1] = np.random.uniform(low, high, size=(num_reset,))
            if has_com_z:
                low, high = dr_cfg.com_offset_z
                # z 方向质心偏移用于覆盖上身/负载高度变化。
                com_offset[:, 2] = np.random.uniform(low, high, size=(num_reset,))
            payload.base_com_offset = com_offset
            randomization = payload

        # ResetPlan 同时携带参考状态、obs/reward info 和 WBT 扩展随机化。
        return ResetPlan(
            env_ids=env_ids,
            qpos=qpos,
            qvel=qvel,
            info_updates=info_updates,
            randomization=randomization,
        )


# --------------------------------------------------------------------------- #
# Env
# --------------------------------------------------------------------------- #


```
### Obs：严格观测结构

| 本任务人工导读 |
| --- |
| _compute_obs 是本任务核心，组织 proprio history 和 critic 特权信息。 |
| 文档应优先解释字段来源和维度，而不是只列 reward。 |

```python
# src/unilab/envs/motion_tracking/g1/tracking_obs.py
    def _compute_obs(
        self,
        info: dict,
        motion_data: Any,
        linvel: np.ndarray,
        gyro: np.ndarray,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        robot_body_pos_w: np.ndarray,
        robot_body_quat_w: np.ndarray,
    ) -> dict[str, np.ndarray]:
        # Stash so the overridden ``_build_actor_obs`` (called inside super)
        # can read env_ids / joint_pos_obs_bias without a signature change.
        # 临时保存 info，让重写的 _build_actor_obs 能拿到 env_ids 和 encoder bias。
        self._obs_compute_info = info
        try:
            # 先复用父类 tracking obs 构造，WBT 只在 actor obs hook 中裁剪/加 history。
            obs = super()._compute_obs(
                info,
                motion_data,
                linvel,
                gyro,
                dof_pos,
                dof_vel,
                robot_body_pos_w,
                robot_body_quat_w,
            )
        finally:
            # 构造完成后清空临时引用，避免下一次 obs 误用旧 reset 信息。
            self._obs_compute_info = None

        # Cache for next-step joint_acc_l2. The reset path overwrites this
        # via the DR provider's ``prev_dof_vel`` info_update.
        # 当前关节速度成为下一步 joint_acc_l2 的 prev_dof_vel。
        info["prev_dof_vel"] = dof_vel.copy()
        return obs

```
### Action：SAC tracking action

| 本任务人工导读 |
| --- |
| 动作沿用 G1 tracking 29 DoF 目标。 |
| SAC 只改变算法/obs 口径，不改变 actuator。 |

```python
# src/unilab/envs/motion_tracking/g1/tracking.py
    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> np.ndarray:
        # WBT SAC 仍输出 G1 29 维关节动作，不改变 actuator contract。
        state.info["last_actions"] = state.info.get("current_actions", np.zeros_like(actions))
        state.info["current_actions"] = actions
        # latency 打开时实际执行上一帧动作，actor obs 中也会看到 current_actions。
        exec_actions = (
            state.info["last_actions"]
            if self._cfg.control_config.simulate_action_latency
            else actions
        )
        bias = state.info.get("default_dof_pos_bias")
        # 若 reset 随机化默认角，控制目标和相对关节观测使用同一偏置。
        base = self.default_angles + bias if bias is not None else self.default_angles
        # 归一化动作乘 action_scale 后叠加默认姿态，得到 position target。
        ctrl: np.ndarray = exec_actions * self._cfg.control_config.action_scale + base
        return ctrl

```
### Reward/Termination

| 本任务人工导读 |
| --- |
| reward 在父类 tracking 基础上追加/调整 WBT 项。 |
| termination 继承 tracking.py 的风险边界，本任务不在 tracking_obs.py 重写 update_state。 |

```python
# src/unilab/envs/motion_tracking/g1/tracking_obs.py
    def _init_reward_functions(self) -> None:
        # 先继承 motion tracking 的 root/body/joint/action/contact reward 表。
        super()._init_reward_functions()
        # 追加 WBT 专用的关节加速度惩罚，约束控制平滑性。
        self._reward_fns["joint_acc_l2"] = self._reward_joint_acc_l2
        # 追加近似 PD torque 惩罚，抑制过大的目标角和速度误差。
        self._reward_fns["joint_torque_l2"] = self._reward_joint_torque_l2

```



## Agent

Agent 输出 G1 29 维动作，算法侧通常使用 offpolicy SAC owner。

训练入口通过 owner YAML 决定注册 env、后端身份和算法参数；CLI 应使用 `--sim` 选择 backend owner，而不是单独 override `training.sim_backend`。

```yaml
# conf/offpolicy/task/sac/g1_wbt_obs/mujoco.yaml
training:
  task_name: G1WBTObs
  sim_backend: mujoco
algo:
  obs_groups: {}
  num_envs: 4096
  max_iterations: 140000
```

## Env

Env 是 tracking SAC 的观测专用子类，强调 obs contract 稳定性。

Env 必须遵守 `NpEnv` contract：`reset()` 返回 `(obs_dict, info_dict)`，`obs` 是 dict，`obs_groups_spec` 决定 learner 使用哪些观测组。

```python
# src/unilab/envs/motion_tracking/g1/tracking_obs.py
    randomize_geom_friction: bool = False
    friction_range: list[float] = field(default_factory=lambda: [0.3, 1.2])
    friction_geom_pattern: str = r"^(left|right)_foot[1-7]_collision$"


@registry.envcfg("G1WBTObs")
@dataclass
class G1WBTObsCfg(G1MotionTrackingSACCfg):
    """SAC whole-body tracking with sim2real obs / DR / reward extensions."""

    noise_config: ObsNoiseConfig = field(default_factory=ObsNoiseConfig)  # type: ignore[assignment]
```

## Obs

观测字段更严格地组织为 WBT 所需输入，critic 仍可包含特权项。

### Obs 字段级拆解

下面的表格把源码里的观测拼接逻辑拆成语义字段。维度如果依赖 history、terrain scan 或 motion body 数量，文档会写成“规模”而不是硬编码，避免和配置漂移。

| 观测字段 | 维度/规模 | 代码来源 | 中文说明 |
| --- | --- | --- | --- |
| current robot state | 随 G1 nu 变化 | `G1MotionTrackingEnv` | 当前 root、关节、末端和 body 状态。 |
| reference root | 位置/朝向/速度 | `MotionLoader` | 参考 motion 中 root 的目标位置、朝向和速度。 |
| reference body | body_names × 状态 | `body_names` | 参考 motion 中关键 body 的位置、朝向、线速度和角速度。 |
| reference joints | nu 维 | `motion_data.joint_pos` | 参考 motion 的关节角。 |
| phase / sampling | clip 时间相关 | `MotionSampler` | 当前 episode 对应 motion clip 的时间位置。 |
| last_actions | nu 维 | info 动作历史 | 帮助策略保持动作连续。 |
| critic extras | 任务相关 | SAC/Deploy 变体 | SAC 变体可给 critic 额外 base linvel 或全身特权信息。 |

owner YAML 中的 `algo.obs_groups` 说明算法从 env obs dict 中读取哪些组。没有写入 owner 的字段保持 env cfg 默认值，文档不臆造未配置项。

```yaml
# conf/offpolicy/task/sac/g1_wbt_obs/mujoco.yaml
algo:
  obs_groups:
    {}

```

代码侧观测通常在 `_get_obs`、`_build_obs`、`obs_groups_spec` 或 base env 中组装：

```text
# 未在 src/unilab/envs/motion_tracking/g1/tracking_obs.py 中找到片段：obs_groups_spec, _get_obs, build_obs, get_obs
```

源码锚点拆解：

| 源码符号 | 中文说明 |
| --- | --- |
| _compute_obs | 读取 backend 传感器和 info 字段，拼接 actor/critic 观测。 |

## Action

动作空间与 G1 tracking 一致。

### Action 逐维拆解

G1 的 action 覆盖 29 个可控关节：腿、腰和双臂都参与控制。行走任务更强调腿部和腰部稳定，motion tracking 则让上肢也跟随参考动作。

当前代表 scene 编译出的 action 维度为 `29`。下表来自 MuJoCo 编译模型的 actuator 顺序，也就是策略输出维度和执行器维度的对应关系。

| Action 维度 | 执行器 | 驱动关节 | 中文含义 | 控制/限制说明 |
| --- | --- | --- | --- | --- |
| 0 | left_hip_pitch_joint | left_hip_pitch_joint | 左髋俯仰关节 | XML 未显式限制，实际由 env 缩放/默认角和 backend 控制 |
| 1 | left_hip_roll_joint | left_hip_roll_joint | 左髋侧摆关节 | XML 未显式限制，实际由 env 缩放/默认角和 backend 控制 |
| 2 | left_hip_yaw_joint | left_hip_yaw_joint | 左髋偏航关节 | XML 未显式限制，实际由 env 缩放/默认角和 backend 控制 |
| 3 | left_knee_joint | left_knee_joint | 左膝关节 | XML 未显式限制，实际由 env 缩放/默认角和 backend 控制 |
| 4 | left_ankle_pitch_joint | left_ankle_pitch_joint | 左踝俯仰关节 | XML 未显式限制，实际由 env 缩放/默认角和 backend 控制 |
| 5 | left_ankle_roll_joint | left_ankle_roll_joint | 左踝侧摆关节 | XML 未显式限制，实际由 env 缩放/默认角和 backend 控制 |
| 6 | right_hip_pitch_joint | right_hip_pitch_joint | 右髋俯仰关节 | XML 未显式限制，实际由 env 缩放/默认角和 backend 控制 |
| 7 | right_hip_roll_joint | right_hip_roll_joint | 右髋侧摆关节 | XML 未显式限制，实际由 env 缩放/默认角和 backend 控制 |
| 8 | right_hip_yaw_joint | right_hip_yaw_joint | 右髋偏航关节 | XML 未显式限制，实际由 env 缩放/默认角和 backend 控制 |
| 9 | right_knee_joint | right_knee_joint | 右膝关节 | XML 未显式限制，实际由 env 缩放/默认角和 backend 控制 |
| 10 | right_ankle_pitch_joint | right_ankle_pitch_joint | 右踝俯仰关节 | XML 未显式限制，实际由 env 缩放/默认角和 backend 控制 |
| 11 | right_ankle_roll_joint | right_ankle_roll_joint | 右踝侧摆关节 | XML 未显式限制，实际由 env 缩放/默认角和 backend 控制 |
| 12 | waist_yaw_joint | waist_yaw_joint | 腰偏航关节 | XML 未显式限制，实际由 env 缩放/默认角和 backend 控制 |
| 13 | waist_roll_joint | waist_roll_joint | 腰侧摆关节 | XML 未显式限制，实际由 env 缩放/默认角和 backend 控制 |
| 14 | waist_pitch_joint | waist_pitch_joint | 腰俯仰关节 | XML 未显式限制，实际由 env 缩放/默认角和 backend 控制 |
| 15 | left_shoulder_pitch_joint | left_shoulder_pitch_joint | 左肩俯仰关节 | XML 未显式限制，实际由 env 缩放/默认角和 backend 控制 |
| 16 | left_shoulder_roll_joint | left_shoulder_roll_joint | 左肩侧摆关节 | XML 未显式限制，实际由 env 缩放/默认角和 backend 控制 |
| 17 | left_shoulder_yaw_joint | left_shoulder_yaw_joint | 左肩偏航关节 | XML 未显式限制，实际由 env 缩放/默认角和 backend 控制 |
| 18 | left_elbow_joint | left_elbow_joint | 左肘关节 | XML 未显式限制，实际由 env 缩放/默认角和 backend 控制 |
| 19 | left_wrist_roll_joint | left_wrist_roll_joint | 左腕侧摆关节 | XML 未显式限制，实际由 env 缩放/默认角和 backend 控制 |
| 20 | left_wrist_pitch_joint | left_wrist_pitch_joint | 左腕俯仰关节 | XML 未显式限制，实际由 env 缩放/默认角和 backend 控制 |
| 21 | left_wrist_yaw_joint | left_wrist_yaw_joint | 左腕偏航关节 | XML 未显式限制，实际由 env 缩放/默认角和 backend 控制 |
| 22 | right_shoulder_pitch_joint | right_shoulder_pitch_joint | 右肩俯仰关节 | XML 未显式限制，实际由 env 缩放/默认角和 backend 控制 |
| 23 | right_shoulder_roll_joint | right_shoulder_roll_joint | 右肩侧摆关节 | XML 未显式限制，实际由 env 缩放/默认角和 backend 控制 |
| 24 | right_shoulder_yaw_joint | right_shoulder_yaw_joint | 右肩偏航关节 | XML 未显式限制，实际由 env 缩放/默认角和 backend 控制 |
| 25 | right_elbow_joint | right_elbow_joint | 右肘关节 | XML 未显式限制，实际由 env 缩放/默认角和 backend 控制 |
| 26 | right_wrist_roll_joint | right_wrist_roll_joint | 右腕侧摆关节 | XML 未显式限制，实际由 env 缩放/默认角和 backend 控制 |
| 27 | right_wrist_pitch_joint | right_wrist_pitch_joint | 右腕俯仰关节 | XML 未显式限制，实际由 env 缩放/默认角和 backend 控制 |
| 28 | right_wrist_yaw_joint | right_wrist_yaw_joint | 右腕偏航关节 | XML 未显式限制，实际由 env 缩放/默认角和 backend 控制 |

控制参数来自 owner YAML 的 `env.control_config`，缺省值由对应 env cfg/dataclass 提供。

| 配置字段 | 当前值 | 中文说明 |
| --- | --- | --- |
| action_scale | 2.0 | 策略输出到目标关节角的缩放系数。 |
| simulate_action_latency | True | 布尔开关。 |

```yaml
# conf/offpolicy/task/sac/g1_wbt_obs/mujoco.yaml
env:
  control_config:
    action_scale: 2.0
    simulate_action_latency: true

```

动作到执行器的实际映射由 backend 的 actuator control range 和 env 的 `apply_action`/控制函数完成：

```python
# src/unilab/envs/motion_tracking/g1/tracking_obs.py
        dof_pos = info["dof_pos"]
        dof_vel = info["dof_vel"]
        last_actions = info.get("last_actions")
        if last_actions is None:
            return np.zeros((self._num_envs,), dtype=get_global_dtype())
        target_q = (
            last_actions * self._cfg.control_config.action_scale + self._effective_default_angles()
        )
        torque = self._base_kp * (target_q - dof_pos) - self._base_kd * dof_vel
        return np.asarray(np.sum(np.square(torque), axis=1), dtype=get_global_dtype())

    # ------------------------------------------------------------------ #
    # Obs
```

源码锚点拆解：

| 源码符号 | 中文说明 |
| --- | --- |

## Reward

reward 沿用 motion tracking SAC 配置。

### Reward 逐项拆解

reward scale 以 owner YAML 的 `reward.scales` 为真源；源码中的 `RewardConfig` 描述可用字段，训练脚本只做 Hydra 组装和注入。正数通常表示奖励，负数通常表示惩罚，0 表示该项在当前 owner 中关闭或仅保留接口。

| Reward key | Scale | 方向 | 中文含义 |
| --- | --- | --- | --- |
| motion_global_root_pos | 1.0 | 奖励 | 参考 motion 的 root 全局位置跟踪奖励。 |
| motion_global_root_ori | 1.0 | 奖励 | 参考 motion 的 root 全局朝向跟踪奖励。 |
| motion_body_pos | 1.0 | 奖励 | 参考 motion 的各 body 位置跟踪奖励。 |
| motion_body_ori | 1.0 | 奖励 | 参考 motion 的各 body 朝向跟踪奖励。 |
| motion_body_lin_vel | 1.0 | 奖励 | 参考 motion 的 body 线速度跟踪奖励。 |
| motion_body_ang_vel | 1.0 | 奖励 | 参考 motion 的 body 角速度跟踪奖励。 |
| motion_joint_pos | 0.0 | 记录/关闭 | 参考 motion 的关节位置跟踪奖励。 |
| motion_joint_vel | 0.0 | 记录/关闭 | 参考 motion 的关节速度跟踪奖励。 |
| action_rate_l2 | -0.1 | 惩罚 | 动作变化 L2 惩罚， motion tracking 中用于约束动作平滑。 |
| joint_limit | -5.0 | 惩罚 | 关节限位惩罚，避免追踪动作时撞到机械限位。 |
| undesired_contacts | -0.1 | 惩罚 | 非期望接触惩罚，例如手、膝、身体等部位异常触地。 |
| joint_acc_l2 | -2.5e-07 | 惩罚 | 仓库 owner YAML 中声明的奖励项；具体实现见本页源码片段和 env 的 reward dispatch。 |
| joint_torque_l2 | -1e-5 | 配置项 | 手部力矩惩罚，降低过大的执行器输出。 |

```yaml
# conf/offpolicy/task/sac/g1_wbt_obs/mujoco.yaml
reward:
  scales:
    motion_global_root_pos: 1.0
    motion_global_root_ori: 1.0
    motion_body_pos: 1.0
    motion_body_ori: 1.0
    motion_body_lin_vel: 1.0
    motion_body_ang_vel: 1.0
    motion_joint_pos: 0.0
    motion_joint_vel: 0.0
    action_rate_l2: -0.1
    joint_limit: -5.0
    undesired_contacts: -0.1
    joint_acc_l2: -2.5e-07
    joint_torque_l2: -1e-5

```

源码中的 reward 入口：

```python
# src/unilab/envs/motion_tracking/g1/tracking_obs.py
            self._dr_manager = DomainRandomizationManager(self, extended_provider)

    # ------------------------------------------------------------------ #
    # Rewards
    # ------------------------------------------------------------------ #

    def _init_reward_functions(self) -> None:
        super()._init_reward_functions()
        self._reward_fns["joint_acc_l2"] = self._reward_joint_acc_l2
        self._reward_fns["joint_torque_l2"] = self._reward_joint_torque_l2

    def _reward_joint_acc_l2(self, info: dict) -> np.ndarray:
        dof_vel = info["dof_vel"]
```

源码锚点拆解：

| 源码符号 | 中文说明 |
| --- | --- |
| _init_reward_functions | 把 reward 名称映射到源码函数，owner YAML 的 scale 会按这些 key 分发。 |
| _reward_joint_acc_l2 | 单个 reward 项的实现函数，通常返回每个 env 的 reward 向量。 |
| _reward_joint_torque_l2 | 单个 reward 项的实现函数，通常返回每个 env 的 reward 向量。 |

## 初始状态

reset 使用 motion reference 和同类随机化。

机器人默认姿态来自 scene/task fragment 中的 keyframe；任务 reset 在此基础上加入命令、motion reference、物体状态或随机化。

```xml
<!-- src/unilab/assets/robots/g1/scene_flat.xml -->
    <contact name="right_foot_contact_3" geom1="floor" geom2="right_foot_contact_3_geom" data="found" num="1" reduce="mindist"/>
  </sensor>

  <keyframe>
    <!-- <key name="home"
      qpos="
      0 0 0.79
```

```python
# src/unilab/envs/motion_tracking/g1/tracking_obs.py
from unilab.dr import (
    DomainRandomizationCapabilities,
    ResetPlan,
    ResetRandomizationPayload,
)
from unilab.dr.dr_utils import (
    build_common_reset_randomization,
    zero_actions,
)
from unilab.dr.types import RESET_TERM_GEOM_FRICTION
from unilab.dtype_config import get_global_dtype
from unilab.envs.locomotion.g1.base import NoiseConfig

```

## 终止条件

终止沿用 tracking 阈值。

终止条件通常由 env 源码中的 `_compute_termination(s)`、reward config 阈值和 `max_episode_seconds` 共同决定。

```yaml
# conf/offpolicy/task/sac/g1_wbt_obs/mujoco.yaml
env:
  anchor_pos_z_threshold: 0.4
  ee_body_pos_z_threshold: 0.5
  truncate_on_clip_end: true

reward:
  {}

```

```python
# src/unilab/envs/motion_tracking/g1/tracking_obs.py
        motion_data = env.motion_loader.get_motion_at_frame(motion_frames)
        qpos, qvel = _build_motion_reference_state(env, env_ids, motion_data)

        info_updates: dict[str, Any] = {
            "current_actions": zero_actions(num_reset, env._num_action),
            "last_actions": zero_actions(num_reset, env._num_action),
            # Seed prev_dof_vel with the post-reset joint velocity so the first
            # joint_acc_l2 sample is physically meaningful (Δv from the new
            # starting velocity, not a spurious step from pre-termination).
            "prev_dof_vel": qvel[:, 6:].astype(get_global_dtype()),
        }

        dr_cfg = env.cfg.domain_rand
        if getattr(dr_cfg, "enable_encoder_bias", False):
            low, high = dr_cfg.encoder_bias_range
            info_updates["joint_pos_obs_bias"] = np.random.uniform(
                low, high, size=(num_reset, env._num_action)
```

## 域随机化

domain_rand 由 SAC owner YAML 控制。

域随机化只在 reset/interval 等冷路径触发，不在 step 热路径解析资产或探测 backend 私有能力。

### 域随机化字段拆解

| 配置字段 | 当前值 | 中文说明 |
| --- | --- | --- |
| added_mass_range | [-1.0, 1.0] | 随机采样范围或控制范围。 |
| com_offset_x | [-0.05, 0.05] | 任务 owner YAML 中的配置字段。 |
| com_offset_y | [-0.05, 0.05] | 任务 owner YAML 中的配置字段。 |
| com_offset_z | [-0.05, 0.05] | 任务 owner YAML 中的配置字段。 |
| enable_encoder_bias | True | 布尔开关。 |
| encoder_bias_range | [-0.01, 0.01] | 随机采样范围或控制范围。 |
| friction_geom_pattern | ^(left\|right)_foot[1-7]_collision$ | 任务 owner YAML 中的配置字段。 |
| friction_range | [0.3, 1.2] | 随机采样范围或控制范围。 |
| gravity_range | [[0.0, 0.0, -9.81], [0.0, 0.0, -9.81]] | 随机采样范围或控制范围。 |
| kd_multiplier_range | [0.85, 1.15] | 随机采样范围或控制范围。 |
| kp_multiplier_range | [0.9, 1.1] | 随机采样范围或控制范围。 |
| max_force | [300.0, 300.0, 120.0] | 任务 owner YAML 中的配置字段。 |
| push_body_name |  | 任务 owner YAML 中的配置字段。 |
| push_interval | 200 | 任务 owner YAML 中的配置字段。 |
| push_robots | True | 布尔开关。 |
| random_com | True | 布尔开关。 |
| randomize_base_mass | True | 域随机化开关。 |
| randomize_com_y | True | 域随机化开关。 |
| randomize_com_z | True | 域随机化开关。 |
| randomize_geom_friction | True | 域随机化开关。 |
| randomize_gravity | False | 域随机化开关。 |
| randomize_kd | True | 域随机化开关。 |
| randomize_kp | True | 域随机化开关。 |

```yaml
# conf/offpolicy/task/sac/g1_wbt_obs/mujoco.yaml
env:
  domain_rand:
    randomize_base_mass: true
    added_mass_range:
    - -1.0
    - 1.0
    random_com: true
    com_offset_x:
    - -0.05
    - 0.05
    randomize_com_y: true
    com_offset_y:
    - -0.05
    - 0.05
    randomize_com_z: true
    com_offset_z:
    - -0.05
    - 0.05
    randomize_gravity: false
    gravity_range:
    - - 0.0
      - 0.0
      - -9.81
    - - 0.0
      - 0.0
      - -9.81
    push_robots: true
    push_interval: 200
    max_force:
    - 300.0
    - 300.0
    - 120.0
    push_body_name: null
    randomize_kp: true
    kp_multiplier_range:
    - 0.9
    - 1.1
    randomize_kd: true
    kd_multiplier_range:
    - 0.85
    - 1.15
    randomize_geom_friction: true
    friction_range:
    - 0.3
    - 1.2
    friction_geom_pattern: ^(left|right)_foot[1-7]_collision$
    enable_encoder_bias: true
    encoder_bias_range:
    - -0.01
    - 0.01

```

```python
# src/unilab/envs/motion_tracking/g1/tracking_obs.py
from typing import Any

import numpy as np

from unilab.base import registry
from unilab.dr import (
    DomainRandomizationCapabilities,
    ResetPlan,
    ResetRandomizationPayload,
)
from unilab.dr.dr_utils import (
    build_common_reset_randomization,
    zero_actions,
```

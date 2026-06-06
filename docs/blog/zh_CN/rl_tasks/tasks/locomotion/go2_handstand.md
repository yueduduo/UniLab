---
title: "Go2 HandStand"
slug: "go2_handstand"
category: "locomotion"
robot: "go2"
registered_envs: [Go2HandStand]
---

# Go2 HandStand

## 任务目标

训练 Go2 从常规四足姿态进入并维持倒立/手倒立式站立，使身体高度、姿态和接触模式满足 handstand 目标。

![Go2 HandStand 场景渲染](../../images/tasks/go2_handstand.png)

> 图片由 `_tools/render_static_images.py` 使用 `src/unilab/assets/robots/go2/scene_flat.xml` 的 MuJoCo scene 离屏渲染生成；如果当前环境无法启用 EGL/OSMesa，生成 manifest 会记录失败原因。

## 证据索引

| 项目 | 内容 |
| --- | --- |
| 任务类别 | 运动控制 |
| 机器人 | go2 |
| 代表 scene | src/unilab/assets/robots/go2/scene_flat.xml |
| 代表源码 | src/unilab/envs/locomotion/go2/handstand.py |
| 代表 owner YAML | conf/ppo/task/go2_handstand/mujoco.yaml |
| 动作维度 | 12 |

### Registry 与 Owner

| 注册名 | 后端 | 配置类 | 配置源码 |
| --- | --- | --- | --- |
| Go2HandStand | motrix, mujoco | Go2HandStandCfg | src/unilab/envs/locomotion/go2/handstand.py |

| 算法组 | 后端 | training.task_name | owner YAML |
| --- | --- | --- | --- |
| ppo | motrix | Go2HandStand | conf/ppo/task/go2_handstand/motrix.yaml |
| ppo | mujoco | Go2HandStand | conf/ppo/task/go2_handstand/mujoco.yaml |

## 代码级执行流

这部分不再用通用关键词套模板，而是为 `go2_handstand` 单独写源码导读。源码片段只负责提供证据；每节说明都围绕本任务自己的配置、状态、观测、动作和 reward 语义展开。

| 阶段 | 证据位置 | 代码级说明 |
| --- | --- | --- |
| 1. Hydra owner | conf/ppo/task/go2_handstand/mujoco.yaml | 训练命令先 compose owner YAML。这里确定 `training.task_name`、后端身份、obs_groups、reward scale 和 env override。 |
| 2. Registry 构造 Env | src/unilab/envs/locomotion/go2/handstand.py | `@registry.envcfg` 注册配置类，`@registry.env` 注册后端实现类；训练脚本只调用 registry，不直接 new 任务类。 |
| 3. Reset/初始状态 | src/unilab/assets/robots/go2/scene_flat.xml | env 从 scene keyframe 或 motion/grasp cache 取初态，再通过 reset randomization 改写 qpos/qvel/info。 |
| 4. Obs dict | src/unilab/envs/locomotion/go2/handstand.py | env 读取 backend 传感器和 info，返回 dict；learner 根据 `obs_groups_spec` 和 owner `algo.obs_groups` 选择 actor/critic 输入。 |
| 5. Action 到控制目标 | src/unilab/envs/locomotion/go2/handstand.py | policy 输出先按 action scale / 控制顺序映射到 actuator，再由 backend step 推进仿真。 |
| 6. Reward dispatch | src/unilab/envs/locomotion/go2/handstand.py | 源码把 reward key 映射到函数，owner YAML 的 scale 决定每项奖励/惩罚是否启用以及权重。 |
| 7. Termination | src/unilab/envs/locomotion/go2/handstand.py | env 根据高度、姿态、接触、参考轨迹误差、物体状态或能量阈值生成 done/terminated。 |

### 本任务阅读重点

| 阅读重点 |
| --- |
| Handstand 不是 joystick 速度跟踪，它把 Go2 的目标从行走切到倒立/非典型支撑姿态。 |
| obs 弱化命令，重点读 gravity、gyro、关节误差和动作历史；reward 则围绕高度、朝向和接触模式。 |
| 源码要看 `Go2HandStandCfg` 与 `Go2HandStandTask`，不要把它和 Go2WalkTask 的 gait phase 混在一起解释。 |

### 本任务注册入口

| 注册名 | 配置类 | 可用后端 |
| --- | --- | --- |
| Go2HandStand | Go2HandStandCfg | motrix, mujoco |

### 本任务源码锚点

| 小节 | 源码文件 | 明确锚点 |
| --- | --- | --- |
| 配置：倒立任务 | src/unilab/envs/locomotion/go2/handstand.py | Go2HandStandCfg, Go2HandStand |
| Obs：姿态稳定 | src/unilab/envs/locomotion/go2/handstand.py | obs_groups_spec, _compute_obs |
| Action：12 维姿态调节 | src/unilab/envs/locomotion/common/base.py | apply_action, default_angles |
| Reward：高度/朝向/接触 | src/unilab/envs/locomotion/go2/handstand.py | _init_reward_functions, height |
| Termination | src/unilab/envs/locomotion/go2/handstand.py | update_state, terminated |

## 关键源码逐段解释（按本任务手写）

### 配置：倒立任务

| 本任务人工导读 |
| --- |
| 配置类选择 handstand 任务的 scene/control/reward 参数。 |
| 任务目标是非典型姿态稳定，不是 x/y/yaw 命令跟踪。 |

```python
# src/unilab/envs/locomotion/go2/handstand.py
# 以 Go2HandStand 注册配置；训练入口通过 owner YAML 选择这个非行走任务。
@registry.envcfg("Go2HandStand")
@dataclass
class Go2HandStandCfg(Go2BaseCfg):
    # handstand 仍使用 Go2 flat scene，任务差异在 env/reward/termination 中表达。
    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(
            model_file=str(ASSETS_ROOT_PATH / "robots" / "go2" / "scene_flat.xml")
        )
    )
    # 每个 episode 最长 20 秒，用于学习从常规姿态进入并维持倒立。
    max_episode_seconds: float = 20.0
    # 初始状态保持 Go2 默认站姿；倒立目标由策略动作和 reward 诱导。
    init_state: InitState = field(default_factory=InitState)
    # 保留 Commands 字段以复用配置结构，但 handstand obs/reward 不把速度命令作为主目标。
    commands: Commands = field(default_factory=Commands)
    # reward_config 由 owner YAML 注入，包含 height/contact/orientation 等 handstand scale。
    reward_config: RewardConfig | None = None
    # 传感器扩展了全局位置、终止接触和惩罚接触，用来判断倒立是否失败。
    sensor: JoystickSensor = field(default_factory=JoystickSensor)  # type: ignore[assignment]
    # 使用 Go2 的 kp/kd 随机化，提升非典型姿态下的控制鲁棒性。
    domain_rand: Go2DomainRandConfig = field(default_factory=Go2DomainRandConfig)


```
### Obs：姿态稳定

| 本任务人工导读 |
| --- |
| 42 维 obs 关注 gyro、gravity、关节差值、关节速度和动作历史。 |
| 没有 joystick command 和 gait phase，说明策略输入围绕平衡控制组织。 |

```python
# src/unilab/envs/locomotion/go2/handstand.py
    @property
    def obs_groups_spec(self) -> dict[str, int]:
        # gyro(3) + gravity(3) + diff(12) + dof_vel(12) + action(12)  = 42
        # actor 不包含 command/phase；critic 额外拿 linvel(3) 和 torso height(1)。
        return {"obs": 42, "critic": 46}

```

```python
# src/unilab/envs/locomotion/go2/handstand.py
    def _compute_obs(
        self,
        info: dict,
        linvel,
        gyro,
        gravity,
        dof_pos,
        dof_vel,
        height,  # , feet_phase
    ) -> dict[str, np.ndarray]:
        # handstand 观测噪声仍沿用 Go2 cfg；owner 可通过 env.noise_config 覆盖。
        noise_cfg = self._cfg.noise_config
        # 关节误差以 home/default_angles 为参考，表示离默认站姿的偏移。
        diff = dof_pos - self.default_angles
        # actor 输入保留 IMU、重力投影、关节角/速度和动作历史。
        gyro = self._obs_noise(gyro, noise_cfg.scale_gyro)
        gravity = self._obs_noise(gravity, noise_cfg.scale_gravity)
        diff = self._obs_noise(diff, noise_cfg.scale_joint_angle)
        dof_vel = self._obs_noise(dof_vel, noise_cfg.scale_joint_vel)
        # linvel 不进入 actor，只作为 critic 特权信息帮助判断身体漂移。
        linvel = self._obs_noise(linvel, noise_cfg.scale_linvel)
        # command = info["commands"]
        # handstand 不使用速度命令；动作历史让策略知道上一帧姿态调节量。
        last_actions = info.get("current_actions", np.zeros_like(diff))
        # actor obs 聚焦姿态稳定，没有 joystick command，也没有 gait phase。
        obs = np.concatenate(
            [gyro, -gravity, diff, dof_vel, last_actions],
            axis=1,
            dtype=get_global_dtype(),
        )
        # critic 追加局部线速度和 torso 高度，帮助 value 评估是否站到目标高度。
        critic = np.concatenate([obs, linvel, height], axis=1, dtype=get_global_dtype())
        return {"obs": obs, "critic": critic}

    # state = jp.hstack([
    #     noisy_linvel,
    #     noisy_gyro,
    #     noisy_gravity,
    #     noisy_joint_angles - self._default_pose,
    #     noisy_joint_vel,
    #     info["last_act"],
    # ])
    # privileged_state = jp.hstack([ TODO
    #     state,
    #     gyro,
    #     accelerometer,
    #     linvel,
    #     angvel,
    #     joint_angles,
    #     joint_vel,
    #     data.actuator_force,
    #     torso_height,
    # ])

```
### Action：12 维姿态调节

| 本任务人工导读 |
| --- |
| Go2 handstand 复用 locomotion common base 的动作映射。 |
| 动作仍控制 Go2 12 个腿部 actuator，策略通过相对默认角偏移学习维持 handstand 姿态。 |

```python
# src/unilab/envs/locomotion/common/base.py
    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> np.ndarray:
        # 记录动作历史；handstand 的 action_rate 惩罚和下一帧 obs 都依赖它。
        state.info["last_actions"] = state.info.get("current_actions", np.zeros_like(actions))
        state.info["current_actions"] = actions
        # 如果开启延迟，用上一帧动作模拟执行器滞后。
        exec_actions = (
            state.info["last_actions"]
            if self._cfg.control_config.simulate_action_latency
            else actions
        )
        # 12 维策略输出转成相对默认角的目标角偏移，用于调节倒立姿态。
        ctrl: np.ndarray = (
            exec_actions * self._cfg.control_config.action_scale + self.default_angles
        )
        # 返回 Go2 12 个腿部 actuator 的位置控制目标。
        return ctrl

```
### Reward：高度/朝向/接触

| 本任务人工导读 |
| --- |
| reward 读取 handstand 专用项，不再以速度跟踪作为主目标。 |
| 接触和姿态项决定是否真正站成目标支撑构型。 |

```python
# src/unilab/envs/locomotion/go2/handstand.py
    def _init_reward_functions(self):
        # handstand reward 不以速度跟踪为核心，而是高度、朝向、接触和目标腿姿态。
        self._reward_fns: dict[str, Any] = {
            # torso 高度靠近 _z_des 时给奖励，是进入倒立高度的主信号。
            "height": self._reward_height,
            # contact 惩罚前足/指定支撑接触，防止错误部位落地。
            "contact": self._cost_contact,
            # 注意 owner YAML 使用同样拼写的 oritentation；这里映射到朝向奖励实现。
            "oritentation": self._reward_orientation,
            # pose 约束非目标关节不要偏离默认姿态太远。
            "pose": self._cost_pose,
            # penalty_contact 惩罚后腿/非期望身体部位接触。
            "penalty_contact": self._reward_penalty_contact,
            # 平滑动作，降低倒立过程中关节目标抖动。
            "action_rate": rewards.action_rate,
            # tar 奖励目标腿关节接近 handstand target_angle。
            "tar": self._reward_tar,
            # 后腿离地时间奖励用于形成倒立后的后腿步进/摆动。
            "feet_air_time": self._reward_feet_air_time,
            # 抑制世界 z 方向速度，减少站起后上下弹跳。
            "world_z_vel_penalty": self._reward_world_z_vel_penalty,
        }

```

```python
# src/unilab/envs/locomotion/go2/handstand.py
@dataclass
class RewardConfig:
    # owner YAML 的 scales 是 reward dispatch 的权重真源。
    scales: dict[str, float]
    # 保留通用 tracking_sigma 字段以复用 RewardContext，handstand 主 reward 不依赖速度跟踪。
    tracking_sigma: float
    # base_height_target 是通用字段；handstand 实际高度目标在任务中用 _z_des。
    base_height_target: float
    # 足端/膝部目标高度供 feet_air_time、拖地或支撑相关 reward 使用。
    target_foot_height: float = 0.1
    knee_height_target: float = 0.08


```
### Termination

| 本任务人工导读 |
| --- |
| 终止由高度、倾斜、接触失败和 horizon 组合。 |
| handstand 的失败定义和普通四足行走不同。 |

```python
# src/unilab/envs/locomotion/go2/handstand.py
    def update_state(self, state: NpEnvState) -> NpEnvState:
        # 读取本帧速度、IMU、重力和关节状态。
        linvel = self.get_local_linvel()
        gyro = self.get_gyro()
        gravity = self._backend.get_sensor_data("upvector")
        dof_pos = self.get_dof_pos()
        dof_vel = self.get_dof_vel()
        # 刷新四足接触力和足端位置，reward 与终止都依赖这些传感器。
        self.feet_force[:, :, :] = 0
        for i in range(len(self._cfg.sensor.feet_force)):
            self.feet_force[:, i, :] = self._backend.get_sensor_data(self._cfg.sensor.feet_force[i])
        for i in range(len(self._cfg.sensor.feet_pos)):
            self.feet_pos[:, i, :] = self._backend.get_sensor_data(self._cfg.sensor.feet_pos[i])
        # torso_height 来自 global_position 的 z，用于 height reward 和后腿步进门控。
        self.torso_height = self._backend.get_sensor_data(self._cfg.sensor.global_pos)[:, -1]
        # ternamate_contact 收集会直接判定失败的前部/机身接触传感器。
        contact_arrays = []
        for name in self._cfg.sensor.ternamate_contact:
            arr = self._backend.get_sensor_data(name)
            contact_arrays.append(arr)
        result = np.concatenate(contact_arrays, axis=1)
        # print(linvel)
        # Update phase for stepping pattern (only when high enough)
        # 只有达到目标高度 80% 后，才允许后腿相位推进。
        height_mask = self.torso_height >= self._z_des * 0.8
        dt = self._cfg.ctrl_dt
        phase_increment = self.gait_frequency * dt
        self.feet_phase = (self.feet_phase + phase_increment) % 1.0
        # Only allow stepping for rear legs (2=RL, 3=RR) when height is sufficient
        # 前腿相位锁定为支撑，后腿才参与 air-time/stepping 奖励。
        self.feet_phase[:, :2] = 0.0  # Front legs phase locked at 0 (should be in stance)
        # Reset rear leg phase when height is too low
        # 高度不足时重置后腿相位，避免未站起阶段获得步进奖励。
        for i in [2, 3]:
            self.feet_phase[~height_mask, i] = 0.0

        # Update feet air time for stepping reward (only rear legs)
        # 只统计 RL/RR 的接触与离地时间，用于后腿摆动 reward。
        contact = self.feet_force[:, [2, 3], 0] > 1.0
        contact_filt = np.logical_or(contact, self._last_contacts)
        self._last_contacts = contact
        # Increment air time
        self._feet_air_time[:, [2, 3]] += self._cfg.ctrl_dt
        # Reset air time for feet in contact
        self._feet_air_time[:, [2, 3]] *= ~contact_filt

        # upvector z 过低表示机身朝向已经越过失败阈值。
        terminated_z = gravity[:, 2] <= -0.25
        # 任何终止接触传感器触发都视为 handstand 失败。
        terminated_contact = np.any(result, axis=1)
        # After 100 steps, terminate if height is too low (failed to maintain target)
        # step_count = state.info.get("steps", np.zeros((self._num_envs,), dtype=np.uint32))
        # terminated_height = (step_count >= 100) & (self.torso_height < self._z_des * 0.8)
        # terminated = np.logical_or(
        #     np.logical_or(terminated_contact, terminated_z),
        #     terminated_height
        # )
        # 当前启用的终止只包含姿态翻倒和终止接触；高度失败逻辑仍保留为注释证据。
        terminated = np.logical_or(terminated_contact, terminated_z)
        # reward 使用 handstand 专用 dispatch，obs 追加 torso_height 给 critic。
        reward = self._compute_reward(state.info, linvel, gyro, dof_pos)
        obs = self._compute_obs(
            state.info,
            linvel,
            gyro,
            gravity,
            dof_pos,
            dof_vel,
            self.torso_height.reshape(-1, 1),  # , self.feet_phase[:,[2,3]]
        )
        return state.replace(obs=obs, reward=reward, terminated=terminated)

```



## Agent

Agent 输出 Go2 12 维关节目标，策略重点学习非典型支撑相下的姿态调节。

训练入口通过 owner YAML 决定注册 env、后端身份和算法参数；CLI 应使用 `--sim` 选择 backend owner，而不是单独 override `training.sim_backend`。

```yaml
# conf/ppo/task/go2_handstand/mujoco.yaml
training:
  task_name: Go2HandStand
  sim_backend: mujoco
algo:
  obs_groups:
    actor:
    - actor
    critic:
    - critic
  num_envs: 1024
  max_iterations: 3000
```

## Env

Env 复用 Go2 locomotion backend，但任务类提供 handstand 专用 obs/reward/termination。

Env 必须遵守 `NpEnv` contract：`reset()` 返回 `(obs_dict, info_dict)`，`obs` 是 dict，`obs_groups_spec` 决定 learner 使用哪些观测组。

```python
# src/unilab/envs/locomotion/go2/handstand.py
        "RR_calf_contact1",
        "RR_calf_contact2",
    ]


@registry.envcfg("Go2HandStand")
@dataclass
class Go2HandStandCfg(Go2BaseCfg):
    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(
            model_file=str(ASSETS_ROOT_PATH / "robots" / "go2" / "scene_flat.xml")
```

## Obs

观测更强调姿态、重力投影、关节状态和动作历史，弱化 joystick 命令。

### Obs 字段级拆解

下面的表格把源码里的观测拼接逻辑拆成语义字段。维度如果依赖 history、terrain scan 或 motion body 数量，文档会写成“规模”而不是硬编码，避免和配置漂移。

| 观测字段 | 维度/规模 | 代码来源 | 中文说明 |
| --- | --- | --- | --- |
| command | 通常 3 维 | `commands` / joystick 采样 | 期望 x/y 线速度和 yaw 角速度，是策略要跟踪的外部目标。 |
| gyro | 3 维 | `get_gyro()` / sensor.gyro | 机身角速度，帮助策略判断身体旋转和姿态变化。 |
| gravity | 3 维 | 局部重力投影 | 把世界重力投到 body 坐标系，用于表示 roll/pitch 倾斜。 |
| dof_pos - default | nu 维 | 关节角与 keyframe 默认角差 | 告诉策略每个关节离默认站姿有多远。 |
| dof_vel | nu 维 | backend dof velocity | 关节速度，用于阻尼和判断腿部摆动速度。 |
| last_actions | nu 维 | info['last_actions'] | 上一控制步动作，帮助策略形成平滑控制。 |
| critic privileged | 任务相关 | `critic` obs group | 训练 critic 可读取更多速度或地形信息，actor 不一定可见。 |

owner YAML 中的 `algo.obs_groups` 说明算法从 env obs dict 中读取哪些组。没有写入 owner 的字段保持 env cfg 默认值，文档不臆造未配置项。

```yaml
# conf/ppo/task/go2_handstand/mujoco.yaml
algo:
  obs_groups:
    actor:
    - actor
    critic:
    - critic

```

代码侧观测通常在 `_get_obs`、`_build_obs`、`obs_groups_spec` 或 base env 中组装：

```python
# src/unilab/envs/locomotion/go2/handstand.py
        self.target_angle = np.array([0, 1.82, -1.16, 0.0, 1.82, -1.16])

    def _init_task_domain_randomization(self) -> None:
        self._init_domain_randomization(Go2HandStandDomainRandomizationProvider())

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        # gyro(3) + gravity(3) + diff(12) + dof_vel(12) + action(12)  = 42
        return {"obs": 42, "critic": 46}

    def _init_reward_functions(self):
        self._reward_fns: dict[str, Any] = {
            "height": self._reward_height,
```

源码锚点拆解：

| 源码符号 | 中文说明 |
| --- | --- |
| obs_groups_spec | 声明 obs dict 中各观测组的维度，是 wrapper/learner 对齐观测的契约。 |
| _compute_obs | 读取 backend 传感器和 info 字段，拼接 actor/critic 观测。 |

## Action

动作仍是 12 个腿部 actuator 的位置目标。

### Action 逐维拆解

四足任务的 action 逐维对应四条腿的 hip/thigh/calf actuator。env 在 `apply_action` 中把策略输出乘以 `action_scale`，再加到 keyframe 默认角上；因此策略学习的是“相对默认站姿的目标角偏移”，不是直接力矩。

当前代表 scene 编译出的 action 维度为 `12`。下表来自 MuJoCo 编译模型的 actuator 顺序，也就是策略输出维度和执行器维度的对应关系。

| Action 维度 | 执行器 | 驱动关节 | 中文含义 | 控制/限制说明 |
| --- | --- | --- | --- | --- |
| 0 | FR_hip | FR_hip_joint | 右前腿髋外展/内收关节 | XML 未显式限制，实际由 env 缩放/默认角和 backend 控制 |
| 1 | FR_thigh | FR_thigh_joint | 右前腿髋俯仰（大腿）关节 | XML 未显式限制，实际由 env 缩放/默认角和 backend 控制 |
| 2 | FR_calf | FR_calf_joint | 右前腿膝关节（小腿摆动） | XML 未显式限制，实际由 env 缩放/默认角和 backend 控制 |
| 3 | FL_hip | FL_hip_joint | 左前腿髋外展/内收关节 | XML 未显式限制，实际由 env 缩放/默认角和 backend 控制 |
| 4 | FL_thigh | FL_thigh_joint | 左前腿髋俯仰（大腿）关节 | XML 未显式限制，实际由 env 缩放/默认角和 backend 控制 |
| 5 | FL_calf | FL_calf_joint | 左前腿膝关节（小腿摆动） | XML 未显式限制，实际由 env 缩放/默认角和 backend 控制 |
| 6 | RR_hip | RR_hip_joint | 右后腿髋外展/内收关节 | XML 未显式限制，实际由 env 缩放/默认角和 backend 控制 |
| 7 | RR_thigh | RR_thigh_joint | 右后腿髋俯仰（大腿）关节 | XML 未显式限制，实际由 env 缩放/默认角和 backend 控制 |
| 8 | RR_calf | RR_calf_joint | 右后腿膝关节（小腿摆动） | XML 未显式限制，实际由 env 缩放/默认角和 backend 控制 |
| 9 | RL_hip | RL_hip_joint | 左后腿髋外展/内收关节 | XML 未显式限制，实际由 env 缩放/默认角和 backend 控制 |
| 10 | RL_thigh | RL_thigh_joint | 左后腿髋俯仰（大腿）关节 | XML 未显式限制，实际由 env 缩放/默认角和 backend 控制 |
| 11 | RL_calf | RL_calf_joint | 左后腿膝关节（小腿摆动） | XML 未显式限制，实际由 env 缩放/默认角和 backend 控制 |

控制参数来自 owner YAML 的 `env.control_config`，缺省值由对应 env cfg/dataclass 提供。

| 配置字段 | 当前值 | 中文说明 |
| --- | --- | --- |

```yaml
# conf/ppo/task/go2_handstand/mujoco.yaml
env:
  control_config:
    {}

```

动作到执行器的实际映射由 backend 的 actuator control range 和 env 的 `apply_action`/控制函数完成：

```text
# 未在 src/unilab/envs/locomotion/go2/handstand.py 中找到片段：apply_action, action_scale, compute_go2w_motor_ctrl, _init_action_space
```

源码锚点拆解：

| 源码符号 | 中文说明 |
| --- | --- |

## Reward

reward 包含高度、姿态、支撑接触、动作平滑和能耗等专用项。

### Reward 逐项拆解

reward scale 以 owner YAML 的 `reward.scales` 为真源；源码中的 `RewardConfig` 描述可用字段，训练脚本只做 Hydra 组装和注入。正数通常表示奖励，负数通常表示惩罚，0 表示该项在当前 owner 中关闭或仅保留接口。

| Reward key | Scale | 方向 | 中文含义 |
| --- | --- | --- | --- |
| contact | -1 | 惩罚 | 接触项，约束指定足端或身体部位的接触模式。 |
| height | 1.0 | 奖励 | 目标高度奖励，用于 footstand/handstand 等姿态任务。 |
| oritentation | 1.0 | 奖励 | 朝向相关奖励/惩罚，用于约束姿态或参考朝向。 |
| pose | -0.3 | 惩罚 | 默认姿态或参考姿态约束，防止无关关节偏离可用构型。 |
| penalty_contact | -0.2 | 惩罚 | 接触项，约束指定足端或身体部位的接触模式。 |
| action_rate | -0.01 | 惩罚 | 动作变化惩罚，抑制相邻控制步之间的目标突变。 |
| tar | 0.3 | 奖励 | 目标前腿姿态项，FootStand 中约束前腿到目标角度。 |
| feet_air_time | 1 | 奖励 | 仓库 owner YAML 中声明的奖励项；具体实现见本页源码片段和 env 的 reward dispatch。 |
| world_z_vel_penalty | -1 | 惩罚 | 速度相关奖励/惩罚，用于跟踪目标速度或抑制不希望的运动。 |

```yaml
# conf/ppo/task/go2_handstand/mujoco.yaml
reward:
  scales:
    contact: -1
    height: 1.0
    oritentation: 1.0
    pose: -0.3
    penalty_contact: -0.2
    action_rate: -0.01
    tar: 0.3
    feet_air_time: 1
    world_z_vel_penalty: -1
  tracking_sigma: 0.25
  base_height_target: 0.3

```

源码中的 reward 入口：

```python
# src/unilab/envs/locomotion/go2/handstand.py

    randomize_kd: bool = True
    kd_multiplier_range: list[float] = field(default_factory=lambda: [0.9, 1.1])


@dataclass
class RewardConfig:
    scales: dict[str, float]
    tracking_sigma: float
    base_height_target: float
    target_foot_height: float = 0.1
    knee_height_target: float = 0.08

```

源码锚点拆解：

| 源码符号 | 中文说明 |
| --- | --- |
| RewardConfig: | 声明 reward 可用参数和默认 scale。 |
| _init_reward_functions | 把 reward 名称映射到源码函数，owner YAML 的 scale 会按这些 key 分发。 |
| _compute_reward | reward 相关源码锚点。 |
| _reward_foot_drag | 单个 reward 项的实现函数，通常返回每个 env 的 reward 向量。 |
| _reward_penalty_contact | 单个 reward 项的实现函数，通常返回每个 env 的 reward 向量。 |
| _reward_stand_contact | 单个 reward 项的实现函数，通常返回每个 env 的 reward 向量。 |
| _reward_swing_feet_z | 单个 reward 项的实现函数，通常返回每个 env 的 reward 向量。 |
| _reward_height | 单个 reward 项的实现函数，通常返回每个 env 的 reward 向量。 |
| _reward_orientation | 单个 reward 项的实现函数，通常返回每个 env 的 reward 向量。 |
| _reward_tar | 单个 reward 项的实现函数，通常返回每个 env 的 reward 向量。 |
| _reward_feet_air_time | 单个 reward 项的实现函数，通常返回每个 env 的 reward 向量。 |
| _reward_world_z_vel_penalty | 单个 reward 项的实现函数，通常返回每个 env 的 reward 向量。 |

## 初始状态

初始状态从 `home` keyframe 开始，训练中通过动作序列学习转入目标姿态。

机器人默认姿态来自 scene/task fragment 中的 keyframe；任务 reset 在此基础上加入命令、motion reference、物体状态或随机化。

```xml
<!-- src/unilab/assets/robots/go2/scene_flat.xml -->
    <contact name="RR_calf_contact2" geom1="floor" geom2="RR_calf_geom2" data="found" num="1" reduce="mindist"/>
  </sensor>

  <keyframe>
    <key name="home" qpos="
        0 0 0.3  1 0 0 0
        0 0.8 -1.5
```

```python
# src/unilab/envs/locomotion/go2/handstand.py
    reward_config: RewardConfig | None = None
    sensor: JoystickSensor = field(default_factory=JoystickSensor)  # type: ignore[assignment]
    domain_rand: Go2DomainRandConfig = field(default_factory=Go2DomainRandConfig)


class Go2HandStandDomainRandomizationProvider(LocomotionDRProvider):
    def _compute_reset_obs(
        self,
        env: Any,
        env_ids: Any,
        info_updates: Any,
        linvel: Any,
        gyro: Any,
```

## 终止条件

终止由高度、倾斜、接触失败或超时触发。

终止条件通常由 env 源码中的 `_compute_termination(s)`、reward config 阈值和 `max_episode_seconds` 共同决定。

```yaml
# conf/ppo/task/go2_handstand/mujoco.yaml
env:
  {}

reward:
  {}

```

```python
# src/unilab/envs/locomotion/go2/handstand.py
        contact = self.feet_force[:, [2, 3], 0] > 1.0
        contact_filt = np.logical_or(contact, self._last_contacts)
        self._last_contacts = contact
        # Increment air time
        self._feet_air_time[:, [2, 3]] += self._cfg.ctrl_dt
        # Reset air time for feet in contact
        self._feet_air_time[:, [2, 3]] *= ~contact_filt

        terminated_z = gravity[:, 2] <= -0.25
        terminated_contact = np.any(result, axis=1)
        # After 100 steps, terminate if height is too low (failed to maintain target)
        # step_count = state.info.get("steps", np.zeros((self._num_envs,), dtype=np.uint32))
        # terminated_height = (step_count >= 100) & (self.torso_height < self._z_des * 0.8)
        # terminated = np.logical_or(
        #     np.logical_or(terminated_contact, terminated_z),
        #     terminated_height
        # )
```

## 域随机化

domain_rand 使用 Go2 物理参数扰动提升非典型姿态下的鲁棒性。

域随机化只在 reset/interval 等冷路径触发，不在 step 热路径解析资产或探测 backend 私有能力。

### 域随机化字段拆解

| 配置字段 | 当前值 | 中文说明 |
| --- | --- | --- |

```yaml
# conf/ppo/task/go2_handstand/mujoco.yaml
env:
  domain_rand:
    {}

```

```python
# src/unilab/envs/locomotion/go2/handstand.py
from unilab.base.backend import create_backend
from unilab.base.np_env import NpEnvState
from unilab.base.scene import SceneCfg
from unilab.dtype_config import get_global_dtype
from unilab.envs.locomotion.common import rewards
from unilab.envs.locomotion.common.commands import Commands
from unilab.envs.locomotion.common.domain_rand import DomainRandConfig
from unilab.envs.locomotion.common.dr_provider import LocomotionDRProvider
from unilab.envs.locomotion.common.rewards import RewardContext
from unilab.envs.locomotion.go2.base import Go2BaseCfg, Go2BaseEnv


@dataclass
```

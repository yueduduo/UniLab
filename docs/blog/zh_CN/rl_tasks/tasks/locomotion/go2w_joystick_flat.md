---
title: "Go2W Joystick Flat"
slug: "go2w_joystick_flat"
category: "locomotion"
robot: "go2w"
registered_envs: [Go2WJoystickFlat]
---

# Go2W Joystick Flat

## 任务目标

训练轮足 Go2W 在平地上利用腿部和轮部共同跟踪 joystick 速度命令。

![Go2W Joystick Flat 场景渲染](../../images/tasks/go2w_joystick_flat.png)

> 图片由 `_tools/render_static_images.py` 使用 `src/unilab/assets/robots/go2w/scene_flat.xml` 的 MuJoCo scene 离屏渲染生成；如果当前环境无法启用 EGL/OSMesa，生成 manifest 会记录失败原因。

## 证据索引

| 项目 | 内容 |
| --- | --- |
| 任务类别 | 运动控制 |
| 机器人 | go2w |
| 代表 scene | src/unilab/assets/robots/go2w/scene_flat.xml |
| 代表源码 | src/unilab/envs/locomotion/go2w/joystick.py |
| 代表 owner YAML | conf/ppo/task/go2w_joystick_flat/mujoco.yaml |
| 动作维度 | 16 |

### Registry 与 Owner

| 注册名 | 后端 | 配置类 | 配置源码 |
| --- | --- | --- | --- |
| Go2WJoystickFlat | mujoco, motrix | Go2WJoystickCfg | src/unilab/envs/locomotion/go2w/joystick.py |

| 算法组 | 后端 | training.task_name | owner YAML |
| --- | --- | --- | --- |
| ppo | motrix | Go2WJoystickFlat | conf/ppo/task/go2w_joystick_flat/motrix.yaml |
| ppo | mujoco | Go2WJoystickFlat | conf/ppo/task/go2w_joystick_flat/mujoco.yaml |

## 代码级执行流

这部分不再用通用关键词套模板，而是为 `go2w_joystick_flat` 单独写源码导读。源码片段只负责提供证据；每节说明都围绕本任务自己的配置、状态、观测、动作和 reward 语义展开。

| 阶段 | 证据位置 | 代码级说明 |
| --- | --- | --- |
| 1. Hydra owner | conf/ppo/task/go2w_joystick_flat/mujoco.yaml | 训练命令先 compose owner YAML。这里确定 `training.task_name`、后端身份、obs_groups、reward scale 和 env override。 |
| 2. Registry 构造 Env | src/unilab/envs/locomotion/go2w/joystick.py | `@registry.envcfg` 注册配置类，`@registry.env` 注册后端实现类；训练脚本只调用 registry，不直接 new 任务类。 |
| 3. Reset/初始状态 | src/unilab/assets/robots/go2w/scene_flat.xml | env 从 scene keyframe 或 motion/grasp cache 取初态，再通过 reset randomization 改写 qpos/qvel/info。 |
| 4. Obs dict | src/unilab/envs/locomotion/go2w/joystick.py | env 读取 backend 传感器和 info，返回 dict；learner 根据 `obs_groups_spec` 和 owner `algo.obs_groups` 选择 actor/critic 输入。 |
| 5. Action 到控制目标 | src/unilab/envs/locomotion/go2w/joystick.py | policy 输出先按 action scale / 控制顺序映射到 actuator，再由 backend step 推进仿真。 |
| 6. Reward dispatch | src/unilab/envs/locomotion/go2w/joystick.py | 源码把 reward key 映射到函数，owner YAML 的 scale 决定每项奖励/惩罚是否启用以及权重。 |
| 7. Termination | src/unilab/envs/locomotion/go2w/joystick.py | env 根据高度、姿态、接触、参考轨迹误差、物体状态或能量阈值生成 done/terminated。 |

### 本任务阅读重点

| 阅读重点 |
| --- |
| Go2W 平地任务必须把腿部 12 DoF 和 4 个轮关节分开讲：action 维度变成 16，控制函数也不同于 Go2。 |
| obs 要看轮关节速度/动作历史如何进入策略输入，reward 要看轮足速度跟踪和姿态惩罚如何组合。 |
| 源码关键在 `Go2WJoystickEnv.apply_action` 与 `compute_go2w_motor_ctrl` 相关路径。 |

### 本任务注册入口

| 注册名 | 配置类 | 可用后端 |
| --- | --- | --- |
| Go2WJoystickFlat | Go2WJoystickCfg | mujoco, motrix |

### 本任务源码锚点

| 小节 | 源码文件 | 明确锚点 |
| --- | --- | --- |
| 配置与轮足模型 | src/unilab/envs/locomotion/go2w/joystick.py | Go2WJoystickCfg, Go2WJoystickFlat |
| Reset：轮足初态 | src/unilab/envs/locomotion/go2w/joystick.py | build_reset_plan, zero_actions |
| Obs：轮足状态 | src/unilab/envs/locomotion/go2w/joystick.py | obs_groups_spec, _compute_obs |
| Action：腿 + 轮 | src/unilab/envs/locomotion/go2w/joystick.py | apply_action, compute_go2w_motor_ctrl |
| Reward/Termination | src/unilab/envs/locomotion/go2w/joystick.py | _init_reward_functions, update_state |

## 关键源码逐段解释（按本任务手写）

### 配置与轮足模型

| 本任务人工导读 |
| --- |
| cfg 指向 Go2W scene，模型包含 12 个腿关节和 4 个轮 actuator。 |
| 轮足控制保持在 env/backend 边界内，训练脚本不解释轮子。 |

```python
# src/unilab/envs/locomotion/go2w/joystick.py
# 注册名就是 owner YAML 里的 training.task_name，训练脚本通过 registry 找到这个 cfg。
@registry.envcfg("Go2WJoystickFlat")
@dataclass
class Go2WJoystickCfg(Go2WBaseCfg):
# flat 任务直接使用平地 scene，轮足模型和 floor 都在这个 scene 中编译。
    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(
            model_file=str(ASSETS_ROOT_PATH / "robots" / "go2w" / "scene_flat.xml")
        )
    )
# 每个 episode 最长 20 秒，超时由基类 truncated 逻辑处理。
    max_episode_seconds: float = 20.0
# init_state/commands/reward_config 都是 Hydra 注入到 env 的任务契约字段。
    init_state: InitState = field(default_factory=InitState)
    commands: Commands = field(default_factory=Commands)
    reward_config: RewardConfig | None = None
# Go2W joystick 读取 local_linvel、gyro 和 upvector；obs/reward 不直接读 XML。
    sensor: JoystickSensor = field(default_factory=JoystickSensor)  # type: ignore[assignment]
# flat owner 可关闭 kp/kd 随机化，但配置类仍声明 Go2W 专用 DR 字段。
    domain_rand: Go2WDomainRandConfig = field(default_factory=Go2WDomainRandConfig)


```
### Reset：轮足初态

| 本任务人工导读 |
| --- |
| reset 初始化 commands、动作历史和轮足状态。 |
| Go2W 的 16 维 action history 必须和 actuator 顺序一致。 |

```python
# src/unilab/envs/locomotion/go2w/joystick.py
    def build_reset_plan(self, env: Any, env_ids: np.ndarray) -> ResetPlan:
# num_reset 是本批要重置的并行环境数量。
        num_reset = len(env_ids)
# 从 scene 的 home keyframe 复制机器人初态，避免 reset 热路径解析 asset。
        qpos = np.tile(env._init_qpos, (num_reset, 1))
        qvel = np.tile(env._init_qvel, (num_reset, 1))
# 平地任务只在 xy 平面给初始位置小扰动。
        qpos[:, 0:2] += np.random.uniform(-0.5, 0.5, (num_reset, 2))
# spawn manager 提供每个 env 的平铺原点，保证多环境不重叠。
        qpos[:, 0:3] += env._spawn.origins_for(env_ids)
# flat Go2W 只随机 yaw，保持 roll/pitch 不被 reset 主动打乱。
        yaw = sample_go2w_reset_yaw(env.cfg.domain_rand, num_reset)
        qpos[:, 3:7] = np_quat_mul(qpos[:, 3:7], np_yaw_to_quat(yaw))
# 给 base 初速度一点扰动，让策略学会从轻微运动状态恢复。
        qvel[:, 0:6] = np.asarray(
            np.random.uniform(-0.5, 0.5, size=(num_reset, 6)), dtype=get_global_dtype()
        )

# reset 时采样腿部 PD 增益；wheel kd 使用 env 缓存，不放进 backend reset payload。
        motor_kp, motor_kd = env.sample_reset_motor_gains(num_reset)
        env.set_motor_gains(env_ids, motor_kp, motor_kd)

# commands 是 joystick 速度目标，后续 _update_commands 会按 resampling_time 更新。
        commands = self._sample_commands(env, num_reset)
        info_updates: dict[str, Any] = {
            "commands": commands,
# Go2W action 是 16 维：前 12 维腿，后 4 维轮。
            "current_actions": zero_actions(num_reset, env._num_action),
            "last_actions": zero_actions(num_reset, env._num_action),
# motor_kp/motor_kd 写入 info，便于 reset 后记录和调试当前随机化结果。
            "motor_kp": motor_kp.astype(get_global_dtype()),
            "motor_kd": motor_kd.astype(get_global_dtype()),
# torques 先置零，第一步 update_state 会写入 pre-step motor control 的结果。
            "torques": np.zeros((num_reset, env._num_action), dtype=get_global_dtype()),
        }
# heading_command 打开时，额外采样目标朝向并由 yaw feedback 转成 yaw rate command。
        if getattr(env.cfg.commands, "heading_command", False):
            info_updates["heading_commands"] = sample_go2w_heading_commands(env, num_reset)
# ResetPlan 把 qpos/qvel/info 和后端支持的随机化 payload 一次性交给 runner。
        return ResetPlan(
            env_ids=env_ids,
            qpos=qpos,
            qvel=qvel,
            info_updates=info_updates,
            randomization=build_go2w_backend_reset_randomization(env, num_reset),
        )

```

```python
# src/unilab/envs/locomotion/go2w/base.py
# 12 个腿关节来自四条腿的 hip/thigh/calf。
NUM_LEG_ACTIONS = len(LEG_JOINT_SENSOR_PREFIXES)
# 4 个 wheel joint 单独计数，不能混进腿部位置控制。
NUM_WHEEL_ACTIONS = len(WHEEL_JOINT_SENSOR_PREFIXES)
# policy action 和 owner-level control 都是 12 + 4 = 16 维。
NUM_GO2W_ACTIONS = len(JOINT_SENSOR_PREFIXES)
```
### Obs：轮足状态

| 本任务人工导读 |
| --- |
| obs 维度大于 Go2，因为包含轮关节相关状态。 |
| critic 继续携带额外速度/特权信息。 |

```python
# src/unilab/envs/locomotion/go2w/joystick.py
    @property
    def obs_groups_spec(self) -> dict[str, int]:
# actor 观测 53 维；critic 多拿 linvel 和 motor_ctrl，合计 72 维。
        return {"obs": 53, "critic": 72}

```

```python
# src/unilab/envs/locomotion/go2w/joystick.py
    def _compute_obs(
        self,
        info: dict,
        linvel: np.ndarray,
        gyro: np.ndarray,
        gravity: np.ndarray,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
    ) -> dict[str, np.ndarray]:
# 噪声大小来自 cfg.noise_config，flat owner 可通过 Hydra 调整 level。
        noise_cfg = self._cfg.noise_config
# 腿部关节用“当前角度 - 默认站姿角度”表达，轮子没有位置误差项。
        leg_diff = dof_pos[:, :NUM_LEG_ACTIONS] - self.default_angles[:NUM_LEG_ACTIONS]
        leg_vel = dof_vel[:, :NUM_LEG_ACTIONS]
# 轮子进入观测的是速度，反映轮部推进状态。
        wheel_vel = dof_vel[:, NUM_LEG_ACTIONS:]
# actor 观测带噪声，帮助 sim-to-real；critic 下面会拿同一批处理后的 obs 再追加特权量。
        gyro = self._obs_noise(gyro, noise_cfg.scale_gyro)
        gravity = self._obs_noise(gravity, noise_cfg.scale_gravity)
        leg_diff = self._obs_noise(leg_diff, noise_cfg.scale_joint_angle)
        leg_vel = self._obs_noise(leg_vel, noise_cfg.scale_joint_vel)
        wheel_vel = self._obs_noise(wheel_vel, noise_cfg.scale_wheel_vel)
        linvel = self._obs_noise(linvel, noise_cfg.scale_linvel)
        num_obs = gyro.shape[0]
# current_actions 作为动作历史输入，帮助策略抑制高频抖动。
        last_actions = info.get("current_actions", np.zeros((num_obs, self._num_action)))
# torques 是 pre-step motor control 的输出，只给 critic 作为特权信息。
        motor_ctrl = info.get("torques", np.zeros((num_obs, self._num_action), dtype=dof_pos.dtype))

# actor 输入顺序：IMU、腿姿态、腿速、轮速、动作历史、joystick 命令。
        obs = np.concatenate(
            [gyro, -gravity, leg_diff, leg_vel, wheel_vel, last_actions, info["commands"]],
            axis=1,
            dtype=get_global_dtype(),
        )
# critic 在 actor obs 后追加 local linvel 和 16 维 motor torque，解释 72 维来源。
        critic = np.concatenate(
            [obs, linvel, motor_ctrl],
            axis=1,
            dtype=get_global_dtype(),
        )
# NpEnv contract 要求 obs 是 dict，wrapper/learner 再按 obs_groups_spec 取组。
        return {"obs": obs, "critic": critic}

```
### Action：腿 + 轮

| 本任务人工导读 |
| --- |
| 腿部按位置目标控制，轮部由 Go2W 专用控制 helper 转换。 |
| 这就是为什么不能把 Go2W 简单当成 16 维 Go2。 |

```python
# src/unilab/envs/locomotion/go2w/joystick.py
    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> np.ndarray:
# 先裁剪 policy 输出，flat owner 默认 clip_actions 来自 Go2W ControlConfig。
        clipped_actions = np.asarray(
            np.clip(
                actions,
                -self._cfg.control_config.clip_actions,
                self._cfg.control_config.clip_actions,
            ),
            dtype=self._np_dtype,
        )
# last/current action 都写入 info，obs 和 action_rate reward 都会用到。
        state.info["last_actions"] = state.info.get(
            "current_actions", np.zeros_like(clipped_actions)
        )
        state.info["current_actions"] = clipped_actions
# 如果模拟控制延迟，本步执行上一帧 action；否则执行当前 policy 输出。
        exec_actions = (
            state.info["last_actions"]
            if self._cfg.control_config.simulate_action_latency
            else clipped_actions
        )

# 前 12 维是腿部位置目标：action_scale 后加默认站姿角。
        leg_targets = (
            exec_actions[:, :NUM_LEG_ACTIONS] * self._leg_action_scale
            + self.default_angles[:NUM_LEG_ACTIONS]
        )
# 后 4 维是轮速目标，不加默认角，直接按 wheel_action_scale 缩放。
        wheel_velocity_targets = (
            exec_actions[:, NUM_LEG_ACTIONS:] * self._cfg.control_config.wheel_action_scale
        )
# 返回 owner-level control，真正转 torque 在 pre-step motor control 中完成。
        return np.concatenate([leg_targets, wheel_velocity_targets], axis=1, dtype=self._np_dtype)

```

```python
# src/unilab/envs/locomotion/go2w/base.py
def compute_go2w_motor_ctrl(
    policy_ctrl: np.ndarray,
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    leg_kp: np.ndarray,
    leg_kd: np.ndarray,
    wheel_kd: np.ndarray,
    ctrl_lower: np.ndarray,
    ctrl_upper: np.ndarray,
    out: np.ndarray,
) -> np.ndarray:
# 腿部把位置目标减去当前关节角，得到位置误差。
    leg_out = out[:, :NUM_LEG_ACTIONS]
    np.subtract(policy_ctrl[:, :NUM_LEG_ACTIONS], joint_pos[:, :NUM_LEG_ACTIONS], out=leg_out)
# 腿部 torque = kp * 位置误差 - kd * 当前关节速度。
    np.multiply(leg_out, leg_kp, out=leg_out)
    leg_out -= leg_kd * joint_vel[:, :NUM_LEG_ACTIONS]
# 轮部把“目标轮速 - 当前轮速”转成阻尼控制输出。
    wheel_out = out[:, NUM_LEG_ACTIONS:]
    np.subtract(policy_ctrl[:, NUM_LEG_ACTIONS:], joint_vel[:, NUM_LEG_ACTIONS:], out=wheel_out)
    np.multiply(wheel_out, wheel_kd, out=wheel_out)
# 最后统一按 actuator ctrl range 裁剪，避免越过后端执行器限制。
    np.clip(out, ctrl_lower, ctrl_upper, out=out)
    return out
```
### Reward/Termination

| 本任务人工导读 |
| --- |
| reward 仍以速度跟踪为主，但加入轮足稳定和动作平滑约束。 |
| 终止关注 base 高度、倾斜和轮足异常。 |

```python
# src/unilab/envs/locomotion/go2w/joystick.py
    def _init_reward_functions(self) -> None:
# reward key 必须和 owner YAML 的 reward.scales 对齐，scale 决定启用和权重。
        self._reward_fns: dict[str, Any] = {
# 速度跟踪是 joystick flat 的主目标：跟踪 xy 线速度和 yaw 角速度。
            "tracking_lin_vel": rewards.tracking_lin_vel,
            "tracking_ang_vel": rewards.tracking_ang_vel,
# 抑制垂直弹跳和 roll/pitch 角速度，保持轮足底盘稳定。
            "lin_vel_z": rewards.lin_vel_z,
            "ang_vel_xy": rewards.ang_vel_xy,
# base_height/orientation/upward 共同约束身体高度和朝上姿态。
            "base_height": rewards.base_height,
            "action_rate": rewards.action_rate,
            "similar_to_default": rewards.similar_to_default,
            "orientation": rewards.orientation,
# Go2W 专用项把 16 维 torque 拆成腿部、轮部和能量相关惩罚。
            "torques": self._reward_torques_l2,
            "joint_torques_l2": self._reward_joint_torques_l2,
            "energy": rewards.energy,
            "dof_vel": self._reward_dof_vel,
            "dof_acc": self._reward_dof_acc,
            "joint_acc_l2": self._reward_dof_acc,
            "wheel_acc": self._reward_wheel_acc,
            "joint_acc_wheel_l2": self._reward_wheel_acc,
# 静止命令、髋关节、默认姿态、镜像项用于避免轮足姿态跑偏。
            "stand_still": self._reward_stand_still,
            "hip_pos": self._reward_hip_pos,
            "dof_error": self._reward_dof_error,
            "joint_pos_penalty": self._reward_joint_pos_penalty,
            "joint_power": self._reward_joint_power,
            "joint_mirror": self._reward_joint_mirror,
            "alive": rewards.alive,
            "upward": rewards.upward,
# wheel_vel 在 flat owner 中 scale 为 0，保留接口但当前不参与总 reward。
            "wheel_vel": self._reward_wheel_vel,
        }

```

```python
# src/unilab/envs/locomotion/go2w/joystick.py
    def update_state(self, state: NpEnvState) -> NpEnvState:
# 每步先根据 resampling_time/heading command 更新速度命令。
        self._update_commands(state.info)
# 这些 backend 读数同时驱动 termination、reward 和 obs。
        linvel = self.get_local_linvel()
        gyro = self.get_gyro()
        gravity = self._backend.get_sensor_data(self._cfg.sensor.gravity)
        dof_pos = self.get_dof_pos()
        dof_vel = self.get_dof_vel()
# pre-step torque 和估算 qacc 写入 info，供 critic/reward 使用。
        state.info["torques"] = self._last_motor_ctrl.copy()
        state.info["qacc"] = self._estimate_dof_acc(dof_vel)
# flat Go2W 的终止只看 upvector 的 z 分量，低于阈值代表倾倒。
        terminated = self._compute_terminated(gravity)
# reward dispatch 使用 owner YAML 的 scales 映射到上面的 _reward_fns。
        reward = self._compute_reward(state.info, linvel, gyro, gravity, dof_pos, dof_vel)
# obs 仍返回 dict，actor/critic 组由 obs_groups_spec 声明。
        obs = self._compute_obs(state.info, linvel, gyro, gravity, dof_pos, dof_vel)
        return state.replace(obs=obs, reward=reward, terminated=terminated)

```



## Agent

Agent 输出 16 维动作，对应 12 个腿关节和 4 个轮关节。

训练入口通过 owner YAML 决定注册 env、后端身份和算法参数；CLI 应使用 `--sim` 选择 backend owner，而不是单独 override `training.sim_backend`。

```yaml
# conf/ppo/task/go2w_joystick_flat/mujoco.yaml
training:
  task_name: Go2WJoystickFlat
  sim_backend: mujoco
algo:
  obs_groups:
    actor:
    - actor
  num_envs: 1024
  max_iterations: 151
```

## Env

Env 使用 Go2W 专用 motor control helper，把轮足控制封装在 env 层而不是训练脚本。

Env 必须遵守 `NpEnv` contract：`reset()` 返回 `(obs_dict, info_dict)`，`obs` 是 dict，`obs_groups_spec` 决定 learner 使用哪些观测组。

```python
# src/unilab/envs/locomotion/go2w/joystick.py
    local_linvel = "local_linvel"
    gyro = "gyro"
    gravity = "upvector"


@registry.envcfg("Go2WJoystickFlat")
@dataclass
class Go2WJoystickCfg(Go2WBaseCfg):
    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(
            model_file=str(ASSETS_ROOT_PATH / "robots" / "go2w" / "scene_flat.xml")
```

## Obs

观测在 Go2 基础上加入轮关节状态和轮足特定动作历史。

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
# conf/ppo/task/go2w_joystick_flat/mujoco.yaml
algo:
  obs_groups:
    actor:
    - actor

```

代码侧观测通常在 `_get_obs`、`_build_obs`、`obs_groups_spec` 或 base env 中组装：

```python
# src/unilab/envs/locomotion/go2w/joystick.py
        )
        self._backend.set_pre_step_control(self._pre_step_motor_control)
        self._init_reward_functions()
        self._init_domain_randomization(Go2WJoystickDomainRandomizationProvider())

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        return {"obs": 53, "critic": 72}

    def reset(self, env_indices: np.ndarray) -> tuple[dict[str, np.ndarray], dict]:
        env_ids = np.asarray(env_indices, dtype=np.int32)
        obs, info = super().reset(env_ids)
        dof_vel = self.get_dof_vel()
```

源码锚点拆解：

| 源码符号 | 中文说明 |
| --- | --- |
| obs_groups_spec | 声明 obs dict 中各观测组的维度，是 wrapper/learner 对齐观测的契约。 |
| _compute_obs | 读取 backend 传感器和 info 字段，拼接 actor/critic 观测。 |

## Action

动作空间包含腿部位置目标与轮部目标，XML 编译 `nu=16`。

### Action 逐维拆解

Go2W 的 action 包含 12 个腿部关节和 4 个轮关节。腿部仍是位置目标，轮部由 Go2W 专用控制逻辑解释，文档表中把轮关节单独标出。

当前代表 scene 编译出的 action 维度为 `16`。下表来自 MuJoCo 编译模型的 actuator 顺序，也就是策略输出维度和执行器维度的对应关系。

| Action 维度 | 执行器 | 驱动关节 | 中文含义 | 控制/限制说明 |
| --- | --- | --- | --- | --- |
| 0 | FR_hip | FR_hip_joint | 右前腿髋外展/内收关节 | -23.7 ~ 23.7 |
| 1 | FR_thigh | FR_thigh_joint | 右前腿髋俯仰（大腿）关节 | -23.7 ~ 23.7 |
| 2 | FR_calf | FR_calf_joint | 右前腿膝关节（小腿摆动） | -45.43 ~ 45.43 |
| 3 | FL_hip | FL_hip_joint | 左前腿髋外展/内收关节 | -23.7 ~ 23.7 |
| 4 | FL_thigh | FL_thigh_joint | 左前腿髋俯仰（大腿）关节 | -23.7 ~ 23.7 |
| 5 | FL_calf | FL_calf_joint | 左前腿膝关节（小腿摆动） | -45.43 ~ 45.43 |
| 6 | RR_hip | RR_hip_joint | 右后腿髋外展/内收关节 | -23.7 ~ 23.7 |
| 7 | RR_thigh | RR_thigh_joint | 右后腿髋俯仰（大腿）关节 | -23.7 ~ 23.7 |
| 8 | RR_calf | RR_calf_joint | 右后腿膝关节（小腿摆动） | -45.43 ~ 45.43 |
| 9 | RL_hip | RL_hip_joint | 左后腿髋外展/内收关节 | -23.7 ~ 23.7 |
| 10 | RL_thigh | RL_thigh_joint | 左后腿髋俯仰（大腿）关节 | -23.7 ~ 23.7 |
| 11 | RL_calf | RL_calf_joint | 左后腿膝关节（小腿摆动） | -45.43 ~ 45.43 |
| 12 | FR_wheel | FR_wheel_joint | 右前腿轮子滚转关节 | -15 ~ 15 |
| 13 | FL_wheel | FL_wheel_joint | 左前腿轮子滚转关节 | -15 ~ 15 |
| 14 | RR_wheel | RR_wheel_joint | 右后腿轮子滚转关节 | -15 ~ 15 |
| 15 | RL_wheel | RL_wheel_joint | 左后腿轮子滚转关节 | -15 ~ 15 |

控制参数来自 owner YAML 的 `env.control_config`，缺省值由对应 env cfg/dataclass 提供。

| 配置字段 | 当前值 | 中文说明 |
| --- | --- | --- |
| Kd | 1.5 | 任务 owner YAML 中的配置字段。 |
| Kp | 50.0 | 任务 owner YAML 中的配置字段。 |
| action_scale | 0.5 | 策略输出到目标关节角的缩放系数。 |
| wheel_Kd | 0.5 | 任务 owner YAML 中的配置字段。 |
| wheel_action_scale | 10.0 | 策略输出到目标关节角的缩放系数。 |

```yaml
# conf/ppo/task/go2w_joystick_flat/mujoco.yaml
env:
  control_config:
    action_scale: 0.5
    wheel_action_scale: 10.0
    Kp: 50.0
    Kd: 1.5
    wheel_Kd: 0.5

```

动作到执行器的实际映射由 backend 的 actuator control range 和 env 的 `apply_action`/控制函数完成：

```python
# src/unilab/envs/locomotion/go2w/joystick.py
        return kp, kd

    def set_motor_gains(self, env_ids: np.ndarray, kp: np.ndarray, kd: np.ndarray) -> None:
        self._motor_kp[env_ids] = np.asarray(kp, dtype=np.float64)
        self._motor_kd[env_ids] = np.asarray(kd, dtype=np.float64)

    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> np.ndarray:
        clipped_actions = np.asarray(
            np.clip(
                actions,
                -self._cfg.control_config.clip_actions,
                self._cfg.control_config.clip_actions,
            ),
```

源码锚点拆解：

| 源码符号 | 中文说明 |
| --- | --- |
| apply_action | 把策略输出缩放为执行器控制目标。 |

## Reward

reward 以速度跟踪为核心，同时惩罚轮足不稳定、动作变化和姿态偏差。

### Reward 逐项拆解

reward scale 以 owner YAML 的 `reward.scales` 为真源；源码中的 `RewardConfig` 描述可用字段，训练脚本只做 Hydra 组装和注入。正数通常表示奖励，负数通常表示惩罚，0 表示该项在当前 owner 中关闭或仅保留接口。

| Reward key | Scale | 方向 | 中文含义 |
| --- | --- | --- | --- |
| tracking_lin_vel | 1.0 | 奖励 | 线速度命令跟踪奖励，鼓励机器人实际平面速度贴近 joystick/命令速度。 |
| tracking_ang_vel | 0.75 | 奖励 | 角速度命令跟踪奖励，鼓励 yaw 角速度贴近转向命令。 |
| lin_vel_z | -5.0 | 惩罚 | 竖直速度惩罚，限制 base 上下弹跳。 |
| ang_vel_xy | -0.1 | 惩罚 | 横滚/俯仰角速度惩罚，抑制身体剧烈摇摆。 |
| base_height | -100.0 | 惩罚 | 身体高度项，使 base 高度保持在任务目标附近。 |
| orientation | -2.0 | 惩罚 | 姿态项，通常基于重力投影或目标朝向惩罚倾斜。 |
| action_rate | -0.005 | 惩罚 | 动作变化惩罚，抑制相邻控制步之间的目标突变。 |
| similar_to_default | -0.5 | 惩罚 | 仓库 owner YAML 中声明的奖励项；具体实现见本页源码片段和 env 的 reward dispatch。 |
| torques | -0.0002 | 惩罚 | 力矩惩罚，降低控制输出过大。 |
| wheel_vel | 0.0 | 记录/关闭 | 速度相关奖励/惩罚，用于跟踪目标速度或抑制不希望的运动。 |
| alive | 0.5 | 奖励 | 存活奖励，鼓励保持非终止状态。 |
| upward | 1.0 | 奖励 | 仓库 owner YAML 中声明的奖励项；具体实现见本页源码片段和 env 的 reward dispatch。 |

```yaml
# conf/ppo/task/go2w_joystick_flat/mujoco.yaml
reward:
  scales:
    tracking_lin_vel: 1.0
    tracking_ang_vel: 0.75
    lin_vel_z: -5.0
    ang_vel_xy: -0.1
    base_height: -100.0
    orientation: -2.0
    action_rate: -0.005
    similar_to_default: -0.5
    torques: -0.0002
    wheel_vel: 0.0
    alive: 0.5
    upward: 1.0
  tracking_sigma: 0.25
  base_height_target: 0.4

```

源码中的 reward 入口：

```python
# src/unilab/envs/locomotion/go2w/joystick.py

    randomize_kd: bool = True
    kd_multiplier_range: list[float] = field(default_factory=lambda: [0.9, 1.1])


@dataclass
class RewardConfig:
    scales: dict[str, float]
    tracking_sigma: float
    base_height_target: float
    only_positive_rewards: bool = False
    joint_pos_penalty_stand_still_scale: float = 5.0
    joint_pos_penalty_velocity_threshold: float = 0.5
```

源码锚点拆解：

| 源码符号 | 中文说明 |
| --- | --- |
| RewardConfig: | 声明 reward 可用参数和默认 scale。 |
| _init_reward_functions | 把 reward 名称映射到源码函数，owner YAML 的 scale 会按这些 key 分发。 |
| _compute_reward | reward 相关源码锚点。 |
| _reward_base_height_values | 单个 reward 项的实现函数，通常返回每个 env 的 reward 向量。 |
| _reward_wheel_vel | 单个 reward 项的实现函数，通常返回每个 env 的 reward 向量。 |
| _reward_torques_l2 | 单个 reward 项的实现函数，通常返回每个 env 的 reward 向量。 |
| _reward_joint_torques_l2 | 单个 reward 项的实现函数，通常返回每个 env 的 reward 向量。 |
| _reward_dof_vel | 单个 reward 项的实现函数，通常返回每个 env 的 reward 向量。 |
| _reward_dof_acc | 单个 reward 项的实现函数，通常返回每个 env 的 reward 向量。 |
| _reward_wheel_acc | 单个 reward 项的实现函数，通常返回每个 env 的 reward 向量。 |
| _reward_stand_still | 单个 reward 项的实现函数，通常返回每个 env 的 reward 向量。 |
| _reward_hip_pos | 单个 reward 项的实现函数，通常返回每个 env 的 reward 向量。 |

## 初始状态

初始姿态来自 Go2W `home` keyframe。

机器人默认姿态来自 scene/task fragment 中的 keyframe；任务 reset 在此基础上加入命令、motion reference、物体状态或随机化。

```xml
<!-- src/unilab/assets/robots/go2w/scene_flat.xml -->
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>
  </worldbody>

  <keyframe>
    <key name="home" qpos="
        0 0 0.42  1 0 0 0
        0 0.8 -1.5 0
```

```python
# src/unilab/envs/locomotion/go2w/joystick.py
    commands: Commands = field(default_factory=Commands)
    reward_config: RewardConfig | None = None
    sensor: JoystickSensor = field(default_factory=JoystickSensor)  # type: ignore[assignment]
    domain_rand: Go2WDomainRandConfig = field(default_factory=Go2WDomainRandConfig)


def build_go2w_backend_reset_randomization(
    env: Any, num_reset: int
) -> ResetRandomizationPayload | None:
    """Build reset DR payloads that are valid for a motor-actuator Go2W model.

    kp/kd are intentionally excluded here. Go2W samples them through the same
    config path as Go2, but applies them inside its owner pre-step motor control.
```

## 终止条件

终止关注 base 高度、倾斜和轮足异常状态。

终止条件通常由 env 源码中的 `_compute_termination(s)`、reward config 阈值和 `max_episode_seconds` 共同决定。

```yaml
# conf/ppo/task/go2w_joystick_flat/mujoco.yaml
env:
  {}

reward:
  {}

```

```python
# src/unilab/envs/locomotion/go2w/joystick.py
        self._update_commands(state.info)
        linvel = self.get_local_linvel()
        gyro = self.get_gyro()
        gravity = self._backend.get_sensor_data(self._cfg.sensor.gravity)
        dof_pos = self.get_dof_pos()
        dof_vel = self.get_dof_vel()
        state.info["torques"] = self._last_motor_ctrl.copy()
        state.info["qacc"] = self._estimate_dof_acc(dof_vel)
        terminated = self._compute_terminated(gravity)
        reward = self._compute_reward(state.info, linvel, gyro, gravity, dof_pos, dof_vel)
        obs = self._compute_obs(state.info, linvel, gyro, gravity, dof_pos, dof_vel)
        return state.replace(obs=obs, reward=reward, terminated=terminated)

    def _compute_terminated(self, gravity: np.ndarray) -> np.ndarray:
        return gravity[:, 2] <= 0.5

    def _compute_obs(
```

## 域随机化

domain_rand 与四足任务类似，但作用到轮足模型的质量、摩擦、推力和控制参数。

域随机化只在 reset/interval 等冷路径触发，不在 step 热路径解析资产或探测 backend 私有能力。

### 域随机化字段拆解

| 配置字段 | 当前值 | 中文说明 |
| --- | --- | --- |
| randomize_kd | False | 域随机化开关。 |
| randomize_kp | False | 域随机化开关。 |

```yaml
# conf/ppo/task/go2w_joystick_flat/mujoco.yaml
env:
  domain_rand:
    randomize_kp: false
    randomize_kd: false

```

```python
# src/unilab/envs/locomotion/go2w/joystick.py

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.backend import create_backend
from unilab.base.np_env import NpEnvState
from unilab.base.scene import SceneCfg
from unilab.dr import DomainRandomizationCapabilities, ResetPlan, ResetRandomizationPayload
from unilab.dr.dr_utils import (
    build_interval_push_plan,
    validate_interval_push_support,
    zero_actions,
)
from unilab.dtype_config import get_global_dtype
```

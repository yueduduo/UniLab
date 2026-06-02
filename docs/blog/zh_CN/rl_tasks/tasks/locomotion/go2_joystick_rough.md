---
title: "Go2 Joystick Rough"
slug: "go2_joystick_rough"
category: "locomotion"
robot: "go2"
registered_envs: [Go2JoystickRough]
---

# Go2 Joystick Rough

## 任务目标

在崎岖地形上训练 Go2 保持速度命令跟踪，并学习利用足端接触和地形高度信息稳定前进。

![Go2 Joystick Rough 场景渲染](../../images/tasks/go2_joystick_rough.png)

> 图片由 `_tools/render_static_images.py` 使用 `src/unilab/assets/robots/go2/scene_flat.xml` 的 MuJoCo scene 离屏渲染生成；如果当前环境无法启用 EGL/OSMesa，生成 manifest 会记录失败原因。

## 证据索引

| 项目 | 内容 |
| --- | --- |
| 任务类别 | 运动控制 |
| 机器人 | go2 |
| 代表 scene | src/unilab/assets/robots/go2/scene_flat.xml |
| 代表源码 | src/unilab/envs/locomotion/go2/rough.py |
| 代表 owner YAML | conf/ppo/task/go2_joystick_rough/mujoco.yaml |
| 动作维度 | 12 |

### Registry 与 Owner

| 注册名 | 后端 | 配置类 | 配置源码 |
| --- | --- | --- | --- |
| Go2JoystickRough | mujoco, motrix | Go2JoystickRoughCfg | src/unilab/envs/locomotion/go2/rough.py |

| 算法组 | 后端 | training.task_name | owner YAML |
| --- | --- | --- | --- |
| ppo | motrix | Go2JoystickRough | conf/ppo/task/go2_joystick_rough/motrix.yaml |
| ppo | mujoco | Go2JoystickRough | conf/ppo/task/go2_joystick_rough/mujoco.yaml |

## 代码级执行流

这部分不再用通用关键词套模板，而是为 `go2_joystick_rough` 单独写源码导读。源码片段只负责提供证据；每节说明都围绕本任务自己的配置、状态、观测、动作和 reward 语义展开。

| 阶段 | 证据位置 | 代码级说明 |
| --- | --- | --- |
| 1. Hydra owner | conf/ppo/task/go2_joystick_rough/mujoco.yaml | 训练命令先 compose owner YAML。这里确定 `training.task_name`、后端身份、obs_groups、reward scale 和 env override。 |
| 2. Registry 构造 Env | src/unilab/envs/locomotion/go2/rough.py | `@registry.envcfg` 注册配置类，`@registry.env` 注册后端实现类；训练脚本只调用 registry，不直接 new 任务类。 |
| 3. Reset/初始状态 | src/unilab/assets/robots/go2/scene_flat.xml | env 从 scene keyframe 或 motion/grasp cache 取初态，再通过 reset randomization 改写 qpos/qvel/info。 |
| 4. Obs dict | src/unilab/envs/locomotion/go2/rough.py | env 读取 backend 传感器和 info，返回 dict；learner 根据 `obs_groups_spec` 和 owner `algo.obs_groups` 选择 actor/critic 输入。 |
| 5. Action 到控制目标 | src/unilab/envs/locomotion/go2/rough.py | policy 输出先按 action scale / 控制顺序映射到 actuator，再由 backend step 推进仿真。 |
| 6. Reward dispatch | src/unilab/envs/locomotion/go2/rough.py | 源码把 reward key 映射到函数，owner YAML 的 scale 决定每项奖励/惩罚是否启用以及权重。 |
| 7. Termination | src/unilab/envs/locomotion/go2/rough.py | env 根据高度、姿态、接触、参考轨迹误差、物体状态或能量阈值生成 done/terminated。 |

### 本任务阅读重点

| 阅读重点 |
| --- |
| Go2 rough 是 Go2 平地任务的地形扩展，必须单独看 `Go2JoystickRoughEnv` 中 terrain、height scan 和 rough reward。 |
| 动作仍是 12 维腿部目标，策略复杂度来自 obs/reward/termination，而不是训练脚本或 action 解释变化。 |
| 这个任务的代码重点是：reset 如何采样地形、critic 如何获得高度扫描、reward 如何门控姿态与足端项。 |

### 本任务注册入口

| 注册名 | 配置类 | 可用后端 |
| --- | --- | --- |
| Go2JoystickRough | Go2JoystickRoughCfg | mujoco, motrix |

### 本任务源码锚点

| 小节 | 源码文件 | 明确锚点 |
| --- | --- | --- |
| 配置与地形 | src/unilab/envs/locomotion/go2/rough.py | Go2JoystickRoughCfg, Go2RoughTerrainCfg |
| Reset：地形初态 | src/unilab/envs/locomotion/go2/rough.py | build_reset_plan, spawn |
| Obs：height scan critic | src/unilab/envs/locomotion/go2/rough.py | obs_groups_spec, _compute_obs |
| Action：裁剪后映射 | src/unilab/envs/locomotion/go2/rough.py | apply_action, clip_actions |
| Reward/Termination | src/unilab/envs/locomotion/go2/rough.py | _init_reward_functions, update_state |

## 关键源码逐段解释（按本任务手写）

### 配置与地形

| 本任务人工导读 |
| --- |
| rough cfg 声明 terrain generator 和 scene fragment。 |
| 地形属于 task scene，robot.xml 仍保持纯机器人描述。 |

```python
# src/unilab/envs/locomotion/go2/rough.py
# rough 任务用独立注册名，owner YAML 的 training.task_name 选择这个配置。
@registry.envcfg("Go2JoystickRough")
@dataclass
class Go2JoystickRoughCfg(Go2JoystickCfg):
    # rough 不直接使用 scene_flat.xml，而是机器人 XML + locomotion task fragment + terrain hfield。
    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(
            model_file=str(ASSETS_ROOT_PATH / "robots" / "go2" / "go2.xml"),
            # keyframe/任务场景属于 fragment，避免把 task 语义写进 robot.xml。
            fragment_files=[
                str(ASSETS_ROOT_PATH / "robots" / "go2" / "locomotion_task.xml"),
            ],
            # TerrainSceneCfg 告诉 backend 生成 hfield 并把它挂到 floor geom。
            terrain=TerrainSceneCfg(
                generator=Go2RoughTerrainCfg(),
                hfield_name="terrain_hfield",
                geom_name="floor",
            ),
        )
    )
    # rough 使用更细的 hip/non-hip action scale 和动作裁剪。
    control_config: RoughControlConfig = field(default_factory=RoughControlConfig)
    # 命令支持 heading_command，yaw 速度可由朝向误差反馈生成。
    commands: RoughCommands = field(default_factory=RoughCommands)
    # critic 的地形高度扫描配置，用于感知脚下/前方高度。
    terrain_scan: HeightScanConfig = field(default_factory=HeightScanConfig)
    # 终止/截断额外包含 terrain out-of-bounds。
    termination_config: RoughTerminationConfig = field(default_factory=RoughTerminationConfig)
    # 传感器在 flat 基础上增加 feet_vel 和 undesired_contact。
    sensor: RoughJoystickSensor = field(default_factory=RoughJoystickSensor)
    # rough reward 有更多足端、能耗、姿态和 gait 参数，必须由 owner YAML 注入。
    reward_config: RoughRewardConfig | None = None


```

```python
# src/unilab/envs/locomotion/go2/rough.py
@dataclass(kw_only=True)
class Go2RoughTerrainCfg(TerrainGeneratorCfg):
    # 每个 terrain cell 的物理尺寸；spawn/curriculum 会按 cell 放置环境。
    size: tuple[float, float] = (8.0, 8.0)
    # 6x6 地形网格让并行环境分布在不同 rough patch 上。
    num_rows: int = 6
    num_cols: int = 6
    # 外边界留空，减少采样点贴近 hfield 边缘。
    border_width: float = 1.0
    add_lights: bool = True
    # hfield 水平分辨率，影响 height scan 和碰撞地形细节。
    horizontal_scale: float = 0.2

    # 子地形比例定义训练会遇到的 rough 类型组合。
    sub_terrains: dict[str, SubTerrainCfg] = field(
        default_factory=lambda: {
            # flat 比例为 0，说明这个 cfg 主要训练非平整地形。
            "flat": flat(proportion=0.0),
            # 正/反台阶训练上台阶和下台阶。
            "pyramid_stairs": pyramid_stairs(
                proportion=0.1,
                step_height_range=(0.025, 0.10),
                step_width=0.4,
                platform_width=3.0,
                border_width=0.2,
            ),
            "pyramid_stairs_inv": pyramid_stairs_inv(
                proportion=0.1,
                step_height_range=(0.025, 0.10),
                step_width=0.4,
                platform_width=3.0,
                border_width=0.2,
            ),
            # 正/反坡训练身体俯仰变化下的速度跟踪。
            "hf_pyramid_slope": hf_pyramid_slope(
                proportion=0.2,
                slope_range=(0.0, 0.3),
                platform_width=2.0,
                border_width=0.2,
            ),
            "hf_pyramid_slope_inv": hf_pyramid_slope_inv(
                proportion=0.2,
                slope_range=(0.0, 0.3),
                platform_width=2.0,
                border_width=0.2,
            ),
            # 随机粗糙和波浪地形提供连续高度扰动。
            "random_rough": random_rough(
                proportion=0.3,
                noise_range=(0.01, 0.06),
                noise_step=0.01,
                border_width=0.2,
            ),
            "wave_terrain": wave_terrain(
                proportion=0.3,
                amplitude_range=(0.0, 0.12),
                num_waves=4,
                border_width=0.2,
            ),
        }
    )


```
### Reset：地形初态

| 本任务人工导读 |
| --- |
| reset 通过 spawn/terrain manager 选择初始 base 位置。 |
| ResetPlan 同时携带 qpos/qvel、命令和随机化 payload。 |

```python
# src/unilab/envs/locomotion/go2/rough.py
    def build_reset_plan(self, env: Any, env_ids: np.ndarray) -> ResetPlan:
        # 只为本次 reset 的 env 构造 qpos/qvel 和 info 更新。
        num_reset = len(env_ids)
        # 从 home keyframe 和初始速度复制出每个 env 的起点。
        qpos = np.tile(env._init_qpos, (num_reset, 1))
        qvel = np.tile(env._init_qvel, (num_reset, 1))
        # 在地形 cell 内随机平移 base，避免每次都从同一点开始。
        qpos[:, 0:2] += np.random.uniform(-0.5, 0.5, (num_reset, 2))
        # 抬高初始 z，防止随机姿态直接插入粗糙地形。
        qpos[:, 2] += np.random.uniform(0.25, 0.5, (num_reset,))
        # origins_for 把局部随机位移映射到当前 env 的 terrain cell。
        qpos[:, 0:3] += env._spawn.origins_for(env_ids)
        # rough reset 随机 base 姿态，让策略学习从更大姿态扰动恢复。
        roll = np.random.uniform(-3.14, 3.14, (num_reset,))
        pitch = np.random.uniform(-3.14, 3.14, (num_reset,))
        yaw = np.random.uniform(-3.14, 3.14, (num_reset,))
        # 使用库函数组合四元数，保持旋转表示一致。
        qpos[:, 3:7] = np_quat_mul(qpos[:, 3:7], np_quat_from_euler_xyz(roll, pitch, yaw))
        # base 线/角速度也随机扰动，提升重置后的恢复能力。
        qvel[:, 0:6] = np.asarray(
            np.random.uniform(-0.5, 0.5, size=(num_reset, 6)), dtype=get_global_dtype()
        )
        # 采样速度命令；heading 模式下 yaw command 会在 step 中由朝向反馈更新。
        commands = self._sample_commands(env, num_reset)
        info_updates: dict[str, Any] = {
            "commands": commands,
            # reset 后动作、加速度和力矩缓存清零，避免上一 episode 污染 reward/obs。
            "current_actions": zero_actions(num_reset, env._num_action),
            "last_actions": zero_actions(num_reset, env._num_action),
            "qacc": np.zeros((num_reset, env._num_action), dtype=get_global_dtype()),
            "torques": np.zeros((num_reset, env._num_action), dtype=get_global_dtype()),
        }
        # heading_command 额外保存目标朝向，后续 _update_commands 用它生成 yaw 速度。
        if env.cfg.commands.heading_command:
            info_updates["heading_commands"] = sample_heading_commands(env, num_reset)
        # 记录 episode 起点，用于 terrain curriculum / done 后统计。
        env._spawn.record_episode_start(env_ids, qpos[:, 0:3])
        return ResetPlan(
            env_ids=env_ids,
            qpos=qpos,
            qvel=qvel,
            info_updates=info_updates,
            # 通用 DR 在 reset 冷路径一次性构造，不进入 step 热路径。
            randomization=build_common_reset_randomization(env, num_reset),
        )


```
### Obs：height scan critic

| 本任务人工导读 |
| --- |
| policy obs 保持 45 维本体输入，critic 增加 base linvel 与 height scan。 |
| 高度扫描是 rough 稳定性的关键特权信息。 |

```python
# src/unilab/envs/locomotion/go2/rough.py
    @property
    def obs_groups_spec(self) -> dict[str, int]:
        # actor 观测是 45 维本体/命令输入；critic 追加线速度、姿态和 height scan。
        return {"obs": 45, "critic": 48 + self._height_scan_dim}

```

```python
# src/unilab/envs/locomotion/go2/rough.py
    def _compute_obs(
        self,
        info: dict,
        linvel: np.ndarray,
        gyro: np.ndarray,
        gravity: np.ndarray,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        feet_phase: np.ndarray,
    ) -> dict[str, np.ndarray]:
        # rough actor 不直接使用 gait phase；足端节律通过 reward/contact timer 约束。
        del feet_phase
        noise_cfg = self._cfg.noise_config
        # 关节角误差仍以 default_angles 为中心。
        diff = dof_pos - self.default_angles
        # actor gyro 和 dof_vel 被缩小，避免高频速度量主导策略输入。
        policy_gyro = self._obs_noise(gyro, noise_cfg.scale_gyro) * 0.25
        # actor 使用 -gravity 作为局部重力方向，表示身体相对世界的倾斜。
        policy_gravity = self._obs_noise(-gravity, noise_cfg.scale_gravity)
        policy_diff = self._obs_noise(diff, noise_cfg.scale_joint_angle)
        policy_dof_vel = self._obs_noise(dof_vel, noise_cfg.scale_joint_vel) * 0.05
        # rough 动作历史使用 current_actions，对应裁剪后的上一步策略输出。
        last_actions = info.get("current_actions", np.zeros_like(diff))
        # commands 是 x/y/yaw 速度目标；heading_command 时 yaw 由朝向误差反馈写回。
        commands = info["commands"]
        # actor 不看 height scan，只用本体状态和命令做可部署输入。
        obs = np.concatenate(
            [policy_gyro, policy_gravity, commands, policy_diff, policy_dof_vel, last_actions],
            axis=1,
            dtype=get_global_dtype(),
        )
        # critic_base 使用未缩放的速度/姿态/关节信息，作为训练期特权状态。
        critic_base = np.concatenate(
            [linvel, gyro, -gravity, commands, diff, dof_vel, last_actions],
            axis=1,
            dtype=get_global_dtype(),
        )
        # height_scan_obs 追加地形高度扫描，帮助 value 估计粗糙地形风险。
        critic = np.concatenate(
            [critic_base, height_scan_obs(self, self._cfg.terrain_scan, critic_base.shape[0])],
            axis=1,
            dtype=get_global_dtype(),
        )
        return {"obs": obs, "critic": critic}

```
### Action：裁剪后映射

| 本任务人工导读 |
| --- |
| 动作先按 clip_actions 裁剪，再缩放为关节目标。 |
| rough 不改变 actuator 顺序，只改变任务环境。 |

```python
# src/unilab/envs/locomotion/go2/rough.py
    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> np.ndarray:
        # 先裁剪策略输出，避免 rough 训练早期极端动作直接打到执行器目标。
        clipped_actions = np.asarray(
            np.clip(
                actions,
                -float(self._cfg.control_config.clip_actions),
                float(self._cfg.control_config.clip_actions),
            ),
            dtype=get_global_dtype(),
        )
        # 保存裁剪后的动作历史，obs 和 action_rate reward 都使用同一份动作。
        state.info["last_actions"] = state.info.get(
            "current_actions", np.zeros_like(clipped_actions)
        )
        state.info["current_actions"] = clipped_actions
        # 可选延迟模拟真实执行链路；默认用当前裁剪动作。
        exec_actions = (
            state.info["last_actions"]
            if self._cfg.control_config.simulate_action_latency
            else clipped_actions
        )
        # hip 用更小 scale，thigh/calf 用较大 scale，并按 actuator 顺序叠加默认角。
        return np.asarray(
            exec_actions * self._action_scale + self._default_angles_actuator,
            dtype=get_global_dtype(),
        )

```
### Reward/Termination

| 本任务人工导读 |
| --- |
| rough reward 对足端拖拽、姿态和地形高度更敏感。 |
| 终止要结合倾斜和地形相对高度判断跌倒。 |

```python
# src/unilab/envs/locomotion/go2/rough.py
    def _init_reward_functions(self):
        # 许多 rough reward 只应在身体基本直立时生效，避免摔倒状态刷分。
        scale_gravity = self._upright_scale  # local alias for lambda capture

        # gated 包装通用 reward，让 reward 按当前 upvector 姿态衰减。
        def gated(fn):
            return lambda ctx: fn(ctx) * scale_gravity(ctx.gravity)

        # joint_pos_penalty needs its three thresholds from the reward config
        def _joint_pos_penalty(ctx: RewardContext) -> np.ndarray:
            # joint_pos_penalty 在站立/低命令/低速度时加重，防止无命令乱动。
            cfg = self._reward_cfg
            return rewards.joint_pos_penalty(
                ctx,
                stand_still_scale=cfg.joint_pos_penalty_stand_still_scale,
                velocity_threshold=cfg.joint_pos_penalty_velocity_threshold,
                command_threshold=cfg.joint_pos_penalty_command_threshold,
            ) * scale_gravity(ctx.gravity)

        def _stand_still(ctx: RewardContext) -> np.ndarray:
            # 低命令时惩罚偏离默认站姿，形成“没命令就站稳”的行为。
            return rewards.stand_still(
                ctx, command_threshold=self._reward_cfg.stand_still_command_threshold
            ) * scale_gravity(ctx.gravity)

        # reward key 与 owner YAML reward.scales 对齐，未配置或 scale=0 的项不会贡献。
        self._reward_fns: dict[str, Any] = {
            # 主任务：粗糙地形上继续跟踪线速度和 yaw 速度。
            "tracking_lin_vel": gated(rewards.tracking_lin_vel),
            "tracking_ang_vel": gated(rewards.tracking_ang_vel),
            # 姿态/速度/能耗惩罚控制身体稳定和执行器负担。
            "lin_vel_z": gated(rewards.lin_vel_z),
            "ang_vel_xy": gated(rewards.ang_vel_xy),
            "dof_torques_l2": gated(rewards.dof_torques_l2),
            "joint_torques_l2": gated(rewards.dof_torques_l2),
            "dof_acc_l2": gated(rewards.dof_acc_l2),
            "joint_acc_l2": gated(rewards.dof_acc_l2),
            "joint_power": gated(rewards.joint_power),
            "stand_still": _stand_still,
            # Go2 姿态与关节构型项，减少髋关节漂移并鼓励对角腿镜像。
            "hip_pos": self._reward_hip_pos,
            "joint_pos_penalty": _joint_pos_penalty,
            "joint_mirror": self._reward_joint_mirror,
            # 动作平滑项。
            "action_rate": rewards.action_rate,
            "action_rate_l2": rewards.action_rate,
            # 接触力、非期望接触和足端滑移/离地时间是 rough 地形稳定性的核心约束。
            "undesired_contacts": self._reward_undesired_contacts,
            "contact_forces": self._reward_contact_forces,
            "feet_air_time": self._reward_feet_air_time,
            "feet_air_time_variance": self._reward_feet_air_time_variance,
            "feet_contact_without_cmd": self._reward_feet_contact_without_cmd,
            "feet_slide": self._reward_feet_slide,
            "feet_height_body": self._reward_feet_height_body,
            "feet_gait": self._reward_feet_gait,
            # upward 奖励维持机身朝上，和 gated reward 一起压制翻倒状态。
            "upward": rewards.upward,
        }

```

```python
# src/unilab/envs/locomotion/go2/rough.py
    def update_state(self, state: NpEnvState) -> NpEnvState:
        # 每步更新命令；heading_command 会根据目标朝向改写 yaw command。
        self._update_commands(state.info)
        # 维持与 flat 相同的对角 gait phase，供足端 gait/contact reward 使用。
        self.phase = np.fmod(self.phase + self._cfg.ctrl_dt * self.gait_frequency, 1.0)
        self.feet_phase[:, 0] = self.phase
        self.feet_phase[:, 3] = self.phase
        self.feet_phase[:, 1] = (self.phase + 0.5) % 1
        self.feet_phase[:, 2] = (self.phase + 0.5) % 1

        # 读取本帧 backend 传感器状态。
        linvel = self.get_local_linvel()
        gyro = self.get_gyro()
        gravity = self._backend.get_sensor_data("upvector")
        dof_pos = self.get_dof_pos()
        dof_vel = self.get_dof_vel()
        # 足端力、位置和速度同时刷新，供接触计时、滑移和高度 reward 使用。
        self.feet_force[:, :, :] = 0
        for i in range(len(self._cfg.sensor.feet_force)):
            self.feet_force[:, i, :] = self._backend.get_sensor_data(self._cfg.sensor.feet_force[i])
        for i in range(len(self._cfg.sensor.feet_pos)):
            self.feet_pos[:, i, :] = self._backend.get_sensor_data(self._cfg.sensor.feet_pos[i])
        for i in range(len(self._cfg.sensor.feet_vel)):
            self.feet_vel[:, i, :] = self._backend.get_sensor_data(self._cfg.sensor.feet_vel[i])
        # 根据当前接触更新 air/contact time，feet_air_time 和 feet_gait 依赖这些缓存。
        self._update_contact_timers(self._foot_contact_mask())
        # 估计 qacc 和 PD torque，供 energy/torque/acc reward 使用。
        state.info["qacc"] = self._estimate_dof_acc(dof_vel)
        state.info["torques"] = self._estimate_pd_torques(state.info, dof_pos, dof_vel)
        # 当前实现不因姿态 terminated，而通过 truncated 处理越界/超时。
        terminated = self._compute_terminated(gravity)
        # reward 使用 rough 专用上下文：地形相对高度、姿态、速度、力矩等。
        reward = self._compute_rough_reward(state.info, linvel, gyro, gravity, dof_pos, dof_vel)
        obs = self._compute_obs(
            state.info, linvel, gyro, gravity, dof_pos, dof_vel, self.feet_phase
        )
        state = state.replace(obs=obs, reward=reward, terminated=terminated)
        # done 后更新 terrain spawn 统计，供 curriculum/log 使用。
        done = state.terminated | state.truncated
        if np.any(done):
            done_indices = np.where(done)[0]
            stats = self._spawn.update_on_done(
                done_indices, self._backend.get_base_pos()[done_indices]
            )
            if stats:
                if "log" not in state.info:
                    state.info["log"] = {}
                for k, v in stats.items():
                    state.info["log"][f"terrain_curriculum/{k}"] = float(v)
        return state

```



## Agent

Agent 仍输出 12 维腿部目标；算法层不直接感知 backend 私有能力，而是通过 env obs 接收地形信息。

训练入口通过 owner YAML 决定注册 env、后端身份和算法参数；CLI 应使用 `--sim` 选择 backend owner，而不是单独 override `training.sim_backend`。

```yaml
# conf/ppo/task/go2_joystick_rough/mujoco.yaml
training:
  task_name: Go2JoystickRough
  sim_backend: mujoco
  play_steps: 500
  play_env_num: 16
  render_spacing: 0.0
  cam_tracking: true
  cam_tracking_env_idx: 0
  cam_tracking_extra_envs: 9
algo:
  obs_groups:
    actor:
    - actor
    critic:
    - critic
  num_envs: 4096
  max_iterations: 1500
```

## Env

Env 使用 rough 任务配置、terrain spawn 和 locomotion task fragment。

Env 必须遵守 `NpEnv` contract：`reset()` 返回 `(obs_dict, info_dict)`，`obs` 是 dict，`obs_groups_spec` 决定 learner 使用哪些观测组。

```python
# src/unilab/envs/locomotion/go2/rough.py
            ),
        }
    )


@registry.envcfg("Go2JoystickRough")
@dataclass
class Go2JoystickRoughCfg(Go2JoystickCfg):
    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(
            model_file=str(ASSETS_ROOT_PATH / "robots" / "go2" / "go2.xml"),
```

## Obs

观测在基础 Go2 观测外加入高度扫描；critic 通常保留更多特权地形/速度信息。

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
# conf/ppo/task/go2_joystick_rough/mujoco.yaml
algo:
  obs_groups:
    actor:
    - actor
    critic:
    - critic

```

代码侧观测通常在 `_get_obs`、`_build_obs`、`obs_groups_spec` 或 base env 中组装：

```python
# src/unilab/envs/locomotion/go2/rough.py
        self._last_air_time = np.zeros((num_envs, len(cfg.sensor.feet_force)), dtype=np.float32)
        self._last_contact_time = np.zeros((num_envs, len(cfg.sensor.feet_force)), dtype=np.float32)
        self._first_foot_contact = np.zeros((num_envs, len(cfg.sensor.feet_force)), dtype=bool)
        init_height_scan_sensor(self, cfg.terrain_scan, cfg.asset.base_name)

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        return {"obs": 45, "critic": 48 + self._height_scan_dim}

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

动作空间不变，rough 任务只改变观测、场景和奖励约束。

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
| action_scale | 0.25 | 策略输出到目标关节角的缩放系数。 |
| clip_actions | 100.0 | 动作裁剪范围，防止策略输出过大。 |
| hip_action_scale | 0.125 | 策略输出到目标关节角的缩放系数。 |
| non_hip_action_scale | 0.25 | 策略输出到目标关节角的缩放系数。 |

```yaml
# conf/ppo/task/go2_joystick_rough/mujoco.yaml
env:
  control_config:
    action_scale: 0.25
    hip_action_scale: 0.125
    non_hip_action_scale: 0.25
    clip_actions: 100.0

```

动作到执行器的实际映射由 backend 的 actuator control range 和 env 的 `apply_action`/控制函数完成：

```python
# src/unilab/envs/locomotion/go2/rough.py
            "feet_slide": self._reward_feet_slide,
            "feet_height_body": self._reward_feet_height_body,
            "feet_gait": self._reward_feet_gait,
            "upward": rewards.upward,
        }

    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> np.ndarray:
        clipped_actions = np.asarray(
            np.clip(
                actions,
                -float(self._cfg.control_config.clip_actions),
                float(self._cfg.control_config.clip_actions),
            ),
```

源码锚点拆解：

| 源码符号 | 中文说明 |
| --- | --- |
| apply_action | 把策略输出缩放为执行器控制目标。 |

## Reward

reward 增加 foot_drag、接触与粗糙地形下的姿态稳定惩罚。

### Reward 逐项拆解

reward scale 以 owner YAML 的 `reward.scales` 为真源；源码中的 `RewardConfig` 描述可用字段，训练脚本只做 Hydra 组装和注入。正数通常表示奖励，负数通常表示惩罚，0 表示该项在当前 owner 中关闭或仅保留接口。

| Reward key | Scale | 方向 | 中文含义 |
| --- | --- | --- | --- |
| lin_vel_z | -2.0 | 惩罚 | 竖直速度惩罚，限制 base 上下弹跳。 |
| ang_vel_xy | -0.05 | 惩罚 | 横滚/俯仰角速度惩罚，抑制身体剧烈摇摆。 |
| joint_torques_l2 | -2.5e-05 | 惩罚 | 力矩惩罚，降低控制输出过大。 |
| joint_acc_l2 | -2.5e-07 | 惩罚 | 仓库 owner YAML 中声明的奖励项；具体实现见本页源码片段和 env 的 reward dispatch。 |
| joint_power | -2e-05 | 惩罚 | 仓库 owner YAML 中声明的奖励项；具体实现见本页源码片段和 env 的 reward dispatch。 |
| stand_still | -2.0 | 惩罚 | 仓库 owner YAML 中声明的奖励项；具体实现见本页源码片段和 env 的 reward dispatch。 |
| hip_pos | -0.5 | 惩罚 | 位置相关奖励/惩罚，用于跟踪目标位置或避免偏离安全区域。 |
| joint_pos_penalty | -1.0 | 惩罚 | 位置相关奖励/惩罚，用于跟踪目标位置或避免偏离安全区域。 |
| joint_mirror | -0.05 | 惩罚 | 仓库 owner YAML 中声明的奖励项；具体实现见本页源码片段和 env 的 reward dispatch。 |
| action_rate | -0.01 | 惩罚 | 动作变化惩罚，抑制相邻控制步之间的目标突变。 |
| undesired_contacts | -1.0 | 惩罚 | 非期望接触惩罚，例如手、膝、身体等部位异常触地。 |
| contact_forces | -0.00015 | 惩罚 | 接触项，约束指定足端或身体部位的接触模式。 |
| tracking_lin_vel | 3.0 | 奖励 | 线速度命令跟踪奖励，鼓励机器人实际平面速度贴近 joystick/命令速度。 |
| tracking_ang_vel | 1.5 | 奖励 | 角速度命令跟踪奖励，鼓励 yaw 角速度贴近转向命令。 |
| feet_air_time | 0.5 | 奖励 | 仓库 owner YAML 中声明的奖励项；具体实现见本页源码片段和 env 的 reward dispatch。 |
| feet_air_time_variance | -1.0 | 惩罚 | 仓库 owner YAML 中声明的奖励项；具体实现见本页源码片段和 env 的 reward dispatch。 |
| feet_contact_without_cmd | 0.1 | 奖励 | 接触项，约束指定足端或身体部位的接触模式。 |
| feet_slide | -0.1 | 惩罚 | 仓库 owner YAML 中声明的奖励项；具体实现见本页源码片段和 env 的 reward dispatch。 |
| feet_height_body | -5.0 | 惩罚 | 目标高度奖励，用于 footstand/handstand 等姿态任务。 |
| feet_gait | 0.5 | 奖励 | 仓库 owner YAML 中声明的奖励项；具体实现见本页源码片段和 env 的 reward dispatch。 |
| upward | 1.0 | 奖励 | 仓库 owner YAML 中声明的奖励项；具体实现见本页源码片段和 env 的 reward dispatch。 |

```yaml
# conf/ppo/task/go2_joystick_rough/mujoco.yaml
reward:
  scales:
    lin_vel_z: -2.0
    ang_vel_xy: -0.05
    joint_torques_l2: -2.5e-05
    joint_acc_l2: -2.5e-07
    joint_power: -2.0e-05
    stand_still: -2.0
    hip_pos: -0.5
    joint_pos_penalty: -1.0
    joint_mirror: -0.05
    action_rate: -0.01
    undesired_contacts: -1.0
    contact_forces: -0.00015
    tracking_lin_vel: 3.0
    tracking_ang_vel: 1.5
    feet_air_time: 0.5
    feet_air_time_variance: -1.0
    feet_contact_without_cmd: 0.1
    feet_slide: -0.1
    feet_height_body: -5.0
    feet_gait: 0.5
    upward: 1.0
  tracking_sigma: 0.25
  base_height_target: 0.3
  stand_still_command_threshold: 0.1
  joint_pos_penalty_stand_still_scale: 5.0
  joint_pos_penalty_velocity_threshold: 0.5
  joint_pos_penalty_command_threshold: 0.1
  contact_threshold: 1.0
  contact_forces_threshold: 100.0
  feet_air_time_threshold: 0.5
  feet_height_body_target: -0.2
  feet_height_body_tanh_mult: 2.0
  feet_gait_std: 0.7071067811865476
  feet_gait_max_err: 0.2
  feet_gait_velocity_threshold: 0.5
  feet_gait_command_threshold: 0.1

```

源码中的 reward 入口：

```python
# src/unilab/envs/locomotion/go2/rough.py
from unilab.envs.locomotion.go2.joystick import (
    Commands,
    Go2JoystickCfg,
    Go2JoystickDomainRandomizationProvider,
    Go2WalkTask,
    JoystickSensor,
    RewardConfig,
)
from unilab.terrains import (
    SubTerrainCfg,
    TerrainGeneratorCfg,
    flat,
    hf_pyramid_slope,
```

源码锚点拆解：

| 源码符号 | 中文说明 |
| --- | --- |
| RoughRewardConfig | 声明 reward 可用参数和默认 scale。 |
| _init_reward_functions | 把 reward 名称映射到源码函数，owner YAML 的 scale 会按这些 key 分发。 |
| _reward_base_height_values | 单个 reward 项的实现函数，通常返回每个 env 的 reward 向量。 |
| _reward_hip_pos | 单个 reward 项的实现函数，通常返回每个 env 的 reward 向量。 |
| _reward_joint_mirror | 单个 reward 项的实现函数，通常返回每个 env 的 reward 向量。 |
| _reward_undesired_contacts | 单个 reward 项的实现函数，通常返回每个 env 的 reward 向量。 |
| _reward_contact_forces | 单个 reward 项的实现函数，通常返回每个 env 的 reward 向量。 |
| _reward_feet_air_time | 单个 reward 项的实现函数，通常返回每个 env 的 reward 向量。 |
| _reward_feet_air_time_variance | 单个 reward 项的实现函数，通常返回每个 env 的 reward 向量。 |
| _reward_feet_contact_without_cmd | 单个 reward 项的实现函数，通常返回每个 env 的 reward 向量。 |
| _reward_feet_slide | 单个 reward 项的实现函数，通常返回每个 env 的 reward 向量。 |
| _reward_feet_height_body | 单个 reward 项的实现函数，通常返回每个 env 的 reward 向量。 |

## 初始状态

reset 在地形上采样 spawn 点，并应用 `home` keyframe 的默认姿态。

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
# src/unilab/envs/locomotion/go2/rough.py

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.np_env import NpEnvState
from unilab.base.scene import SceneCfg, TerrainSceneCfg
from unilab.dr import DomainRandomizationManager, ResetPlan
from unilab.dr.dr_utils import build_common_reset_randomization, zero_actions
from unilab.dtype_config import get_global_dtype
from unilab.envs.common.rotation import (
    np_quat_apply_inverse,
    np_quat_from_euler_xyz,
    np_quat_mul,
)
```

## 终止条件

终止关注地形相对 base 高度、过大倾斜和异常接触。

终止条件通常由 env 源码中的 `_compute_termination(s)`、reward config 阈值和 `max_episode_seconds` 共同决定。

```yaml
# conf/ppo/task/go2_joystick_rough/mujoco.yaml
env:
  {}

reward:
  {}

```

```python
# src/unilab/envs/locomotion/go2/rough.py
            self.feet_force[:, i, :] = self._backend.get_sensor_data(self._cfg.sensor.feet_force[i])
        for i in range(len(self._cfg.sensor.feet_pos)):
            self.feet_pos[:, i, :] = self._backend.get_sensor_data(self._cfg.sensor.feet_pos[i])
        for i in range(len(self._cfg.sensor.feet_vel)):
            self.feet_vel[:, i, :] = self._backend.get_sensor_data(self._cfg.sensor.feet_vel[i])
        self._update_contact_timers(self._foot_contact_mask())
        state.info["qacc"] = self._estimate_dof_acc(dof_vel)
        state.info["torques"] = self._estimate_pd_torques(state.info, dof_pos, dof_vel)
        terminated = self._compute_terminated(gravity)
        reward = self._compute_rough_reward(state.info, linvel, gyro, gravity, dof_pos, dof_vel)
        obs = self._compute_obs(
            state.info, linvel, gyro, gravity, dof_pos, dof_vel, self.feet_phase
        )
        state = state.replace(obs=obs, reward=reward, terminated=terminated)
        done = state.terminated | state.truncated
        if np.any(done):
            done_indices = np.where(done)[0]
```

## 域随机化

域随机化覆盖 terrain、摩擦、质量、COM、推力与控制参数。

域随机化只在 reset/interval 等冷路径触发，不在 step 热路径解析资产或探测 backend 私有能力。

### 域随机化字段拆解

| 配置字段 | 当前值 | 中文说明 |
| --- | --- | --- |
| added_mass_range | [-1.0, 3.0] | 随机采样范围或控制范围。 |
| kd_multiplier_range | [0.5, 2.0] | 随机采样范围或控制范围。 |
| kp_multiplier_range | [0.5, 2.0] | 随机采样范围或控制范围。 |
| max_force | [1.0, 1.0, 0.5] | 任务 owner YAML 中的配置字段。 |
| push_interval | 625 | 任务 owner YAML 中的配置字段。 |
| push_robots | True | 布尔开关。 |
| random_com | True | 布尔开关。 |
| randomize_base_mass | True | 域随机化开关。 |
| randomize_kd | True | 域随机化开关。 |
| randomize_kp | True | 域随机化开关。 |

```yaml
# conf/ppo/task/go2_joystick_rough/mujoco.yaml
env:
  domain_rand:
    randomize_base_mass: true
    added_mass_range:
    - -1.0
    - 3.0
    random_com: true
    randomize_kp: true
    kp_multiplier_range:
    - 0.5
    - 2.0
    randomize_kd: true
    kd_multiplier_range:
    - 0.5
    - 2.0
    push_robots: true
    push_interval: 625
    max_force:
    - 1.0
    - 1.0
    - 0.5

```

```python
# src/unilab/envs/locomotion/go2/rough.py
import numpy as np

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.np_env import NpEnvState
from unilab.base.scene import SceneCfg, TerrainSceneCfg
from unilab.dr import DomainRandomizationManager, ResetPlan
from unilab.dr.dr_utils import build_common_reset_randomization, zero_actions
from unilab.dtype_config import get_global_dtype
from unilab.envs.common.rotation import (
    np_quat_apply_inverse,
    np_quat_from_euler_xyz,
    np_quat_mul,
```

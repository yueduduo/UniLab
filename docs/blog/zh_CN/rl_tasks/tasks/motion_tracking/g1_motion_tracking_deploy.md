---
title: "G1 Motion Tracking Deploy"
slug: "g1_motion_tracking_deploy"
category: "motion_tracking"
robot: "g1"
registered_envs: [G1MotionTrackingDeploy]
---

# G1 Motion Tracking Deploy

## 任务目标

为部署/评估口径整理 G1 motion tracking 观测，使策略输入更接近实际部署所需信息。

![G1 Motion Tracking Deploy 场景渲染](../../images/tasks/g1_motion_tracking_deploy.png)

> 图片由 `_tools/render_static_images.py` 使用 `src/unilab/assets/robots/g1/scene_flat.xml` 的 MuJoCo scene 离屏渲染生成；如果当前环境无法启用 EGL/OSMesa，生成 manifest 会记录失败原因。

## 证据索引

| 项目 | 内容 |
| --- | --- |
| 任务类别 | 动作追踪 |
| 机器人 | g1 |
| 代表 scene | src/unilab/assets/robots/g1/scene_flat.xml |
| 代表源码 | src/unilab/envs/motion_tracking/g1/tracking.py |
| 代表 owner YAML | conf/ppo/task/g1_motion_tracking_deploy/mujoco.yaml |
| 动作维度 | 29 |

### Registry 与 Owner

| 注册名 | 后端 | 配置类 | 配置源码 |
| --- | --- | --- | --- |
| G1MotionTrackingDeploy | motrix, mujoco | G1MotionTrackingDeployEnvCfg | src/unilab/envs/motion_tracking/g1/tracking.py |

| 算法组 | 后端 | training.task_name | owner YAML |
| --- | --- | --- | --- |
| ppo | motrix | G1MotionTrackingDeploy | conf/ppo/task/g1_motion_tracking_deploy/motrix.yaml |
| ppo | mujoco | G1MotionTrackingDeploy | conf/ppo/task/g1_motion_tracking_deploy/mujoco.yaml |

## 代码级执行流

这部分不再用通用关键词套模板，而是为 `g1_motion_tracking_deploy` 单独写源码导读。源码片段只负责提供证据；每节说明都围绕本任务自己的配置、状态、观测、动作和 reward 语义展开。

| 阶段 | 证据位置 | 代码级说明 |
| --- | --- | --- |
| 1. Hydra owner | conf/ppo/task/g1_motion_tracking_deploy/mujoco.yaml | 训练命令先 compose owner YAML。这里确定 `training.task_name`、后端身份、obs_groups、reward scale 和 env override。 |
| 2. Registry 构造 Env | src/unilab/envs/motion_tracking/g1/tracking.py | `@registry.envcfg` 注册配置类，`@registry.env` 注册后端实现类；训练脚本只调用 registry，不直接 new 任务类。 |
| 3. Reset/初始状态 | src/unilab/assets/robots/g1/scene_flat.xml | env 从 scene keyframe 或 motion/grasp cache 取初态，再通过 reset randomization 改写 qpos/qvel/info。 |
| 4. Obs dict | src/unilab/envs/motion_tracking/g1/tracking.py | env 读取 backend 传感器和 info，返回 dict；learner 根据 `obs_groups_spec` 和 owner `algo.obs_groups` 选择 actor/critic 输入。 |
| 5. Action 到控制目标 | src/unilab/envs/motion_tracking/g1/tracking.py | policy 输出先按 action scale / 控制顺序映射到 actuator，再由 backend step 推进仿真。 |
| 6. Reward dispatch | src/unilab/envs/motion_tracking/g1/tracking.py | 源码把 reward key 映射到函数，owner YAML 的 scale 决定每项奖励/惩罚是否启用以及权重。 |
| 7. Termination | src/unilab/envs/motion_tracking/g1/tracking.py | env 根据高度、姿态、接触、参考轨迹误差、物体状态或能量阈值生成 done/terminated。 |

### 本任务阅读重点

| 阅读重点 |
| --- |
| Deploy 任务和普通 tracking 共享运动跟踪目标，但重点是部署可用观测组织。 |
| 应读 `G1MotionTrackingDeployEnv`，看它如何覆盖/整理 actor 输入，而不是重复普通 tracking 文案。 |
| reward 和 reset 可沿用 tracking，但文档必须说明 deploy 的差异点在 obs contract。 |

### 本任务注册入口

| 注册名 | 配置类 | 可用后端 |
| --- | --- | --- |
| G1MotionTrackingDeploy | G1MotionTrackingDeployEnvCfg | motrix, mujoco |

### 本任务源码锚点

| 小节 | 源码文件 | 明确锚点 |
| --- | --- | --- |
| 配置：Deploy 注册 | src/unilab/envs/motion_tracking/g1/tracking.py | G1MotionTrackingDeployEnvCfg, G1MotionTrackingDeploy |
| Reset：沿用 motion reference | src/unilab/envs/motion_tracking/g1/tracking.py | build_reset_plan, motion_loader.get_motion_at_frame |
| Obs：部署 actor 输入 | src/unilab/envs/motion_tracking/g1/tracking.py | G1MotionTrackingDeployEnv, _compute_obs |
| Action：29 DoF tracking | src/unilab/envs/motion_tracking/g1/tracking.py | apply_action, simulate_action_latency |
| Reward/Termination | src/unilab/envs/motion_tracking/g1/tracking.py | _init_reward_functions, update_state |

## 关键源码逐段解释（按本任务手写）

### 配置：Deploy 注册

| 本任务人工导读 |
| --- |
| deploy cfg 单独注册，便于 owner YAML 使用部署口径。 |
| 它不改变任务目标，只改变环境暴露的观测组织。 |

```python
# src/unilab/envs/motion_tracking/g1/tracking.py
# deploy 任务使用独立注册名，owner YAML 可以明确选择部署口径。
@registry.envcfg("G1MotionTrackingDeploy")
@dataclass
class G1MotionTrackingDeployEnvCfg(G1MotionTrackingCfg):
    """Registered deploy configuration for G1 motion tracking."""

    # 不新增配置字段，仍复用 G1MotionTrackingCfg 的 motion/reward/reset 定义。
    pass


```
### Reset：沿用 motion reference

| 本任务人工导读 |
| --- |
| reset 仍从 motion clip 采样参考初态。 |
| domain randomization 和普通 tracking 一致，由 owner YAML 控制启用项。 |

```python
# src/unilab/envs/motion_tracking/g1/tracking.py
    def build_reset_plan(self, env: Any, env_ids: np.ndarray) -> ResetPlan:
        num_reset = len(env_ids)
        # deploy 版本仍从 motion sampler 采参考帧，不改变任务目标来源。
        motion_frames = env.motion_sampler.sample_frames(env_ids)
        # motion_data 提供 root/body/joint 的参考状态，后续 qpos/qvel 直接对齐它。
        motion_data = env.motion_loader.get_motion_at_frame(motion_frames)
        qpos, qvel = _build_motion_reference_state(env, env_ids, motion_data)

        # reset 后动作历史清零，保证部署 actor 的 last_actions 从确定状态开始。
        info_updates = {
            "current_actions": zero_actions(num_reset, env._num_action),
            "last_actions": zero_actions(num_reset, env._num_action),
        }
        # 随机化仍通过 ResetPlan 下发给 backend，不在训练脚本里分支处理。
        randomization = build_common_reset_randomization(
            env, num_reset, base_kp=self._base_kp, base_kd=self._base_kd
        )

        dr_cfg = env.cfg.domain_rand
        if getattr(dr_cfg, "randomize_geom_friction", False):
            assert self._base_geom_friction is not None
            assert self._foot_geom_ids is not None
            payload = randomization or ResetRandomizationPayload()
            low, high = dr_cfg.friction_range
            # 每个 env 采样一个脚底摩擦缩放，模拟部署场景中地面差异。
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

        if getattr(dr_cfg, "randomize_joint_default_pos", False):
            low, high = dr_cfg.joint_default_pos_range
            # 关节默认角偏置写入 info，obs/action 映射会共同读取，保持口径一致。
            info_updates["default_dof_pos_bias"] = np.random.uniform(
                low, high, size=(num_reset, env._num_action)
            ).astype(get_global_dtype())

        # 返回的 ResetPlan 是 reset 冷路径唯一出口，状态和 DR payload 一并生效。
        return ResetPlan(
            env_ids=env_ids,
            qpos=qpos,
            qvel=qvel,
            info_updates=info_updates,
            randomization=randomization,
        )

```
### Obs：部署 actor 输入

| 本任务人工导读 |
| --- |
| deploy env 重点调整 actor obs，减少训练期不可部署的特权项。 |
| critic/训练辅助信息仍由 env contract 管理。 |

```python
# src/unilab/envs/motion_tracking/g1/tracking.py
@registry.envcfg("G1MotionTrackingDeploy")
@dataclass
class G1MotionTrackingDeployEnvCfg(G1MotionTrackingCfg):
    """Registered deploy configuration for G1 motion tracking."""

    # 部署注册类本身不改 reward/action，只让配置层选择 deploy 任务名。
    pass


```

```python
# src/unilab/envs/motion_tracking/g1/tracking.py
    def _compute_obs(
        self,
        info: dict,
        motion_data,
        linvel: np.ndarray,
        gyro: np.ndarray,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        robot_body_pos_w: np.ndarray,
        robot_body_quat_w: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Compute observations as dict with actor and critic groups."""
        num_envs = linvel.shape[0]
        dtype = get_global_dtype()
        n_action = dof_pos.shape[1]
        # body 数量决定 critic 特权 body transform 的规模。
        n_body = self._n_motion_bodies

        # Get anchor states
        # deploy actor 关注可部署输入，critic 仍可使用参考/机器人 anchor 对齐信息。
        anchor_pos_w = motion_data.body_pos_w[:, self.anchor_body_idx]
        anchor_quat_w = motion_data.body_quat_w[:, self.anchor_body_idx]
        robot_anchor_pos_w = robot_body_pos_w[:, self.anchor_body_idx]
        robot_anchor_quat_w = robot_body_quat_w[:, self.anchor_body_idx]

        # Motion anchor pose in robot frame
        if num_envs == self._num_envs:
            # 常规 step 复用缓存，避免部署观测热路径反复分配。
            motion_anchor_pos_b = self._motion_anchor_pos_b
            motion_anchor_ori_b = self._motion_anchor_ori_b
            joint_pos_rel = self._joint_pos_rel
            zero_actions = self._zero_actions
        else:
            motion_anchor_pos_b = np.empty((num_envs, 3), dtype=dtype)
            motion_anchor_ori_b = np.empty((num_envs, 6), dtype=dtype)
            joint_pos_rel = np.empty((num_envs, n_action), dtype=dtype)
            zero_actions = np.zeros((num_envs, n_action), dtype=dtype)
        _write_motion_anchor_transform(
            robot_anchor_pos_w,
            robot_anchor_quat_w,
            anchor_pos_w,
            anchor_quat_w,
            motion_anchor_pos_b,
            motion_anchor_ori_b,
        )

        # Joint positions and velocities
        bias = info.get("default_dof_pos_bias")
        # 相对关节角扣除默认角偏置，和 apply_action 的控制基准保持一致。
        effective_default = self.default_angles + bias if bias is not None else self.default_angles
        np.subtract(dof_pos, effective_default, out=joint_pos_rel)
        last_actions = info.get("current_actions")
        if not isinstance(last_actions, np.ndarray):
            last_actions = zero_actions

        if num_envs == self._num_envs:
            command = self._motion_command
        else:
            command = np.empty((num_envs, n_action * 2), dtype=dtype)
        # command 是参考关节角/速度，deploy 策略仍通过它知道当前应跟踪的 motion 阶段。
        command[:, :n_action] = motion_data.joint_pos
        command[:, n_action : n_action * 2] = motion_data.joint_vel

        # Per-step observation noise on sensor channels (actor only).
        # Critic uses the clean originals — asymmetric actor–critic contract.
        noise_cfg = self._cfg.noise_config
        noise_enabled = noise_cfg.level > 0.0
        if noise_enabled:
            # 噪声仅作用于 actor 的传感通道，训练 critic 不被部署噪声污染。
            linvel_actor = self._obs_noise(linvel, noise_cfg.scale_linvel)
            gyro_actor = self._obs_noise(gyro, noise_cfg.scale_gyro)
            joint_pos_actor = self._obs_noise(joint_pos_rel, noise_cfg.scale_joint_angle)
            dof_vel_actor = self._obs_noise(dof_vel, noise_cfg.scale_joint_vel)
        else:
            linvel_actor = linvel
            gyro_actor = gyro
            joint_pos_actor = joint_pos_rel
            dof_vel_actor = dof_vel

        # Actor observations (noisy proprioception)
        # actor_obs 是部署导出的主要输入，必须和部署端 ObservationManager 口径对齐。
        actor_obs = self._build_actor_obs(
            command=command,
            motion_anchor_pos_b=motion_anchor_pos_b,
            motion_anchor_ori_b=motion_anchor_ori_b,
            noisy_linvel=linvel_actor,
            noisy_gyro=gyro_actor,
            noisy_joint_pos_rel=joint_pos_actor,
            noisy_dof_vel=dof_vel_actor,
            last_actions=last_actions,
        )

        # Critic observations (clean proprioception + privileged body transforms)
        # critic_obs 保留干净 proprioception 和 body 特权信息，只服务训练期价值函数。
        critic_obs = np.empty((num_envs, self._critic_obs_width), dtype=dtype)
        offset = 0
        critic_obs[:, offset : offset + command.shape[1]] = command
        offset += command.shape[1]
```
### Action：29 DoF tracking

| 本任务人工导读 |
| --- |
| 动作口径与普通 tracking 一致。 |
| 部署任务不引入新的 actuator 或训练脚本分支。 |

```python
# src/unilab/envs/motion_tracking/g1/tracking.py
    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> np.ndarray:
        # deploy 任务不新增 actuator，仍记录 29 维动作历史供 latency/obs/reward 使用。
        state.info["last_actions"] = state.info.get("current_actions", np.zeros_like(actions))
        state.info["current_actions"] = actions
        # simulate_action_latency=True 时下发上一帧动作，贴近部署控制链路延迟。
        exec_actions = (
            state.info["last_actions"]
            if self._cfg.control_config.simulate_action_latency
            else actions
        )
        bias = state.info.get("default_dof_pos_bias")
        # reset 随机化的默认角偏置会进入控制目标基准，避免 obs/action 口径错位。
        base = self.default_angles + bias if bias is not None else self.default_angles
        ctrl: np.ndarray = exec_actions * self._cfg.control_config.action_scale + base
        return ctrl

```
### Reward/Termination

| 本任务人工导读 |
| --- |
| reward 和 termination 继承 tracking 主逻辑。 |
| 文档中应把 deploy 差异写在 obs，而不是虚构新的 reward。 |

```python
# src/unilab/envs/motion_tracking/g1/tracking.py
    def _init_reward_functions(self):
        # deploy 版本复用 tracking reward 表，不为部署观测另造 reward。
        self._reward_fns = {
            # root/body/joint imitation 项衡量机器人与参考 motion 的贴合程度。
            "motion_global_root_pos": self._reward_motion_global_root_pos,
            "motion_global_root_ori": self._reward_motion_global_root_ori,
            "motion_body_pos": self._reward_motion_body_pos,
            "motion_body_ori": self._reward_motion_body_ori,
            "motion_body_lin_vel": self._reward_motion_body_lin_vel,
            "motion_body_ang_vel": self._reward_motion_body_ang_vel,
            "motion_ee_body_pos_z": self._reward_motion_ee_body_pos_z,
            "motion_joint_pos": self._reward_motion_joint_pos,
            "motion_joint_vel": self._reward_motion_joint_vel,
            # 控制平滑、关节限位和异常接触负责部署稳定性。
            "action_rate_l2": self._reward_action_rate_l2,
            "joint_limit": self._reward_joint_limit,
            "undesired_contacts": self._reward_undesired_contacts,
        }

```

```python
# src/unilab/envs/motion_tracking/g1/tracking.py
    def update_state(self, state: NpEnvState) -> NpEnvState:
        # 每步先清空 clip 结束标记，避免上一帧截断状态残留。
        self._clip_end_truncated.fill(False)

        # Get current motion data
        # motion_data 是 reward、termination 和 obs 的同一份参考真值。
        motion_data = self._get_current_motion()

        # Get robot state
        # backend 读取的本体/关节状态构成 deploy actor 的可观测输入。
        linvel = self.get_local_linvel()
        gyro = self.get_gyro()
        dof_pos = self.get_dof_pos()
        dof_vel = self.get_dof_vel()

        # Get body states
        (
            robot_body_pos_w,
            robot_body_quat_w,
            robot_body_lin_vel_w,
            robot_body_ang_vel_w,
        ) = self._get_body_state_w()

        # Compute relative body transforms (for observations and rewards)
        self._update_relative_transforms(motion_data, robot_body_pos_w, robot_body_quat_w)

        # Compute terminations
        # termination 仍使用 tracking 失败边界；deploy 差异不改变失败定义。
        terminated = self._compute_terminations(motion_data, robot_body_pos_w, robot_body_quat_w)

        # Update failure statistics for adaptive sampling
        # 失败统计继续反馈给 adaptive sampler，帮助训练覆盖困难参考帧。
        self.motion_sampler.update_failure_stats(terminated)

        # Compute reward
        # reward 沿用普通 tracking，对 root/body/joint 等误差逐项加权。
        reward = self._compute_reward(
            state.info,
            motion_data,
            robot_body_pos_w,
            robot_body_quat_w,
            robot_body_lin_vel_w,
            robot_body_ang_vel_w,
            dof_pos,
            dof_vel,
        )

        # Compute observations
        # obs dict 输出 actor/critic 组，部署任务的关键差异就在这里体现。
        obs = self._compute_obs(
            state.info,
            motion_data,
            linvel,
            gyro,
            dof_pos,
            dof_vel,
            robot_body_pos_w,
            robot_body_quat_w,
        )

        # Advance motion frames
        done_env_ids = self.motion_sampler.step()
        if len(done_env_ids) > 0:
            if self._cfg.truncate_on_clip_end:
                self._clip_end_truncated[done_env_ids] = True
            else:
                # Match BeyondMimic: clip boundaries are command resampling points, not
                # episode boundaries; sync the simulated robot to the new reference.
                # clip 到尾时若 episode 未失败，就同步到下一段参考而不是强制结束。
                resample_env_ids = done_env_ids[~terminated[done_env_ids]]
                if len(resample_env_ids) > 0:
                    self._resample_reference_state(resample_env_ids)
                    self._refresh_observation_rows(obs, state.info, resample_env_ids)

        return state.replace(obs=obs, reward=reward, terminated=terminated)

```



## Agent

Agent 输出 29 维动作，但 deploy env 使用更收敛的 actor 观测组织。

训练入口通过 owner YAML 决定注册 env、后端身份和算法参数；CLI 应使用 `--sim` 选择 backend owner，而不是单独 override `training.sim_backend`。

```yaml
# conf/ppo/task/g1_motion_tracking_deploy/mujoco.yaml
training:
  task_name: G1MotionTrackingDeploy
  sim_backend: mujoco
  play_steps: 1000
algo:
  obs_groups:
    actor:
    - actor
  num_envs: 1024
  max_iterations: 15000
```

## Env

Env 继承 motion tracking 流程，重点调整 obs 组合和部署相关接口。

Env 必须遵守 `NpEnv` contract：`reset()` 返回 `(obs_dict, info_dict)`，`obs` 是 dict，`obs_groups_spec` 决定 learner 使用哪些观测组。

```python
# src/unilab/envs/motion_tracking/g1/tracking.py
    )
    undesired_contact_z_threshold: float = 0.05
    terminate_on_undesired_contacts: bool = False


@registry.envcfg("G1MotionTracking")
@dataclass
class G1MotionTrackingEnvCfg(G1MotionTrackingCfg):
    """Registered configuration for G1 motion tracking."""

    pass
```

## Obs

观测减少训练期特权信息，保留部署可获得的本体状态、参考信号和动作历史。

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
# conf/ppo/task/g1_motion_tracking_deploy/mujoco.yaml
algo:
  obs_groups:
    actor:
    - actor

```

代码侧观测通常在 `_get_obs`、`_build_obs`、`obs_groups_spec` 或 base env 中组装：

```python
# src/unilab/envs/motion_tracking/g1/tracking.py
    def _get_current_motion(self) -> MotionData:
        if self._motion_data_buffer is None:
            return self.motion_sampler.get_current_motion()
        return self.motion_sampler.get_current_motion(self._motion_data_buffer)

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        # Actor: command(2n) + motion_anchor_pos_b(3) + motion_anchor_ori_b(6)
        #        + linvel(3) + gyro(3) + joint_pos(n) + joint_vel(n) + actions(n)
        # Critic mirrors BeyondMimic physical terms without actor observation noise:
        #        command, motion anchor, robot body pos/ori, linvel, gyro, joints, actions.
        n = self._num_action
        actor_width = getattr(self, "_actor_obs_width", self._actor_obs_dim(n))
```

源码锚点拆解：

| 源码符号 | 中文说明 |
| --- | --- |
| obs_groups_spec | 声明 obs dict 中各观测组的维度，是 wrapper/learner 对齐观测的契约。 |
| _compute_obs | 读取 backend 传感器和 info 字段，拼接 actor/critic 观测。 |

## Action

动作空间与 G1 motion tracking 一致。

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
| action_scale | [0.5475464629911068, 0.35066146637882434, 0.5475464629911068, 0.35066146637882434, 0.43857731392336724, 0.43857731392336724, 0.5475464629911068, 0.35066146637882434, 0.5475464629911068, 0.35066146637882434, 0.43857731392336724, 0.43857731392336724, 0.5475464629911068, 0.43857731392336724, 0.43857731392336724, 0.43857731392336724, 0.43857731392336724, 0.43857731392336724, 0.43857731392336724, 0.43857731392336724, 0.07450087032950714, 0.07450087032950714, 0.43857731392336724, 0.43857731392336724, 0.43857731392336724, 0.43857731392336724, 0.43857731392336724, 0.07450087032950714, 0.07450087032950714] | 策略输出到目标关节角的缩放系数。 |

```yaml
# conf/ppo/task/g1_motion_tracking_deploy/mujoco.yaml
env:
  control_config:
    action_scale:
    - 0.5475464629911068
    - 0.35066146637882434
    - 0.5475464629911068
    - 0.35066146637882434
    - 0.43857731392336724
    - 0.43857731392336724
    - 0.5475464629911068
    - 0.35066146637882434
    - 0.5475464629911068
    - 0.35066146637882434
    - 0.43857731392336724
    - 0.43857731392336724
    - 0.5475464629911068
    - 0.43857731392336724
    - 0.43857731392336724
    - 0.43857731392336724
    - 0.43857731392336724
    - 0.43857731392336724
    - 0.43857731392336724
    - 0.43857731392336724
    - 0.07450087032950714
    - 0.07450087032950714
    - 0.43857731392336724
    - 0.43857731392336724
    - 0.43857731392336724
    - 0.43857731392336724
    - 0.43857731392336724
    - 0.07450087032950714
    - 0.07450087032950714

```

动作到执行器的实际映射由 backend 的 actuator control range 和 env 的 `apply_action`/控制函数完成：

```python
# src/unilab/envs/motion_tracking/g1/tracking.py
            if bias is not None:
                if env_ids is not None:
                    return self.default_angles + bias[env_ids]
                return self.default_angles + bias
        return self.default_angles

    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> np.ndarray:
        state.info["last_actions"] = state.info.get("current_actions", np.zeros_like(actions))
        state.info["current_actions"] = actions
        exec_actions = (
            state.info["last_actions"]
            if self._cfg.control_config.simulate_action_latency
            else actions
```

源码锚点拆解：

| 源码符号 | 中文说明 |
| --- | --- |
| apply_action | 把策略输出缩放为执行器控制目标。 |

## Reward

reward 和终止逻辑沿用 motion tracking 主任务。

### Reward 逐项拆解

reward scale 以 owner YAML 的 `reward.scales` 为真源；源码中的 `RewardConfig` 描述可用字段，训练脚本只做 Hydra 组装和注入。正数通常表示奖励，负数通常表示惩罚，0 表示该项在当前 owner 中关闭或仅保留接口。

| Reward key | Scale | 方向 | 中文含义 |
| --- | --- | --- | --- |
| motion_global_root_pos | 0.5 | 奖励 | 参考 motion 的 root 全局位置跟踪奖励。 |
| motion_global_root_ori | 0.5 | 奖励 | 参考 motion 的 root 全局朝向跟踪奖励。 |
| motion_body_pos | 1.0 | 奖励 | 参考 motion 的各 body 位置跟踪奖励。 |
| motion_body_ori | 1.0 | 奖励 | 参考 motion 的各 body 朝向跟踪奖励。 |
| motion_body_lin_vel | 1.0 | 奖励 | 参考 motion 的 body 线速度跟踪奖励。 |
| motion_body_ang_vel | 1.0 | 奖励 | 参考 motion 的 body 角速度跟踪奖励。 |
| motion_joint_pos | 0.0 | 记录/关闭 | 参考 motion 的关节位置跟踪奖励。 |
| motion_joint_vel | 0.0 | 记录/关闭 | 参考 motion 的关节速度跟踪奖励。 |
| action_rate_l2 | -0.1 | 惩罚 | 动作变化 L2 惩罚， motion tracking 中用于约束动作平滑。 |
| joint_limit | -10.0 | 惩罚 | 关节限位惩罚，避免追踪动作时撞到机械限位。 |

```yaml
# conf/ppo/task/g1_motion_tracking_deploy/mujoco.yaml
reward:
  scales:
    motion_global_root_pos: 0.5
    motion_global_root_ori: 0.5
    motion_body_pos: 1.0
    motion_body_ori: 1.0
    motion_body_lin_vel: 1.0
    motion_body_ang_vel: 1.0
    motion_joint_pos: 0.0
    motion_joint_vel: 0.0
    action_rate_l2: -0.1
    joint_limit: -10.0
  std_root_pos: 0.3
  std_root_ori: 0.4
  std_body_pos: 0.3
  std_body_ori: 0.4
  std_body_lin_vel: 1.0
  std_body_ang_vel: 3.14
  std_joint_pos: 0.2
  std_joint_vel: 1.0

```

源码中的 reward 入口：

```python
# src/unilab/envs/motion_tracking/g1/tracking.py
from unilab.envs.locomotion.g1.base import G1BaseCfg, G1BaseEnv

from .motion_loader import MotionData, MotionLoader, MotionSampler


@dataclass
class RewardConfig:
    """Reward configuration for motion tracking."""

    scales: dict[str, float] = field(
        default_factory=lambda: {
            "motion_global_root_pos": 0.5,
            "motion_global_root_ori": 0.5,
```

源码锚点拆解：

| 源码符号 | 中文说明 |
| --- | --- |
| RewardConfig: | 声明 reward 可用参数和默认 scale。 |
| _init_reward_functions | 把 reward 名称映射到源码函数，owner YAML 的 scale 会按这些 key 分发。 |
| _reward_term_is_active | 单个 reward 项的实现函数，通常返回每个 env 的 reward 向量。 |
| _compute_reward | reward 相关源码锚点。 |
| _exp_reward_from_error | 单个 reward 项的实现函数，通常返回每个 env 的 reward 向量。 |
| _reward_motion_global_root_pos | 单个 reward 项的实现函数，通常返回每个 env 的 reward 向量。 |
| _reward_motion_global_root_ori | 单个 reward 项的实现函数，通常返回每个 env 的 reward 向量。 |
| _reward_motion_body_pos | 单个 reward 项的实现函数，通常返回每个 env 的 reward 向量。 |
| _reward_motion_body_ori | 单个 reward 项的实现函数，通常返回每个 env 的 reward 向量。 |
| _reward_motion_body_lin_vel | 单个 reward 项的实现函数，通常返回每个 env 的 reward 向量。 |
| _reward_motion_body_ang_vel | 单个 reward 项的实现函数，通常返回每个 env 的 reward 向量。 |
| _reward_motion_ee_body_pos_z | 单个 reward 项的实现函数，通常返回每个 env 的 reward 向量。 |

## 初始状态

reset 仍从 motion reference 初始化，并应用同类随机化。

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
# src/unilab/envs/motion_tracking/g1/tracking.py
    DomainRandomizationCapabilities,
    DomainRandomizationProvider,
    IntervalRandomizationPlan,
    ResetPlan,
)
from unilab.dr.dr_utils import (
    build_common_reset_randomization,
    build_interval_push_plan,
    validate_common_reset_randomization,
    validate_interval_push_support,
    zero_actions,
)
from unilab.dr.types import RESET_TERM_GEOM_FRICTION, ResetRandomizationPayload
```

## 终止条件

终止遵循 anchor/末端/接触误差阈值。

终止条件通常由 env 源码中的 `_compute_termination(s)`、reward config 阈值和 `max_episode_seconds` 共同决定。

```yaml
# conf/ppo/task/g1_motion_tracking_deploy/mujoco.yaml
env:
  {}

reward:
  {}

```

```python
# src/unilab/envs/motion_tracking/g1/tracking.py
            robot_body_lin_vel_w,
            robot_body_ang_vel_w,
        ) = self._get_body_state_w()

        # Compute relative body transforms (for observations and rewards)
        self._update_relative_transforms(motion_data, robot_body_pos_w, robot_body_quat_w)

        # Compute terminations
        terminated = self._compute_terminations(motion_data, robot_body_pos_w, robot_body_quat_w)

        # Update failure statistics for adaptive sampling
        self.motion_sampler.update_failure_stats(terminated)

        # Compute reward
        reward = self._compute_reward(
            state.info,
            motion_data,
```

## 域随机化

domain_rand 与 motion tracking cfg 一致，实际启用项由 owner YAML 控制。

域随机化只在 reset/interval 等冷路径触发，不在 step 热路径解析资产或探测 backend 私有能力。

### 域随机化字段拆解

| 配置字段 | 当前值 | 中文说明 |
| --- | --- | --- |
| added_mass_range | [-1.5, 1.5] | 随机采样范围或控制范围。 |
| com_offset_x | [-0.025, 0.025] | 任务 owner YAML 中的配置字段。 |
| com_offset_y | [-0.05, 0.05] | 任务 owner YAML 中的配置字段。 |
| com_offset_z | [-0.05, 0.05] | 任务 owner YAML 中的配置字段。 |
| friction_range | [0.3, 1.2] | 随机采样范围或控制范围。 |
| joint_default_pos_range | [-0.01, 0.01] | 随机采样范围或控制范围。 |
| max_force | [1.0, 1.0, 0.5] | 任务 owner YAML 中的配置字段。 |
| push_interval | 750 | 任务 owner YAML 中的配置字段。 |
| push_robots | True | 布尔开关。 |
| random_com | True | 布尔开关。 |
| randomize_base_mass | True | 域随机化开关。 |
| randomize_geom_friction | True | 域随机化开关。 |
| randomize_joint_default_pos | True | 域随机化开关。 |

```yaml
# conf/ppo/task/g1_motion_tracking_deploy/mujoco.yaml
env:
  domain_rand:
    random_com: true
    com_offset_x:
    - -0.025
    - 0.025
    com_offset_y:
    - -0.05
    - 0.05
    com_offset_z:
    - -0.05
    - 0.05
    randomize_base_mass: true
    added_mass_range:
    - -1.5
    - 1.5
    push_robots: true
    push_interval: 750
    max_force:
    - 1.0
    - 1.0
    - 0.5
    randomize_geom_friction: true
    friction_range:
    - 0.3
    - 1.2
    randomize_joint_default_pos: true
    joint_default_pos_range:
    - -0.01
    - 0.01

```

```python
# src/unilab/envs/motion_tracking/g1/tracking.py
from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.backend import create_backend
from unilab.base.np_env import NpEnvState
from unilab.base.scene import SceneCfg
from unilab.dr import (
    DomainRandomizationCapabilities,
    DomainRandomizationProvider,
    IntervalRandomizationPlan,
    ResetPlan,
)
from unilab.dr.dr_utils import (
    build_common_reset_randomization,
```

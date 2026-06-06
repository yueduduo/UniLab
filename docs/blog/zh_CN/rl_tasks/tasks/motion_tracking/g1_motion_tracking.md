---
title: "G1 Motion Tracking"
slug: "g1_motion_tracking"
category: "motion_tracking"
robot: "g1"
registered_envs: [G1MotionTracking, G1MotionTrackingSAC]
---

# G1 Motion Tracking

## 任务目标

让 G1 全身跟踪参考 motion clip，在 root、body、末端和关节层面逼近离线动作数据。

![G1 Motion Tracking 场景渲染](../../images/tasks/g1_motion_tracking.png)

> 图片由 `_tools/render_static_images.py` 使用 `src/unilab/assets/robots/g1/scene_flat.xml` 的 MuJoCo scene 离屏渲染生成；如果当前环境无法启用 EGL/OSMesa，生成 manifest 会记录失败原因。

## 证据索引

| 项目 | 内容 |
| --- | --- |
| 任务类别 | 动作追踪 |
| 机器人 | g1 |
| 代表 scene | src/unilab/assets/robots/g1/scene_flat.xml |
| 代表源码 | src/unilab/envs/motion_tracking/g1/tracking.py |
| 代表 owner YAML | conf/appo/task/g1_motion_tracking/mujoco.yaml |
| 动作维度 | 29 |

### Registry 与 Owner

| 注册名 | 后端 | 配置类 | 配置源码 |
| --- | --- | --- | --- |
| G1MotionTracking | motrix, mujoco | G1MotionTrackingEnvCfg | src/unilab/envs/motion_tracking/g1/tracking.py |
| G1MotionTrackingSAC | motrix, mujoco | G1MotionTrackingSACCfg | src/unilab/envs/motion_tracking/g1/tracking_sac.py |

| 算法组 | 后端 | training.task_name | owner YAML |
| --- | --- | --- | --- |
| appo | motrix | G1MotionTracking | conf/appo/task/g1_motion_tracking/motrix.yaml |
| appo | mujoco | G1MotionTracking | conf/appo/task/g1_motion_tracking/mujoco.yaml |
| sac | motrix | G1MotionTrackingSAC | conf/offpolicy/task/sac/g1_motion_tracking/motrix.yaml |
| sac | mujoco | G1MotionTrackingSAC | conf/offpolicy/task/sac/g1_motion_tracking/mujoco.yaml |
| ppo | motrix | G1MotionTracking | conf/ppo/task/g1_motion_tracking/motrix.yaml |
| ppo | mujoco | G1MotionTracking | conf/ppo/task/g1_motion_tracking/mujoco.yaml |

## 代码级执行流

这部分不再用通用关键词套模板，而是为 `g1_motion_tracking` 单独写源码导读。源码片段只负责提供证据；每节说明都围绕本任务自己的配置、状态、观测、动作和 reward 语义展开。

| 阶段 | 证据位置 | 代码级说明 |
| --- | --- | --- |
| 1. Hydra owner | conf/appo/task/g1_motion_tracking/mujoco.yaml | 训练命令先 compose owner YAML。这里确定 `training.task_name`、后端身份、obs_groups、reward scale 和 env override。 |
| 2. Registry 构造 Env | src/unilab/envs/motion_tracking/g1/tracking.py | `@registry.envcfg` 注册配置类，`@registry.env` 注册后端实现类；训练脚本只调用 registry，不直接 new 任务类。 |
| 3. Reset/初始状态 | src/unilab/assets/robots/g1/scene_flat.xml | env 从 scene keyframe 或 motion/grasp cache 取初态，再通过 reset randomization 改写 qpos/qvel/info。 |
| 4. Obs dict | src/unilab/envs/motion_tracking/g1/tracking.py | env 读取 backend 传感器和 info，返回 dict；learner 根据 `obs_groups_spec` 和 owner `algo.obs_groups` 选择 actor/critic 输入。 |
| 5. Action 到控制目标 | src/unilab/envs/motion_tracking/g1/tracking.py | policy 输出先按 action scale / 控制顺序映射到 actuator，再由 backend step 推进仿真。 |
| 6. Reward dispatch | src/unilab/envs/motion_tracking/g1/tracking.py | 源码把 reward key 映射到函数，owner YAML 的 scale 决定每项奖励/惩罚是否启用以及权重。 |
| 7. Termination | src/unilab/envs/motion_tracking/g1/tracking.py | env 根据高度、姿态、接触、参考轨迹误差、物体状态或能量阈值生成 done/terminated。 |

### 本任务阅读重点

| 阅读重点 |
| --- |
| G1 Motion Tracking 的核心不是 joystick，而是从 motion npz 采样参考帧并构造机器人/参考误差。 |
| reset 从 motion reference 生成 qpos/qvel，obs 同时包含机器人当前状态和参考 motion anchor/body 目标。 |
| reward 是 root/body/joint/action 多层 tracking 误差，不应按 locomotion 速度跟踪来写。 |

### 本任务注册入口

| 注册名 | 配置类 | 可用后端 |
| --- | --- | --- |
| G1MotionTracking | G1MotionTrackingEnvCfg | motrix, mujoco |
| G1MotionTrackingSAC | G1MotionTrackingSACCfg | motrix, mujoco |

### 本任务源码锚点

| 小节 | 源码文件 | 明确锚点 |
| --- | --- | --- |
| 配置：MotionLoader | src/unilab/envs/motion_tracking/g1/tracking.py | G1MotionTrackingEnvCfg, G1MotionTrackingCfg |
| Reset：参考帧初态 | src/unilab/envs/motion_tracking/g1/tracking.py | build_reset_plan, motion_loader.get_motion_at_frame |
| Obs：机器人 + 参考目标 | src/unilab/envs/motion_tracking/g1/tracking.py | obs_groups_spec, _compute_obs |
| Action：延迟与默认角偏移 | src/unilab/envs/motion_tracking/g1/tracking.py | apply_action, simulate_action_latency |
| Reward/Termination | src/unilab/envs/motion_tracking/g1/tracking.py | _init_reward_functions, update_state |

## 关键源码逐段解释（按本任务手写）

### 配置：MotionLoader

| 本任务人工导读 |
| --- |
| cfg 声明 motion_file、body_names、anchor 和 tracking 阈值。 |
| MotionLoader 在 env 初始化/冷路径读取 motion 数据。 |

```python
# src/unilab/envs/motion_tracking/g1/tracking.py
# 将通用 motion tracking 配置注册成 Hydra 可选择的 G1MotionTracking 任务。
@registry.envcfg("G1MotionTracking")
@dataclass
class G1MotionTrackingEnvCfg(G1MotionTrackingCfg):
    """Registered configuration for G1 motion tracking."""

    # 该注册类不改字段，真实任务语义全部来自父类 G1MotionTrackingCfg。
    pass


```

```python
# src/unilab/envs/motion_tracking/g1/tracking.py
@dataclass
class G1MotionTrackingCfg(G1BaseCfg):
    """Configuration for G1 motion tracking environment."""

    # 默认场景是 G1 平地 scene，keyframe/传感器等 task 资源在 scene 中表达。
    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(
            model_file=str(ASSETS_ROOT_PATH / "robots" / "g1" / "scene_flat.xml")
        )
    )
    # Kept at the historical single-clip default for backward compatibility.
    # motion_file 是参考动作来源，MotionLoader 只在初始化/采样冷路径读取 npz。
    motion_file: str | list[str] = str(
        ASSETS_ROOT_PATH / "motions" / "g1" / "dance1_subject2_part.npz"
    )
    # motion_file: str | list[str] = str(ASSETS_ROOT_PATH / "motions" / "g1" / "gangnam_style.npz")
    # motion_file: str | list[str] = str(ASSETS_ROOT_PATH / "motions" / "g1" / "fight1_subject5_from_csv.npz") #LAFAN
    # motion_file: str | list[str] = str(ASSETS_ROOT_PATH / "motions" / "g1" / "dance_basic_slide_180_R_loop_001__A322_M.npz") #LAFAN
    # motion_file: str | list[str] = str(ASSETS_ROOT_PATH / "motions" / "g1" / "playing_violin_R_003__A327_from_csv.npz") #Seed
    # torso_link 作为 anchor，用来计算机器人当前躯干与参考躯干的相对位姿。
    anchor_body_name: str = "torso_link"
    # body_names 决定 motion 文件里哪些刚体参与 body-level obs/reward/termination。
    body_names: tuple[str, ...] = (
        "pelvis",
        "left_hip_roll_link",
        "left_knee_link",
        "left_ankle_roll_link",
        "right_hip_roll_link",
        "right_knee_link",
        "right_ankle_roll_link",
        "torso_link",
        "left_shoulder_roll_link",
        "left_elbow_link",
        "left_wrist_yaw_link",
        "right_shoulder_roll_link",
        "right_elbow_link",
        "right_wrist_yaw_link",
    )
    # adaptive 采样会根据失败统计倾向困难帧，适合长 motion imitation。
    sampling_mode: Literal["start", "clip_start", "uniform", "adaptive", "mixed"] = "adaptive"
    sampling_start_ratio: float = 0.0
    truncate_on_clip_end: bool = False
    max_episode_seconds: float = 10.0
    # reward/dr 配置由 owner YAML 覆盖，训练脚本不解释这些业务字段。
    reward_config: RewardConfig = field(default_factory=RewardConfig)
    pose_randomization: PoseRandomization = field(default_factory=PoseRandomization)
    velocity_randomization: VelocityRandomization = field(default_factory=VelocityRandomization)
    domain_rand: Domain_Rand = field(default_factory=Domain_Rand)
    # reset 时在参考关节角附近采样，增强从 motion reference 附近恢复的能力。
    joint_position_range: tuple[float, float] = (-0.1, 0.1)
    # Termination thresholds
    # 以下阈值共同定义 imitation 失败边界：躯干高度/朝向、末端高度和异常接触。
    anchor_pos_z_threshold: float = 0.25
    anchor_ori_threshold: float = 0.8
    ee_body_pos_z_threshold: float = 0.25
    ee_body_names: tuple[str, ...] = (
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
    )
    undesired_contact_z_threshold: float = 0.05
    terminate_on_undesired_contacts: bool = False


```
### Reset：参考帧初态

| 本任务人工导读 |
| --- |
| reset 采样 motion frame，并用参考动作构造 qpos/qvel。 |
| randomization 可叠加在参考初态上，而不是替代 motion 目标。 |

```python
# src/unilab/envs/motion_tracking/g1/tracking.py
    def build_reset_plan(self, env: Any, env_ids: np.ndarray) -> ResetPlan:
        num_reset = len(env_ids)
        # sampler 给每个重置 env 选一个参考帧，adaptive 模式会利用失败统计。
        motion_frames = env.motion_sampler.sample_frames(env_ids)
        # MotionLoader 返回该帧的 root、body、joint 参考状态。
        motion_data = env.motion_loader.get_motion_at_frame(motion_frames)
        # qpos/qvel 以参考 motion 为主，再叠加 pose/velocity/joint 随机化。
        qpos, qvel = _build_motion_reference_state(env, env_ids, motion_data)

        # reset 后动作历史清零，避免上一段 episode 的控制量污染首帧 obs/action_rate。
        info_updates = {
            "current_actions": zero_actions(num_reset, env._num_action),
            "last_actions": zero_actions(num_reset, env._num_action),
        }
        # 通用 reset DR 负责质量、COM、kp/kd 等低频随机化 payload。
        randomization = build_common_reset_randomization(
            env, num_reset, base_kp=self._base_kp, base_kd=self._base_kd
        )

        dr_cfg = env.cfg.domain_rand
        if getattr(dr_cfg, "randomize_geom_friction", False):
            assert self._base_geom_friction is not None
            assert self._foot_geom_ids is not None
            payload = randomization or ResetRandomizationPayload()
            low, high = dr_cfg.friction_range
            # 每个 env 采一个脚底摩擦缩放，并广播到匹配的 foot geoms。
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
            # default_dof_pos_bias 会同时影响 obs 的相对关节角和 action 的目标基准。
            info_updates["default_dof_pos_bias"] = np.random.uniform(
                low, high, size=(num_reset, env._num_action)
            ).astype(get_global_dtype())

        # ResetPlan 是 env/backend contract：状态、info 和随机化一次性交给 reset 流程。
        return ResetPlan(
            env_ids=env_ids,
            qpos=qpos,
            qvel=qvel,
            info_updates=info_updates,
            randomization=randomization,
        )

```
### Obs：机器人 + 参考目标

| 本任务人工导读 |
| --- |
| actor obs 包含 command、motion anchor、linvel/gyro、joint 和 action。 |
| critic 额外加入 body pos/ori 等 tracking 特权项。 |

```python
# src/unilab/envs/motion_tracking/g1/tracking.py
    @property
    def obs_groups_spec(self) -> dict[str, int]:
        # Actor: command(2n) + motion_anchor_pos_b(3) + motion_anchor_ori_b(6)
        #        + linvel(3) + gyro(3) + joint_pos(n) + joint_vel(n) + actions(n)
        # Critic mirrors BeyondMimic physical terms without actor observation noise:
        #        command, motion anchor, robot body pos/ori, linvel, gyro, joints, actions.
        n = self._num_action
        # actor_width 是策略实际读取的单步观测宽度。
        actor_width = getattr(self, "_actor_obs_width", self._actor_obs_dim(n))
        # critic_width 在 actor 基础上加入每个 tracked body 的位置/姿态特权信息。
        critic_width = getattr(
            self,
            "_critic_obs_width",
            self._critic_base_obs_dim(n) + len(self._cfg.body_names) * 9,
        )
        return {"obs": actor_width, "critic": critic_width}

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
        # n_body 对应 cfg.body_names 的 motion tracking 刚体数量。
        n_body = self._n_motion_bodies

        # Get anchor states
        # 参考 anchor 与机器人当前 anchor 分别来自 motion_data 和 backend body 状态。
        anchor_pos_w = motion_data.body_pos_w[:, self.anchor_body_idx]
        anchor_quat_w = motion_data.body_quat_w[:, self.anchor_body_idx]
        robot_anchor_pos_w = robot_body_pos_w[:, self.anchor_body_idx]
        robot_anchor_quat_w = robot_body_quat_w[:, self.anchor_body_idx]

        # Motion anchor pose in robot frame
        if num_envs == self._num_envs:
            # 全量 env 更新时复用预分配缓存，减少热路径分配。
            motion_anchor_pos_b = self._motion_anchor_pos_b
            motion_anchor_ori_b = self._motion_anchor_ori_b
            joint_pos_rel = self._joint_pos_rel
            zero_actions = self._zero_actions
        else:
            # reset observation 只计算部分 env，因此需要临时数组匹配 batch 大小。
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
        # 相对关节角以默认姿态为零点；若 reset 随机化默认角，需要同步扣除 bias。
        effective_default = self.default_angles + bias if bias is not None else self.default_angles
        np.subtract(dof_pos, effective_default, out=joint_pos_rel)
        last_actions = info.get("current_actions")
        if not isinstance(last_actions, np.ndarray):
            last_actions = zero_actions

        if num_envs == self._num_envs:
            command = self._motion_command
        else:
            command = np.empty((num_envs, n_action * 2), dtype=dtype)
        # command 拼接参考关节角和参考关节速度，是策略跟踪 motion 的主要目标信号。
        command[:, :n_action] = motion_data.joint_pos
        command[:, n_action : n_action * 2] = motion_data.joint_vel

        # Per-step observation noise on sensor channels (actor only).
        # Critic uses the clean originals — asymmetric actor–critic contract.
        noise_cfg = self._cfg.noise_config
        noise_enabled = noise_cfg.level > 0.0
        if noise_enabled:
            # 噪声只加在 actor 传感通道，critic 保持干净状态用于非对称训练。
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
        # actor_obs 是部署时策略输入：参考目标 + anchor 误差 + 本体感知 + 动作历史。
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
        # critic_obs 显式写入干净 command/body 信息，用于价值函数估计 tracking 误差。
        critic_obs = np.empty((num_envs, self._critic_obs_width), dtype=dtype)
        offset = 0
        critic_obs[:, offset : offset + command.shape[1]] = command
        offset += command.shape[1]
```
### Action：延迟与默认角偏移

| 本任务人工导读 |
| --- |
| 动作可按配置模拟 latency，之后叠加 default_dof_pos_bias。 |
| 这是 sim2real/部署前常见控制口径。 |

```python
# src/unilab/envs/motion_tracking/g1/tracking.py
    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> np.ndarray:
        # 保存上一帧和当前帧动作，reward 的 action_rate 和 obs 的 last_actions 都依赖它。
        state.info["last_actions"] = state.info.get("current_actions", np.zeros_like(actions))
        state.info["current_actions"] = actions
        # 开启 latency 时，本步真正下发上一帧动作，模拟实机控制延迟。
        exec_actions = (
            state.info["last_actions"]
            if self._cfg.control_config.simulate_action_latency
            else actions
        )
        bias = state.info.get("default_dof_pos_bias")
        # 控制目标围绕默认关节角展开；默认角随机化时 action 基准也跟着移动。
        base = self.default_angles + bias if bias is not None else self.default_angles
        # 策略输出是归一化 29 维动作，按 action_scale 映射为 position target。
        ctrl: np.ndarray = exec_actions * self._cfg.control_config.action_scale + base
        return ctrl

```
### Reward/Termination

| 本任务人工导读 |
| --- |
| reward map 是多层 motion tracking：root、body、joint、velocity、action_rate。 |
| termination 看 anchor z/ori、ee z、undesired contact 和 clip 结束。 |

```python
# src/unilab/envs/motion_tracking/g1/tracking.py
    def _init_reward_functions(self):
        # reward_config.scales 中的名字会通过这张表映射到真实计算函数。
        self._reward_fns = {
            # root/body 项约束全身 motion imitation 的空间姿态和速度。
            "motion_global_root_pos": self._reward_motion_global_root_pos,
            "motion_global_root_ori": self._reward_motion_global_root_ori,
            "motion_body_pos": self._reward_motion_body_pos,
            "motion_body_ori": self._reward_motion_body_ori,
            "motion_body_lin_vel": self._reward_motion_body_lin_vel,
            "motion_body_ang_vel": self._reward_motion_body_ang_vel,
            # ee/joint 项补充末端高度和关节级误差，权重由 YAML 决定是否启用。
            "motion_ee_body_pos_z": self._reward_motion_ee_body_pos_z,
            "motion_joint_pos": self._reward_motion_joint_pos,
            "motion_joint_vel": self._reward_motion_joint_vel,
            # 控制正则和异常接触项负责稳定性与失败惩罚。
            "action_rate_l2": self._reward_action_rate_l2,
            "joint_limit": self._reward_joint_limit,
            "undesired_contacts": self._reward_undesired_contacts,
        }

```

```python
# src/unilab/envs/motion_tracking/g1/tracking.py
    def update_state(self, state: NpEnvState) -> NpEnvState:
        # 每步先清空 clip-end truncation 标记，后面 motion sampler 再按需写入。
        self._clip_end_truncated.fill(False)

        # Get current motion data
        # 当前参考帧由 motion_sampler 维护，reward/obs/termination 都基于同一份 motion_data。
        motion_data = self._get_current_motion()

        # Get robot state
        # 本体速度、IMU、关节状态来自 backend，是 actor/critic/reward 的公共输入。
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
        # 将参考 body 与当前 body 的相对变换缓存起来，避免 obs/reward 重复计算。
        self._update_relative_transforms(motion_data, robot_body_pos_w, robot_body_quat_w)

        # Compute terminations
        # termination 使用 anchor/ee/contact 等边界判断 motion tracking 是否失败。
        terminated = self._compute_terminations(motion_data, robot_body_pos_w, robot_body_quat_w)

        # Update failure statistics for adaptive sampling
        # adaptive sampler 根据失败帧更新后续 reset 的采样分布。
        self.motion_sampler.update_failure_stats(terminated)

        # Compute reward
        # reward 对比当前机器人状态与参考 motion，逐项加权后输出每个 env 的标量。
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
        # obs 必须保持 dict contract，actor/critic 组由 obs_groups_spec 声明维度。
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
                resample_env_ids = done_env_ids[~terminated[done_env_ids]]
                if len(resample_env_ids) > 0:
                    self._resample_reference_state(resample_env_ids)
                    self._refresh_observation_rows(obs, state.info, resample_env_ids)

        return state.replace(obs=obs, reward=reward, terminated=terminated)

```



## Agent

Agent 输出 G1 29 维动作；PPO/APPO/SAC 变体共享 motion tracking 目标，但 critic/算法配置不同。

训练入口通过 owner YAML 决定注册 env、后端身份和算法参数；CLI 应使用 `--sim` 选择 backend owner，而不是单独 override `training.sim_backend`。

```yaml
# conf/appo/task/g1_motion_tracking/mujoco.yaml
training:
  task_name: G1MotionTracking
  sim_backend: mujoco
  play_steps: 1000
algo:
  obs_groups: {}
  num_envs: 1024
  max_iterations: 5000
```

## Env

Env 使用 `MotionLoader` 读取 npz motion，并在 reset 时采样 clip 时间。

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

观测围绕当前机器人状态、参考 motion 相位/目标和动作历史构造；SAC 变体会额外给 critic base linvel。

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
# conf/appo/task/g1_motion_tracking/mujoco.yaml
algo:
  obs_groups:
    {}

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

动作控制 G1 29 个 actuator，通过 action_scale 映射到关节目标。

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

```yaml
# conf/appo/task/g1_motion_tracking/mujoco.yaml
env:
  control_config:
    {}

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

reward 包含 root pos/ori、body pos/ori、body velocity、joint pos/vel、action_rate 和 joint_limit 等 tracking 项。

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
# conf/appo/task/g1_motion_tracking/mujoco.yaml
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

初始状态从 motion reference 构造，并叠加 pose/velocity/joint 随机化。

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

终止由 anchor z/ori 误差、末端 z 误差、undesired contact 和 clip 截断策略决定。

终止条件通常由 env 源码中的 `_compute_termination(s)`、reward config 阈值和 `max_episode_seconds` 共同决定。

```yaml
# conf/appo/task/g1_motion_tracking/mujoco.yaml
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

domain_rand 支持 base mass、COM、gravity、push、kp/kd、friction 和 joint default pos 扰动。

域随机化只在 reset/interval 等冷路径触发，不在 step 热路径解析资产或探测 backend 私有能力。

### 域随机化字段拆解

| 配置字段 | 当前值 | 中文说明 |
| --- | --- | --- |

```yaml
# conf/appo/task/g1_motion_tracking/mujoco.yaml
env:
  domain_rand:
    {}

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

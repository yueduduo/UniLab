# K1SoccerDribble 参数版本记录

## 记录规范
- 每次改参都新增一个带后缀的 YAML 快照文件：`motrix_vXXX_*.yaml`
- 主配置文件始终为：`../motrix.yaml`
- 训练命令固定：
  - `uv run train --algo flashsac --task k1_soccer_dribble --sim motrix training.device=cuda training.no_play=true`

## v001（pre_stable）
- 文件：`motrix_v001_pre_stable.yaml`
- 时间：2026-06-01
- 目的：第一版“可训练”参数，强调带球与姿态约束。
- 关键参数：
  - `algo.updates_per_step=2`
  - `algo.tau=0.05`
  - `env.control_config.action_scale=0.22`
  - `reward.scales.ball_over_speed=-2.5`
  - `reward.scales.leg_twist=-1.2`
  - `reward.ball_speed_cap=1.5`

## v002（stable）
- 文件：`motrix_v002_stable.yaml`
- 时间：2026-06-01
- 目的：降低 SAC 后期回落风险，优先训练稳定性。
- 关键参数调整（相对 v001）：
  - 学习稳定性：
    - `algo.updates_per_step: 2 -> 1`
    - `algo.learning_starts: 50 -> 200`
    - `algo.tau: 0.05 -> 0.02`
    - `algo.algo_params.temp_target_sigma: 0.22`（新增）
  - 动作平滑与命令范围：
    - `env.control_config.action_scale: 0.22 -> 0.20`
    - `env.commands.vel_limit.max_vx: 0.75 -> 0.65`
  - 奖励权重再平衡：
    - `tracking_lin_vel: 2.0 -> 1.6`
    - `ball_progress: 1.2 -> 1.5`
    - `ball_keep: 1.2 -> 1.0`
    - `ball_speed_match: 0.9 -> 0.7`
    - `ball_moving: 1.0 -> 0.8`
    - `ball_lost: -1.8 -> -1.2`
    - `ball_over_speed: -2.5 -> -1.4`
    - `leg_twist: -1.2 -> -0.6`
  - 终止/约束阈值：
    - `min_forward_speed_for_gait_reward: 0.05 -> 0.03`
    - `ball_lost_distance: 1.2 -> 1.35`
    - `ball_lost_distance_hard: 2.0 -> 2.4`
    - `ball_speed_cap: 1.5 -> 1.35`
    - `ball_move_speed_target: 0.25 -> 0.20`

## v003（anti_collapse）
- 文件：`motrix_v003_anti_collapse.yaml`
- 时间：2026-06-01
- 目的：针对“约 8k 后 reward 回落”做抗退化调整，优先保探索与学习稳定性。
- 关键参数调整（相对 v002）：
  - 学习稳定性：
    - `algo.learning_starts: 200 -> 500`
    - `algo.replay_buffer_n: 4096 -> 2048`（降低旧分布滞后）
    - `algo.tau: 0.02 -> 0.01`
    - `algo.actor_lr: 2e-4`（新增）
    - `algo.critic_lr: 2e-4`（新增）
  - 温度/探索：
    - `algo.algo_params.temp_initial_value: 0.02`（新增）
    - `algo.algo_params.temp_target_sigma: 0.22 -> 0.30`
  - 奖励再平衡（降低互相拉扯）：
    - `ball_progress: 1.5 -> 1.7`
    - `ball_speed_match: 0.7 -> 0.6`
    - `ball_moving: 0.8 -> 0.6`
    - `ball_lost: -1.2 -> -0.8`
    - `ball_over_speed: -1.4 -> -1.0`
    - `leg_twist: -0.6 -> -0.4`
  - 终止放宽：
    - `ball_lost_distance_hard: 2.4 -> 2.8`

## v004（fix_inward_foot）
- 文件：`motrix_v004_fix_inward_foot.yaml`
- 时间：2026-06-01
- 目的：针对“脚内扣”问题，增加足部 roll 姿态的直接约束。
- 关键参数调整（相对 v003）：
  - 新增奖励项：
    - `leg_roll_posture`（惩罚 `Left/Right_Hip_Roll` 与 `Left/Right_Ankle_Roll` 偏离默认姿态）
  - 奖励权重：
    - `leg_roll_posture: -1.2`（新增）
- 代码同步：
  - `soccer_dribble.py` 新增 `leg_roll_posture` 奖励函数并注册到 reward dispatch。

## v005（reduce_orbit）
- 文件：`motrix_v005_reduce_orbit.yaml`
- 时间：2026-06-01
- 目的：针对“绕球转圈不直踢/带球”问题，增加逼近与反绕圈约束。
- 关键参数调整（相对 v004）：
  - 新增奖励项：
    - `ball_approach`（远球阶段奖励向球逼近的径向闭合速度）
    - `ball_orbit`（惩罚相对球的切向绕圈速度）
  - 奖励权重：
    - `ball_approach: +0.9`
    - `ball_orbit: -1.2`
  - 新增阈值：
    - `ball_approach_speed_cap: 0.6`
- 代码同步：
  - `soccer_dribble.py` 新增 `_reward_ball_approach` 与 `_reward_ball_orbit` 并注册。

## v006（no_warmstart_ball_still）
- 文件：`motrix_v006_no_warmstart_ball_still.yaml`
- 时间：2026-06-01
- 目的：从 0 训练（禁用旧权重预热）并在球水平几乎不动时加入小惩罚。
- 关键参数调整（相对 v005）：
  - warm-start：
    - `algo.warm_start.enabled: true -> false`
  - 新增奖励项：
    - `ball_still`（惩罚球水平速度低于阈值）
  - 奖励权重：
    - `ball_still: -0.2`（小惩罚）
  - 新增阈值：
    - `ball_still_speed_threshold: 0.06`
- 代码同步：
  - `soccer_dribble.py` 新增 `_reward_ball_still` 并注册到 reward dispatch。

## v007（fix_inward_yaw_only）
- 文件：`motrix_v007_fix_inward_yaw_only.yaml`
- 时间：2026-06-01
- 目的：按人工可视化确认结果，精确限制“脚朝向另一只脚横过去”的错误姿态。
- 关键参数调整（相对 v006）：
  - 删除旧逻辑：
    - `leg_twist`（移除）
    - `leg_roll_posture`（移除）
  - 新增替代逻辑：
    - `feet_inward_yaw`（仅惩罚三种目标错误：左脚内横 / 右脚内横 / 双脚都朝内）
  - 奖励权重：
    - `feet_inward_yaw: -1.4`
- 代码同步：
  - `soccer_dribble.py` 删除旧脚姿态惩罚函数，新增 `_reward_feet_inward_yaw` 并注册。

## v008（alive_0_8）
- 文件：`motrix_v008_alive_0_8.yaml`
- 时间：2026-06-01
- 目的：提升“存活导向”，降低早期频繁倾倒终止的学习挫败。
- 关键参数调整（相对 v007）：
  - `alive: 0.2 -> 0.8`

## v009（warmstart_k1walkflat）
- 文件：`motrix_v009_warmstart_k1walkflat.yaml`
- 时间：2026-06-01
- 目的：按用户要求恢复 `K1WalkFlat` 预热，避免从零起步导致早期频繁倒地。
- 关键参数调整（相对 v008）：
  - `algo.warm_start.enabled: false -> true`
  - `algo.warm_start.source_task: K1WalkFlat`（保留）
  - `algo.warm_start.source_algo_log_name: flash_sac`（保留）
  - `algo.warm_start.source_load_run: -1`（保留，默认取最新 run）

## v010（cmd_toward_ball）
- 文件：`motrix_v010_cmd_toward_ball.yaml`
- 时间：2026-06-01
- 目的：命令观测不再依赖随机速度采样，改为“从机器人指向球”的方向命令。
- 关键参数调整（相对 v009）：
  - `env.commands.vel_limit` 从前向单轴改为平面双向：
    - `[[0.35, 0.0, 0.0], [0.65, 0.0, 0.0]]`
    - `-> [[-0.65, -0.65, 0.0], [0.65, 0.65, 0.0]]`
- 代码同步：
  - `soccer_dribble.py` 新增 `_command_toward_ball`，每步用机器人到球的相对方向构造命令；
  - 观测构建和奖励上下文都统一使用该命令。

## v011（boost_approach_front）
- 文件：`motrix_v011_boost_approach_front.yaml`
- 时间：2026-06-01
- 目的：针对 `ball_approach` / `ball_front` 贡献偏小的问题，提高两项在总回报中的权重与可达性。
- 关键参数调整（相对 v010）：
  - 权重增强：
    - `ball_approach: 0.9 -> 2.0`
    - `ball_front: 0.8 -> 1.6`
  - 形状放宽：
    - `ball_front_lateral_sigma: 0.10 -> 0.16`（横向偏差更宽容，front 项更容易取高）
    - `ball_approach_speed_cap: 0.6 -> 1.0`（approach 项上限提高）

## v012（replay_1024）
- 文件：`motrix_v012_replay_1024.yaml`
- 时间：2026-06-01
- 目的：缩小 replay buffer，降低旧样本滞后，提升当前策略主导的数据新鲜度。
- 关键参数调整（相对 v011）：
  - `replay_buffer_n: 2048 -> 1024`

## v013（anti_fall_cmd_smooth）
- 文件：`motrix_v013_anti_fall_cmd_smooth.yaml`
- 时间：2026-06-01
- 目的：缓解中后期“急冲急转后倒地”问题，平滑“指向球”命令并放宽倒地终止阈值。
- 关键参数调整（相对 v012）：
  - 终止阈值放宽：
    - `min_base_height: 0.35 -> 0.30`
    - `max_tilt_deg: 35.0 -> 45.0`
  - 新增命令平滑参数：
    - `ball_cmd_deadzone: 0.08`
    - `ball_cmd_kp: 1.2`
- 代码同步：
  - `soccer_dribble.py` 的 `_command_toward_ball` 从“按距离直接给速”改为“超过 keep+deadzone 后按比例增长”，降低近球急转导致的摔倒概率。


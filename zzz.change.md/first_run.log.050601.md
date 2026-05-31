# first_run.log.050601

日期：2026-06-01  
范围：G1WalkFlat APPO 训练 / Motrix owner 配置 / APPO 键盘交互回放

---

## 新增文件

### `conf/appo/task/g1_walk_flat/motrix.yaml`

- 为 APPO + `g1_walk_flat` 补 Motrix owner 配置（原先仅有 `mujoco.yaml`）。
- `training.sim_backend: motrix`
- `algo.num_envs: 2048`（与 PPO motrix owner 一致）
- env / reward 对齐 `conf/ppo/task/g1_walk_flat/motrix.yaml`（如 `action_scale: 0.5`、`vel_limit`、gait phase reward 等）
- APPO 训练超参沿用 mujoco 版（`max_iterations: 5000`、`save_interval: 1000`）

---

## 修改文件

### `src/unilab/visualization/interactive_playback.py`

- 新增 APPO learner checkpoint 识别与加载：
  - `load_checkpoint_payload` / `is_appo_learner_checkpoint`
  - `infer_actor_input_dim_from_checkpoint_path`（同时支持 `actor_state_dict` 与 `actor`）
  - `load_algo_config_from_run_dir`（从 checkpoint 同目录 `run_config.json` 读取 actor 结构）
  - `build_appo_actor` / `build_appo_inference_policy`
- `create_rsl_rl_playback_session`：检测到 APPO 格式时直接加载 `checkpoint["actor"]`，不再走 `OnPolicyRunner`（避免 `actor_state_dict` KeyError）。
- PPO / RSL-RL checkpoint 行为不变。

### `scripts/play_interactive.py`

- 新增 `_resolve_playback_checkpoint`：在 `rsl_rl_ppo` / `appo` 日志目录间自动解析 checkpoint。
- 新增 `_resolve_playback_algo_log_name`：从 APPO checkpoint 的 `run_config.json` 推断 `algo_log_name`。
- `_infer_checkpoint_actor_input_dim` 改为复用 shared helper，支持 APPO checkpoint。
- APPO 回放时使用 `BackendAdapter(..., algo_name="appo")` 构建 env override。
- 文档示例中补充 APPO + 键盘遥控用法。

### `tests/visualization/test_interactive_playback.py`

- 新增 APPO checkpoint 输入维度推断测试。
- 新增 APPO checkpoint 加载 playback session 测试。
- 修正 RSL-RL checkpoint 测试：使用临时真实 `.pt` 文件。

---

## 训练命令

### APPO + MuJoCo（已完成 run）

```bash
uv run train --algo appo --task g1_walk_flat --sim mujoco training.no_play=true
```

本次完成 run：

- 日志：`logs/appo/G1WalkFlat/2026-06-01_00-11-17_mujoco/`
- checkpoint：`model_5000.pt`
- 总步数：491,520,000（5000 iter）

### APPO + Motrix（新增 owner 后）

```bash
uv run train --algo appo --task g1_walk_flat --sim motrix training.no_play=true
```

前置：已安装 Motrix extra（`make setup-motrix` 或 `uv sync --extra motrix`）。

---

## 回放 / 评估命令

### 录视频（无键盘）

```bash
uv run eval --algo appo --task g1_walk_flat --sim mujoco \
  --load-run 2026-06-01_00-11-17_mujoco
```

输出：`logs/appo/G1WalkFlat/2026-06-01_00-11-17_mujoco/play_video.mp4`

注意：不要加 `training.log_root=logs`，APPO 默认日志根为 `logs/appo/`。

### FlashSAC checkpoint 评估（参考，非本次代码改动）

```bash
uv run eval --algo flashsac --task g1_walk_flat --sim mujoco \
  --load-run 2026-05-31_22-35-08_mujoco \
  training.log_root=logs \
  training.export_onnx=false
```

FlashSAC 无键盘入口；仅 `eval` 录视频。

---

## 键盘交互启动命令

需要桌面环境（`DISPLAY` 可用），MuJoCo 原生 viewer：

```bash
uv run scripts/play_interactive.py task=g1_walk_flat/mujoco \
  algo.load_run=2026-06-01_00-11-17_mujoco \
  interactive.action_mode=policy \
  interactive.keyboard=true
```

也可直接指定 checkpoint 文件：

```bash
uv run scripts/play_interactive.py task=g1_walk_flat/mujoco \
  algo.load_run=logs/appo/G1WalkFlat/2026-06-01_00-11-17_mujoco/model_5000.pt \
  interactive.action_mode=policy \
  interactive.keyboard=true
```

键盘操作：

| 按键 | 功能 |
|------|------|
| ↑ / ↓ | 前进 / 后退（vx） |
| ← / → | 左转 / 右转（vyaw） |
| Enter | 停车 |
| Space | 暂停 / 继续 |
| Esc | 退出 |

---

## 已知限制

- `play_interactive` 仅支持 MuJoCo viewer，Motrix 训练权重需切回 `task=g1_walk_flat/mujoco` 做键盘回放。
- SAC / FlashSAC checkpoint 仍不能用于 `play_interactive`；请用 `uv run eval`。
- 日志面板 `Steps/s` 不含 Wait 时间，端到端吞吐需用 wall-clock 估算。

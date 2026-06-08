# UniLab Agent Principles

**Always use `uv run`, not python**.

UniLab 是一个 **高性能、模块化、contract 驱动** 的 RL infrastructure 仓库。

## Core Principles

1. **Contract first**: 不为了一次通过绕过 env / backend / runner contract。
2. **Fix at owner layer**: `scripts/` 只组装流程，不承载长期业务规则。
3. **Config first**: task / reward / backend 优先通过 Hydra + registry 表达。
4. **Backend isolation**: MuJoCo / Motrix 差异留在 backend 适配层和配置层。
5. **Evidence only**: support claim 只写仓库里已有的注册、配置、测试或 benchmark 事实。
6. **Validate near risk**: 在最接近风险的边界补验证，不只跑顶层命令。
7. **Cold-path asset access only**: asset/XML/model metadata 只允许在 init / materialization / cache 等低频路径处理；热路径不能解析 asset，也不能靠 `getattr` / `hasattr` 探测 backend 私有能力。

## High-Risk Areas

| 区域 | 不可破坏的不变量 |
|------|----------------|
| Env  | `NpEnvState.obs` 必须是 dict；`reset()` 返回 `(obs_dict, info_dict)`；`obs_groups_spec` 影响 wrapper 和 learner 维度。 |
| Config / Reward | reward 通过 Hydra 注入；后端切换必须通过 `task=<task>/<backend>` 选择 owner YAML，`training.sim_backend` 只是 owner YAML 的身份字段，不能单独 override 来切后端。算法超参数直接走 YAML compose，不经 Python 层解释。 |
| Backend | backend-specific 逻辑留在 backend / env 适配层，不向训练脚本扩散。env 层只能调用 `SimBackend`（`base.py`）中已声明的方法；若某方法只在 MuJoCo 或 Motrix 中存在，必须先将其加入 `SimBackend` 抽象接口（可抛 `NotImplementedError`），禁止直接在 env 里调用 backend 子类的私有方法（即"功能泄漏/feature leakage"）。新增 backend 专有能力时，需同步更新 `SimBackend`。 |
| Asset / Metadata | `ASSETS_ROOT_PATH`、`model_file`、XML / asset 元数据只允许在 init / materialization / cache 等低频路径访问；`step/reset/domain randomization` 等热路径不得解析 asset 或基于 asset 元数据做运行时分支。 |
| Asset / XML structure | `<keyframe>` 必须放在 task-level XML（`scene_*.xml` 或 `locomotion_task.xml` 等 fragment），**禁止放进 robot.xml**。robot.xml 是纯机器人描述（body / joint / actuator / sensor），跟 task / 场景无关；keyframe 是 task 起始姿态，属于场景或 task 资源。motrix 后端需要 keyframe 时通过 `scene.fragment_files` 引用 fragment XML。 |
| Async | 不绕开 runner lifecycle，也不另起 collector / learner 同步协议。 |

## Pointers

- PPO: `scripts/train_rsl_rl.py`
- MLX PPO: `scripts/train_mlx_ppo.py`
- APPO: `scripts/train_appo.py`
- SAC / TD3: `scripts/train_offpolicy.py`
- env contract: `src/unilab/base/np_env.py`
- backend contract: `src/unilab/base/backend/base.py`
- training run helpers: `src/unilab/training/run.py`
- visualization helpers: `src/unilab/visualization/`
- env shared numeric helpers: `src/unilab/envs/common/rotation.py`, `src/unilab/envs/common/math.py`
- MLX rotation helpers: `src/unilab/algos/mlx/common/rotation.py`
- config schema: `src/unilab/structured_configs.py`
- async runner: `src/unilab/ipc/async_runner.py`

## K1SoccerPenaltyKick 场景查看与录视频

### 初始布局（右脚与球同 Y 轴）

stand keyframe 下，机器人 base 在 `y=0` 时右脚 site 约在 `y=-0.0962`。为让右脚与点球点（`y=0`）对齐，base Y 上移 `0.0962 m`：

- 常量：`src/unilab/envs/locomotion/k1/soccer_penalty_constants.py`（`K1_PENALTY_RIGHT_FOOT_Y_OFFSET_FROM_BASE_M`、`K1_PENALTY_ROBOT_START_XY`）
- keyframe：`src/unilab/assets/robots/k1/scene_soccer_penalty_kick.xml`（base `1.0 0.0962 ...`）
- 训练 reset 仍可能有 Y 方向 `±0.15 m` jitter（`K1_PENALTY_RESET_Y_OFFSET_M`），eval 回放不一定每次完全对齐

### 静态场景预览

```bash
# 生成 PNG（默认输出到 conf/.../figures/scene_preview.png）
uv run scripts/view_k1_penalty_kick_scene.py --preview-only

# 打开 MuJoCo 交互查看器
uv run scripts/view_k1_penalty_kick_scene.py
```

脚本会打印 `right_foot` / `ball` 坐标及 `dy`（对齐时应 ≈ 0）。

### 策略回放录视频

用 `uv run eval`，`--render-mode record` 无头导出 mp4；checkpoint 可传 run 目录名或 `.pt` 绝对路径。

**单 env、32 s、多次 reset 不同起点**（`ctrl_dt=0.02` → `play_steps=1600`；env 默认 autoreset，episode 结束后会重新采样 reset 位置）：

```bash
cd ~/projects/UniLab && uv run eval --algo flashsac --task k1_soccer_penalty_kick --sim mujoco \
  --render-mode record \
  algo.load_run=/path/to/logs/flash_sac/K1SoccerPenaltyKick/<run>/model_<iter>.pt \
  training.export_onnx=false training.play_env_num=1 training.play_steps=1600
```

短片段（约 16 s）示例：

```bash
cd ~/projects/UniLab && uv run eval --algo flashsac --task k1_soccer_penalty_kick --sim mujoco \
  --render-mode record \
  algo.load_run=/path/to/logs/flash_sac/K1SoccerPenaltyKick/<run>/model_<iter>.pt \
  training.export_onnx=false training.play_env_num=1 training.play_steps=800
```

- 默认输出：`<run_dir>/play_video.mp4`（可手动 `cp` 为 `play_video_model_<iter>_1env_32s.mp4`）
- `training.play_env_num=1`：只看单机器人；`16` 会并排 16 个 env，起点各不相同但画面拥挤
- `training.play_steps`：控制步数 = 视频秒数 / `ctrl_dt`（K1 点球 `ctrl_dt=0.02`，32 s → 1600）
- 训练结束自动录制的 `play_video.mp4` 默认 `play_env_num=16`；要看 reset 多样性请用上面单 env 命令
- `--load-run` 只接受 run 目录名；传 `.pt` 路径时用 `algo.load_run=...` override
- owner YAML：`conf/offpolicy/task/flashsac/k1_soccer_penalty_kick/mujoco.yaml`

## GitHub CLI (gh) 速查

### Issue 查看
```bash
gh issue view <number>
gh api repos/<owner>/<repo>/issues/<number> --jq '.body'
```

### PR 创建与管理
```bash
gh pr create --title "标题" --body "内容" --base main
gh pr list
gh pr view
```

### PR Gate

创建或更新 PR 前必须满足：

1. 最终提交已经完成，且 `git status --short --branch` 确认工作树干净。
2. 最终提交已经通过 `make test-all`。
3. 如果用户明确说明已经跑过 `make test-all`，不要重复跑；但必须在 PR body 的 Validation 里记录 `make test-all` 已完成。
4. 如果 `make test-all` 未通过且用户没有明确 override，不要创建或更新 PR。

### CI 工作流查看
```bash
gh run list
gh run list --workflow=<workflow-name>
gh run view <run-id>
gh run list --status=failure
```

### 常用组合
```bash
gh api repos/unilabsim/UniLab/issues/174 --jq '.title, .body'
git push -u origin fix/issue-174-mlx-ppo-config-alignment
gh pr create --title "fix: xxx" --body "Fixes #174" --base main
```

## Context

- 架构标准与验证详情：[docs/sphinx/source/zh_CN/4-developer_guide/0-index.md](docs/sphinx/source/zh_CN/4-developer_guide/0-index.md)
- 协作流程与 PR 规范：[docs/sphinx/source/zh_CN/4-developer_guide/5-contributing_workflow.md](docs/sphinx/source/zh_CN/4-developer_guide/5-contributing_workflow.md)
- 开发者入口（环境、命令、提交规范）：[CONTRIBUTING.md](CONTRIBUTING.md)
- 文档本地构建与发布到 UniLab-doc：[docs/sphinx/README.md#本地发布到-unilab-doc](docs/sphinx/README.md#本地发布到-unilab-doc)

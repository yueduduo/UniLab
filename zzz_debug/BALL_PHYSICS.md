# 足球滚动不减速问题排查与修复记录

本文档记录 `zzz_debug/` 下「推球调试工具」中，球施力后速度不逐渐减小的问题、排查过程与最终修改。涉及文件：

| 文件 | 作用 |
|------|------|
| `scene_ball_pitch.xml` | 球 + 球场场景 |
| `push_ball.py` | 滚轮施力交互主脚本 |
| `load_ball_field.py` | 场景加载与渲染入口 |
| `ground_pick.py` | 鼠标 → 地面拾取 |
| `benchmark_sim_dt.py` | headless 对比 `sim_dt` / `ctrl_dt` 组合 |

运行方式：

```bash
uv run zzz_debug/push_ball.py
```

---

## 1. 现象

在 Motrix 可视化窗口中，对球施加水平冲量后：

- HUD 显示球获得初速度并开始滚动；
- 速度很快稳定在某个值（约 **1.59 m/s**），之后 **长时间不再下降**；
- 预期行为：球应在地面摩擦/滚动阻力下 **逐渐减速至停止**。

Headless 复现（无渲染、无 Python 干预）同样成立：给 `dof_vel[:3] = [2, 0, 0]` 后，`vx` 约在 20 步内升到 ~1.588，此后 **200+ 步保持不变**。

---

## 2. 初步误判（已废弃）

曾怀疑 Motrix 缺少滚动阻力，在 `push_ball.py` 中每步手动施加：

- `apply_ground_drag()`：`add_external_force` / `add_external_torque` 线性阻尼；
- `_sync_roll_spin()`：冲量后强行把角速度与线速度按 `v/r` 对齐。

这 **不符合 UniLab 原则**（物理规则应写在场景/XML，不应在 debug 脚本里 hack），且只是掩盖症状。对照 `sim_soccer2` 后确认：**同引擎下 sim_soccer2 场景里球会正常减速**，问题出在 **场景碰撞配置**，而非 Motrix 全局缺失阻力。

上述 Python 代码已全部删除。

---

## 3. 对照 sim_soccer2 的发现

参考路径：`sim_soccer2/simulation/motrixsim/assets/environments/soccer/world.xml`

### 3.1 球场碰撞

| 项目 | UniLab 旧 debug / `scene_soccer_dribble_minimal.xml` | sim_soccer2 |
|------|------------------------------------------------------|-------------|
| 碰撞体 | `type="plane"` 无限平面 | `type="box"` 薄盒 `size="6 4.5 0.01"`，`pos z≈0.014` |
| 名称 | `COL_Collider` | 同名 `COL_Collider` |
| 可视 | 单独绿色 `pitch` plane（无碰撞） | 单独 mesh 地板（无碰撞） |

### 3.2 足球碰撞

| 项目 | UniLab 旧 debug | sim_soccer2 |
|------|-----------------|-------------|
| 碰撞形状 | 独立 **`sphere`** `ball_geom` | **mesh** `Robot_Football.obj`（无单独 sphere） |
| 可视 | mesh `ball_visual`（group 1，无碰撞） | 同一 mesh（默认 group，可渲染） |
| joint | `ball-root` + `damping="0.15"` | `ball-root` free，**无 damping** |
| 质量 | sphere 显式 `mass="0.43"` | mesh 体积推算，约 **1.23 kg** |

### 3.3 仿真步长（XML 与代码路径不一致）

| 来源 | timestep | 说明 |
|------|----------|------|
| `world.xml` | `0.005` | 球场资源里的 `<option>` |
| `K1_22dof.xml` | `0.001` | 机器人 MJCF 里的 `<option>` |
| sim_soccer2 **运行时** | **`0.001`** | `MultiRobotMotrixSim` 读 `model.options.timestep`；场景合并以 **robot XML 为根**，**不会**并入 `world.xml` 的 `<option>` |
| `runtime_config.py` 常量 | `SIM_DT=0.005`, `control_decimation=4` | **未写回** `model.options.timestep`，主路径不生效 |
| 旧 debug | `0.02` | 已废弃 |

按说明文档启动（`sim2sim_runner.py` / Sim Manager）时，K1 默认 **策略周期** `ctrl_dt = sim_dt × control_decimation = 0.001 × 4 = **0.004 s**`（decider `config.yaml` 默认 `sim_hz=200` 只限制墙钟 ZMQ 步频，不改变每步仿真时间增量）。

---

## 4. Headless 对比实验（Motrix）

在 `zzz_debug/` 目录下加载临时/正式 XML，设 `qvel[:3]=[2,0,0]`，观察 4 s 内 `vx`：

| 配置 | t=0 | t=1s | t=2s | t=3s | t=4s |
|------|-----|------|------|------|------|
| plane + **sphere**（旧） | 0 | 1.588 | 1.588 | 1.588 | 1.588 |
| box + sphere | 0 | 1.588 | 1.588 | 1.588 | 1.588 |
| plane + **mesh**（sim_soccer2 方式） | 0 | 1.050 | 0.674 | 0.250 | 0.013 |
| **当前最终配置**（plane + 双 mesh） | 0 | 1.296 | 1.053 | — | 0.743 |

**结论：**

1. **决定性因素**是球的碰撞体类型：**sphere 在 Motrix 中与地面接触时几乎无滚动减速**；**mesh 碰撞**才有合理滚动阻力。
2. 将 plane 换成 box ** alone 不能**修复 sphere 不减速。
3. 增大 XML 中 `friction` roll、`joint damping`、`option viscosity` 对 **sphere 方案** 无明显效果（headless 已测）。

sim_soccer2 完整 world（mesh 球 + box 场地，`timestep=0.005`）同样在 4 s 内从 ~1.37 m/s 降至 ~0.90 m/s。

---

## 5. 修改过程（时间线）

### 5.1 第一版 debug 场景

- 复制 `scene_soccer_dribble_minimal.xml` 思路：绿色 plane 可视 + plane `COL_Collider` + sphere `ball_geom` + mesh `ball_visual`。
- **问题**：球滚起来后速度「冻住」。

### 5.2 错误尝试：Python 层 drag

- 在 `push_ball.py` 每步 `apply_ground_drag()` + 冲量后 `_sync_roll_spin()`。
- HUD 上能看到速度变化，但属于 **人为阻尼**，与正式任务物理不一致。

### 5.3 对照 sim_soccer2，改 XML

1. `COL_Collider`：`plane` → 薄 **box**（与 sim_soccer2 一致）。
2. 球碰撞：`sphere ball_geom` → **mesh**（`Robot_Football.obj`）。
3. 去掉 `ball-root` 的 `damping="0.15"`。
4. `timestep`：`0.02` → **`0.005`**（`load_ball_field.py` 中 `SIM_DT` 同步修改）。
5. 删除 `push_ball.py` 中全部 drag / roll-sync hack。

### 5.5 薄 box 场地导致 z 轴穿透（APPO 回放崩溃）

将 `scene_soccer_dribble_minimal.xml` 的 `COL_Collider` 换成 sim_soccer2 式薄 box 后，带球回放出现：

- 球 `vz` 莫名增大、`|v|_xy` 长期钉在 ~1.04 m/s
- 球/机器人 z 乱跳，最终 Motrix panic：`LTL factorization failed: NotPositiveDefinite`

Headless 对比（K1 + 球，mesh 碰撞，竖直踢出 vz=8 m/s）：

| 落地后 | plane 场地 | 薄 box 场地 |
|--------|------------|-------------|
| t≈1.8s | z≈0.3，正常触地反弹 | z≈2.0，**vz 反转为 +11**（穿透地板） |
| t≈3.0s | z≈0.3 附近 | z≈**8.4** 悬在空中 |

原因：Motrix 下 **薄 box（厚 0.02 m）** 挡不住带水平速度的 mesh 球高速落体，球穿地后 z 发散，进而拖垮约束求解。

**修复**：`K1SoccerDribble` 最小场景 **保留 infinite plane** 作 `COL_Collider`；滚动减速只依赖 **mesh 球**（与 box 无关）。sim_soccer2 全尺寸球场用 box 是另一套资产/尺度，不能直接套到 minimal plane 场景。

### 5.7 mesh 球在 plane 上 z 向弹跳（机器人走动时更明显）

`COL_Collider` 是 **刚体 plane**，不会被机器人「压弯」或「压抖」。

Motrix 下 **mesh 碰撞** 即使用 `solref/solimp` 压到 ~3 cm 波动，带球回放仍可能看到小幅上下跳；**sphere 碰撞** 在同条件下 `ball_z` 波动约 **一半**（headless env + 腿部激励：mesh ~2.6 cm vs sphere ~1.2 cm）。

**带球任务**（`scene_soccer_dribble_minimal.xml`）：使用 **mesh + 软接触**（`solref="0.02 1"`），优先保证 **脱脚后自减速**；z 向会有 ~1–3 cm 小幅波动（Motrix 限制，sphere 无法同时满足减速）。

| 碰撞体 | z 触地稳定性 | 脱脚后地面滚动减速 |
|--------|--------------|-------------------|
| mesh + 软接触（**当前带球场景**） | 一般（~2–3 cm） | **有** |
| sphere | 较好（~1 cm） | Motrix 几乎无 |

`zzz_debug/push_ball.py` 使用相同 mesh 方案验证 coast 物理。

---

### 5.6 副作用：球在窗口中不可见

- 第一版 mesh 修复把 **唯一** 球 geom 设为 `group="3"`（碰撞组）。
- Motrix **默认不渲染 group 3**，球仍在仿真中运动，但视口里看不见。
- sim_soccer2 的球 mesh **没有** `group="3"`，因此可见。

**修复**：恢复双 geom（与 UniLab 正式场景相同模式）：

```xml
<geom name="ball_visual" type="mesh" ... contype="0" conaffinity="0" group="1"/>  <!-- 仅渲染 -->
<geom name="ball"         type="mesh" ... group="3"/>                              <!-- 仅碰撞 -->
```

Headless 验证：双 mesh 仍保持正常减速（4 s 内 vx 从 ~1.17 降至 ~0.44）。

---

## 5.8 物理步长：0.005 不稳，0.002 为 UniLab 当前默认

完全贴合 sim_soccer2（box `COL_Collider` + mesh 球 + 硬接触 default + 无 joint damping）后，Motrix **impulse solver** 与 MuJoCo Newton 对步长敏感度不同：

| `sim_dt` | box+mesh（早期 headless settle） | 说明 |
|----------|--------------------------------|------|
| 0.005 | 不稳，z 抖动、易发散 | 不宜作 Motrix 默认 |
| **0.002** | **极稳**，减速正常；UniLab 正式任务当前采用 | `scene_soccer_dribble_minimal.xml` + `K1SoccerDribbleCfg` |
| 0.001 | 见 **§10**（与 sim_soccer2 运行时一致）；正常速度下与 0.002 相当 | 早期在其它条件下曾测到 z 发散，后验需区分「正常推球」与「vx≥3 穿地」 |

初始穿透修正：box 顶面必须对齐 `z=0`（`pos="0 0 -0.1" size="6 4.5 0.1"`），否则球/脚初始嵌入 box 触发 `NotPositiveDefinite` panic。

---

## 6. 当前最终配置摘要（完全贴合 sim_soccer2）

`scene_soccer_dribble_minimal.xml` / `scene_ball_pitch.xml` 要点：

- **碰撞地面**：`COL_Collider` **box**（`pos="0 0 -0.1" size="6 4.5 0.1"`，顶面 z=0），贴合 sim_soccer2 的 box 类型
- **球**：`ball_visual`（group 1 贴图，`mass="0"`）+ `ball`/`ball_geom` **mesh 碰撞**（group 3，`mass="0.43"`）
- **接触**：**硬接触 default `solref="0.001 1"`**（不 override），贴合 sim_soccer2
- **joint**：`ball-root` free，**无 damping**，贴合 sim_soccer2
- **步长（UniLab 当前）**：`timestep=0.002` / `cfg.sim_dt=0.002`，`ctrl_dt=0.02` → **decimation=10**
- **步长（sim_soccer2 运行时）**：`sim_dt=0.001`，`control_decimation=4` → **ctrl_dt=0.004**（见 §10）

实测（headless，正式场景 @ 0.002/0.02）：踢出后 vx **0.86 → 0.30**（持续减速）；机器人走动时 ball z 波动 **p95-p5 ≈ 1.3 mm**（几乎不跳）。

`push_ball.py` 仅通过 `data.set_dof_vel()` 施加冲量，**不再**每步修改力/力矩。

---

## 7. 与 UniLab 正式任务场景

`src/unilab/assets/robots/k1/scene_soccer_dribble_minimal.xml`（`K1SoccerDribble` / `play_latest_auto_appo.py` 所用场景）+ `K1SoccerDribbleCfg.sim_dt=0.002` 已同步：

- `COL_Collider`：**box**（顶面 z=0，加厚防穿透）
- `ball_geom`：**mesh** 碰撞 + 硬接触 default（自减速；`mass="0.43"`）
- `ball_visual`：`mass="0"`，避免双 mesh 重复计质量
- `ball-root`：无 joint damping
- `sim_dt=0.002`（Motrix box+mesh 稳定步长）

> 注意：旧 checkpoint 在不同物理（plane/sphere、sim_dt=0.005）下训练，回放手感可能与训练时不同；新训练会用到此物理。

---

## 8. 已知其他说明

- **球质量**：mesh 碰撞体质量约 1.23 kg（由 mesh 体积与 `density` 决定），高于 FIFA 式 0.43 kg sphere；与 sim_soccer2 一致。如需指定质量，需在 asset/geom 层显式配置，而非 Python hack。
- **鼠标拾取**：`ground_pick.py` 通过截屏检测 `ball_visual` 像素校准，与碰撞 mesh 分离无冲突。
- **施力实现**：冲量必须经 `data.set_dof_vel()` 写入；直接改 `data.dof_vel` 切片在 Motrix 中不生效（属 API 使用问题，与减速无关）。

---

## 9. 验证命令

Headless 快速检查减速（在仓库根目录）：

```bash
cd zzz_debug && uv run python -c "
from load_ball_field import load_ball_pitch_scene, ball_pos
import numpy as np
model, data = load_ball_pitch_scene()
ball = model.get_link('ball')
qvel = np.zeros(len(data.dof_vel), np.float32)
qvel[:3] = [2.0, 0, 0]
data.set_dof_vel(qvel)
ts = float(model.options.timestep)
for t in range(5):
    v = float(ball.get_linear_velocity(data)[0])
    print(f't={t}s vx={v:.3f}')
    for _ in range(int(1/ts)):
        model.step(data)
"
```

预期：`|vx|` 随时间单调下降（非恒定在 ~1.59）。

可视化交互：

```bash
uv run zzz_debug/push_ball.py
```

施力后观察左上角 HUD 中 `|v|` 与 `|v_xy|` 应持续减小。

---

## 10. `sim_dt` / `ctrl_dt` benchmark：`0.001 / 0.004` vs `0.002 / 0.020`

复现脚本：`zzz_debug/benchmark_sim_dt.py`（场景 `scene_ball_pitch.xml`：box `COL_Collider` + mesh 球，与正式带球一致）。

```bash
uv run zzz_debug/benchmark_sim_dt.py
uv run zzz_debug/benchmark_sim_dt.py --env-standing   # 附带 K1SoccerDribble 站立 5s
```

### 10.1 纯球滚动（初始 vx=1.5 m/s，仿真 4s）

| 配置 | decimation | 4s 末 vx | z 抖动 (p95-p5) |
|------|------------|----------|-----------------|
| **0.001 / 0.004**（对齐 sim_soccer2 仿真时间尺度） | 4 | 0.350 m/s | **0.45 mm** |
| **0.002 / 0.020**（UniLab 当前） | 10 | 0.332 m/s | **0.42 mm** |

中间时刻 vx 几乎重合（例：t=1s 约 0.86 m/s，t=2s 约 0.66–0.69 m/s）。**正常推球速度下，0.001 并不差于 0.002**，mesh 滚动减速行为一致。

### 10.2 高速穿地压力（初始 vx=3.0 m/s）

| 配置 | 4s 末 vx | z min | 结论 |
|------|----------|-------|------|
| 0.001 | ~1.44 m/s（冻住） | **-7.27 m** | 穿地发散 |
| 0.002 | ~1.46 m/s（冻住） | **-7.63 m** | 同样穿地 |

高速下 **两种步长都会穿地**；这是 box+mesh 在 Motrix 下的共同弱点，不是 0.001 独有。

### 10.3 完整带球环境（K1 站立 + zero action，仿真 5s）

| 配置 | ctrl_dt | ball z (p95-p5) | ball \|vx\| max |
|------|---------|-----------------|----------------|
| 0.001 / 0.004 | 0.004 | **1.86 mm** | 0.025 m/s |
| 0.002 / 0.020 | 0.020 | **2.89 mm** | 0.052 m/s |

站立扰动下 0.001/0.004 **无 solver panic**，球 z 略稳于当前默认。

### 10.4 结论（2026-06 实测）

1. **Motrix + box+mesh + 硬接触** 在 **vx ≈ 0.3–1.5 m/s** 时，`sim_dt=0.001` 与 `0.002` **物理效果相当**（减速、z 稳定均正常）。
2. **sim_soccer2 按文档启动** 的真实仿真步长是 **`sim_dt=0.001`、`ctrl_dt=0.004`**（来自加载后 `model.options.timestep` + `control_decimation=4`），不是 `world.xml` 里的 0.005，也不是 `runtime_config.SIM_DT=0.005`。
3. UniLab 当前 **`sim_dt=0.002`、`ctrl_dt=0.02`** 是 Motrix 上经验选的稳定默认；若要对齐 sim_soccer2 **策略时间尺度**，可改为 `0.001 / 0.004`，但 **已有 APPO checkpoint 在 0.002/0.02 下训练**，切换后回放手感会变，需重训或接受差异。
4. **不建议** 为对齐 sim_soccer2 单独改 XML `timestep=0.001` 而不改 `cfg.sim_dt`：Motrix backend 以 `cfg.sim_dt` 覆盖 `model.options.timestep`。
5. **vx ≥ 3 m/s** 的极端冲量下两种步长都不稳；带球任务应依赖 mesh 减速 + 合理步长，而非更高初速砸向地面。

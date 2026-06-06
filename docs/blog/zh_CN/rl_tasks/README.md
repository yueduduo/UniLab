---
title: "UniLab 强化学习任务与机器人资产文档"
slug: "unilab-rl-tasks"
lang: "zh-CN"
---

# UniLab 强化学习任务与机器人资产文档

这是一套面向博客发布的中文文档快照，来源于当前 UniLab 仓库的 registry、Hydra owner YAML、env 源码和 MJCF 资产。它不改变训练入口，也不把任务规则写入脚本；所有结论都尽量指向仓库内证据。

## 目录结构

- `robots/`：每个机器人一个子目录，包含机器人介绍、关节/执行器/传感器表和复制后的资产快照。
- `tasks/`：每个强化学习任务一篇 Markdown，按 locomotion、motion_tracking、manipulation、manip_loco 分类。
- `images/`：MuJoCo 离屏渲染的机器人默认姿态图与任务场景图。
- `_generated/`：inventory、asset manifest、render manifest 等机器生成事实源。
- `_tools/`：文档生成与配图脚本。

## 机器人索引

| 机器人 | 定位 | 文档 | 主场景 |
| --- | --- | --- | --- |
| Unitree Go1 | 四足速度跟踪与崎岖地形基线机器人 | [文档](robots/go1/index.md) | `src/unilab/assets/robots/go1/scene_flat.xml` |
| Unitree Go2 | 四足运动控制、倒立与前足站立任务机器人 | [文档](robots/go2/index.md) | `src/unilab/assets/robots/go2/scene_flat.xml` |
| Unitree Go2W | 轮足式 joystick 运动控制机器人 | [文档](robots/go2w/index.md) | `src/unilab/assets/robots/go2w/scene_flat.xml` |
| Unitree G1 | 人形行走与全身动作追踪机器人 | [文档](robots/g1/index.md) | `src/unilab/assets/robots/g1/scene_flat.xml` |
| Booster K1 | 人形行走与足球盘带机器人 | [文档](robots/k1/index.md) | `src/unilab/assets/robots/k1/scene_flat.xml` |
| Go2 + Airbot Arm | 四足移动操作机器人 | [文档](robots/go2_arm/index.md) | `src/unilab/assets/robots/go2_arm/scene_flat.xml` |
| Allegro Hand | 灵巧手球体旋转与抓取生成任务 | [文档](robots/allegro_hand/index.md) | `src/unilab/assets/robots/allegro_hand/scene.xml` |
| Sharpa Wave Hand | 触觉灵巧手旋转与抓取任务 | [文档](robots/sharpa_wave/index.md) | `src/unilab/assets/robots/sharpa_wave/scene.xml` |

## 任务索引

| 任务 | 类别 | 机器人 | 算法组 | 后端 | 文档 |
| --- | --- | --- | --- | --- | --- |
| Go1 Joystick Flat | 运动控制 | go1 | appo, ppo, td3 | motrix, mujoco | [文档](tasks/locomotion/go1_joystick_flat.md) |
| Go1 Joystick Rough | 运动控制 | go1 | ppo | motrix, mujoco | [文档](tasks/locomotion/go1_joystick_rough.md) |
| Go2 Joystick Flat | 运动控制 | go2 | appo, flashsac, ppo, td3 | motrix, mujoco | [文档](tasks/locomotion/go2_joystick_flat.md) |
| Go2 Joystick Rough | 运动控制 | go2 | ppo | motrix, mujoco | [文档](tasks/locomotion/go2_joystick_rough.md) |
| Go2 HandStand | 运动控制 | go2 | ppo | motrix, mujoco | [文档](tasks/locomotion/go2_handstand.md) |
| Go2 FootStand | 运动控制 | go2 | ppo | mujoco | [文档](tasks/locomotion/go2_footstand.md) |
| Go2W Joystick Flat | 运动控制 | go2w | ppo | motrix, mujoco | [文档](tasks/locomotion/go2w_joystick_flat.md) |
| Go2W Joystick Rough | 运动控制 | go2w | ppo | motrix, mujoco | [文档](tasks/locomotion/go2w_joystick_rough.md) |
| G1 Walk Flat | 运动控制 | g1 | appo, flashsac, ppo, sac, td3 | motrix, mujoco | [文档](tasks/locomotion/g1_walk_flat.md) |
| G1 Walk Rough | 运动控制 | g1 | sac | motrix, mujoco | [文档](tasks/locomotion/g1_walk_rough.md) |
| K1 Walk Flat | 运动控制 | k1 | appo, flashsac, ppo | motrix, mujoco | [文档](tasks/locomotion/k1_walk_flat.md) |
| K1 Soccer Dribble | 运动控制 | k1 | appo, flashsac | motrix | [文档](tasks/locomotion/k1_soccer_dribble.md) |
| Go2 Arm Manip-Loco | 移动操作 | go2_arm | ppo, ppo_him | motrix, mujoco | [文档](tasks/manip_loco/go2_arm_manip_loco.md) |
| G1 Motion Tracking | 动作追踪 | g1 | appo, ppo, sac | motrix, mujoco | [文档](tasks/motion_tracking/g1_motion_tracking.md) |
| G1 Motion Tracking Deploy | 动作追踪 | g1 | ppo | motrix, mujoco | [文档](tasks/motion_tracking/g1_motion_tracking_deploy.md) |
| G1 Flip Tracking | 动作追踪 | g1 | appo, ppo, sac | motrix, mujoco | [文档](tasks/motion_tracking/g1_flip_tracking.md) |
| G1 Wall Flip Tracking | 动作追踪 | g1 | appo, ppo, sac | motrix, mujoco | [文档](tasks/motion_tracking/g1_wall_flip_tracking.md) |
| G1 Climb Tracking | 动作追踪 | g1 | appo, ppo | motrix, mujoco | [文档](tasks/motion_tracking/g1_climb_tracking.md) |
| G1 Box Tracking | 动作追踪 | g1 | ppo | motrix, mujoco | [文档](tasks/motion_tracking/g1_box_tracking.md) |
| G1 WBT Obs | 动作追踪 | g1 | sac | mujoco | [文档](tasks/motion_tracking/g1_wbt_obs.md) |
| Allegro In-Hand Rotation | 灵巧操作 | allegro_hand | appo, ppo | motrix, mujoco | [文档](tasks/manipulation/allegro_inhand.md) |
| Allegro In-Hand Grasp | 灵巧操作 | allegro_hand | ppo | motrix, mujoco | [文档](tasks/manipulation/allegro_inhand_grasp.md) |
| Sharpa In-Hand Rotation | 灵巧操作 | sharpa_wave | appo, hora_distill, ppo, sac | motrix, mujoco | [文档](tasks/manipulation/sharpa_inhand.md) |
| Sharpa In-Hand Grasp | 灵巧操作 | sharpa_wave | ppo | motrix, mujoco | [文档](tasks/manipulation/sharpa_inhand_grasp.md) |

## 生成命令

```bash
uv run docs/blog/zh_CN/rl_tasks/_tools/generate_rl_docs.py
uv run docs/blog/zh_CN/rl_tasks/_tools/render_static_images.py
```

## 事实源摘要

| 事实源 | 数量 |
| --- | --- |
| Registry env | 27 |
| 文档任务 | 24 |
| 机器人 | 8 |
| Owner YAML | 80 |

## 阅读建议

先从机器人页理解资产和关节，再进入任务页阅读 Agent/Env/Obs/Action/Reward/初始状态/终止条件/域随机化。任务页中的 YAML 片段代表当前 owner 配置，源码片段展示对应 env 的 owner layer 实现。

```
cd /home/m/projects/UniLab/docs/blog/zh_CN/rl_tasks/_site
uv run python -m http.server 8000
```
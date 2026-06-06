---
title: "Allegro Hand 机器人资产与关节说明"
slug: "allegro_hand"
robot: "allegro_hand"
source_scene: "src/unilab/assets/robots/allegro_hand/scene.xml"
---

# Allegro Hand 机器人资产与关节说明

![Allegro Hand 默认姿态](../../images/robots/allegro_hand.png)

Allegro Hand 是 16 DoF 灵巧手，任务场景中 include 手本体和球体资产，用于手内旋转控制与抓取初始化生成。关节表按手指列出，可直接对应 action 维度。

## 机器人定位

- 用途：灵巧手球体旋转与抓取生成任务
- 主场景：`src/unilab/assets/robots/allegro_hand/scene.xml`
- 默认 keyframe：`home`
- 资产快照：`robots/allegro_hand/assets/`

这份文档只把 `src/unilab/assets/robots/allegro_hand/` 下的机器人、场景、mesh、texture 等素材复制到博客目录。motion、checkpoint、cache 不属于机器人结构说明，未复制进文档资产目录。

## 编译模型概览

| 字段 | 数值 | 说明 |
| --- | --- | --- |
| nq | 23 | 广义坐标维度，free joint 的四元数会占 4 维 |
| nv | 22 | 广义速度维度 |
| nu | 16 | 执行器数量，也就是默认 action 维度 |
| njnt | 17 | MuJoCo 编译后的 joint 数，包含 free/object joint |
| nsensor | 4 | 传感器数量 |
| keyframes | home | scene/task 层 keyframe |

## 资产结构

<details>
<summary>已复制文件（默认折叠，点击展开）</summary>

| 已复制文件 |
| --- |
| docs/blog/zh_CN/rl_tasks/robots/allegro_hand/assets/allegro_right.xml |
| docs/blog/zh_CN/rl_tasks/robots/allegro_hand/assets/assets/base_link.stl |
| docs/blog/zh_CN/rl_tasks/robots/allegro_hand/assets/assets/base_link_left.stl |
| docs/blog/zh_CN/rl_tasks/robots/allegro_hand/assets/assets/link_0.0.stl |
| docs/blog/zh_CN/rl_tasks/robots/allegro_hand/assets/assets/link_1.0.stl |
| docs/blog/zh_CN/rl_tasks/robots/allegro_hand/assets/assets/link_12.0_left.stl |
| docs/blog/zh_CN/rl_tasks/robots/allegro_hand/assets/assets/link_12.0_right.stl |
| docs/blog/zh_CN/rl_tasks/robots/allegro_hand/assets/assets/link_13.0.stl |
| docs/blog/zh_CN/rl_tasks/robots/allegro_hand/assets/assets/link_14.0.stl |
| docs/blog/zh_CN/rl_tasks/robots/allegro_hand/assets/assets/link_15.0.stl |
| docs/blog/zh_CN/rl_tasks/robots/allegro_hand/assets/assets/link_15.0_tip.stl |
| docs/blog/zh_CN/rl_tasks/robots/allegro_hand/assets/assets/link_2.0.stl |
| docs/blog/zh_CN/rl_tasks/robots/allegro_hand/assets/assets/link_3.0.stl |
| docs/blog/zh_CN/rl_tasks/robots/allegro_hand/assets/assets/link_3.0_tip.stl |
| docs/blog/zh_CN/rl_tasks/robots/allegro_hand/assets/assets/link_4.0.stl |
| docs/blog/zh_CN/rl_tasks/robots/allegro_hand/assets/assets/ycb/056_tennis_ball/google_16k/kinbody.xml |
| docs/blog/zh_CN/rl_tasks/robots/allegro_hand/assets/assets/ycb/056_tennis_ball/google_16k/nontextured.ply |
| docs/blog/zh_CN/rl_tasks/robots/allegro_hand/assets/assets/ycb/056_tennis_ball/google_16k/nontextured.stl |
| docs/blog/zh_CN/rl_tasks/robots/allegro_hand/assets/assets/ycb/056_tennis_ball/google_16k/texture_map.png |
| docs/blog/zh_CN/rl_tasks/robots/allegro_hand/assets/assets/ycb/056_tennis_ball/google_16k/textured.dae |
| docs/blog/zh_CN/rl_tasks/robots/allegro_hand/assets/assets/ycb/056_tennis_ball/google_16k/textured.mtl |
| docs/blog/zh_CN/rl_tasks/robots/allegro_hand/assets/assets/ycb/056_tennis_ball/google_16k/textured.obj |
| docs/blog/zh_CN/rl_tasks/robots/allegro_hand/assets/assets/ycb/056_tennis_ball/google_16k/textured.obj.mtl |
| docs/blog/zh_CN/rl_tasks/robots/allegro_hand/assets/assets/ycb/056_tennis_ball/google_16k/textured_backup.obj |
| docs/blog/zh_CN/rl_tasks/robots/allegro_hand/assets/assets/ycb/056_tennis_ball/google_16k/textured_vhacd.obj |
| docs/blog/zh_CN/rl_tasks/robots/allegro_hand/assets/assets/ycb/056_tennis_ball/google_16k/textured_vhacd_backup.obj |
| docs/blog/zh_CN/rl_tasks/robots/allegro_hand/assets/assets/ycb/056_tennis_ball.urdf |
| docs/blog/zh_CN/rl_tasks/robots/allegro_hand/assets/ball.xml |
| docs/blog/zh_CN/rl_tasks/robots/allegro_hand/assets/scene.xml |

</details>

关键 XML 片段如下。`<include>` 把纯机器人 XML 放入 scene，`<keyframe>` 留在 scene 或 task fragment 中，符合 UniLab 的资产分层约束。

```xml
<!-- src/unilab/assets/robots/allegro_hand/scene.xml -->
<mujoco model="allegro inhand rotation scene">
  <include file="allegro_right.xml"/>
  <include file="ball.xml"/>

  <statistic center="0 0 0.1" extent="0.8"/>
```

## 关节说明

`free` joint 表示浮动基座或任务物体自由度，不直接对应 policy action；`controlled=是` 表示该 joint 被 actuator 直接驱动，是 action 空间的一部分。“中文含义”按机器人运动学命名拆解，用于快速理解每个关节控制的身体部位和运动方向。

| ID | 关节名称 | 中文含义 | 类型 | 有限位 | 关节限制 | 受控 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | ffj0 | 食指第 1 个弯曲/展开关节 | hinge | 是 | -0.47 ~ 0.47 | 是 |
| 1 | ffj1 | 食指第 2 个弯曲/展开关节 | hinge | 是 | -0.196 ~ 1.61 | 是 |
| 2 | ffj2 | 食指第 3 个弯曲/展开关节 | hinge | 是 | -0.174 ~ 1.709 | 是 |
| 3 | ffj3 | 食指第 4 个弯曲/展开关节 | hinge | 是 | -0.227 ~ 1.618 | 是 |
| 4 | mfj0 | 中指第 1 个弯曲/展开关节 | hinge | 是 | -0.47 ~ 0.47 | 是 |
| 5 | mfj1 | 中指第 2 个弯曲/展开关节 | hinge | 是 | -0.196 ~ 1.61 | 是 |
| 6 | mfj2 | 中指第 3 个弯曲/展开关节 | hinge | 是 | -0.174 ~ 1.709 | 是 |
| 7 | mfj3 | 中指第 4 个弯曲/展开关节 | hinge | 是 | -0.227 ~ 1.618 | 是 |
| 8 | rfj0 | 无名指第 1 个弯曲/展开关节 | hinge | 是 | -0.47 ~ 0.47 | 是 |
| 9 | rfj1 | 无名指第 2 个弯曲/展开关节 | hinge | 是 | -0.196 ~ 1.61 | 是 |
| 10 | rfj2 | 无名指第 3 个弯曲/展开关节 | hinge | 是 | -0.174 ~ 1.709 | 是 |
| 11 | rfj3 | 无名指第 4 个弯曲/展开关节 | hinge | 是 | -0.227 ~ 1.618 | 是 |
| 12 | thj0 | 拇指第 1 个弯曲/展开关节 | hinge | 是 | 0.263 ~ 1.396 | 是 |
| 13 | thj1 | 拇指第 2 个弯曲/展开关节 | hinge | 是 | -0.105 ~ 1.163 | 是 |
| 14 | thj2 | 拇指第 3 个弯曲/展开关节 | hinge | 是 | -0.189 ~ 1.644 | 是 |
| 15 | thj3 | 拇指第 4 个弯曲/展开关节 | hinge | 是 | -0.162 ~ 1.719 | 是 |
| 16 | ball_joint | 足球或球体的 6 自由度自由关节 | free | 否 | 无 | 否 |

## 执行器说明

执行器表来自 MuJoCo 编译模型的 `actuator` 段。RL action 的每一维最终会映射到这些执行器的控制目标或控制范围。

| ID | 执行器名称 | 驱动关节 | 中文含义 | 控制范围 |
| --- | --- | --- | --- | --- |
| 0 | ffa0 | ffj0 | 食指第 1 个弯曲/展开关节 | -0.47 ~ 0.47 |
| 1 | ffa1 | ffj1 | 食指第 2 个弯曲/展开关节 | -0.196 ~ 1.61 |
| 2 | ffa2 | ffj2 | 食指第 3 个弯曲/展开关节 | -0.174 ~ 1.709 |
| 3 | ffa3 | ffj3 | 食指第 4 个弯曲/展开关节 | -0.227 ~ 1.618 |
| 4 | mfa0 | mfj0 | 中指第 1 个弯曲/展开关节 | -0.47 ~ 0.47 |
| 5 | mfa1 | mfj1 | 中指第 2 个弯曲/展开关节 | -0.196 ~ 1.61 |
| 6 | mfa2 | mfj2 | 中指第 3 个弯曲/展开关节 | -0.174 ~ 1.709 |
| 7 | mfa3 | mfj3 | 中指第 4 个弯曲/展开关节 | -0.227 ~ 1.618 |
| 8 | rfa0 | rfj0 | 无名指第 1 个弯曲/展开关节 | -0.47 ~ 0.47 |
| 9 | rfa1 | rfj1 | 无名指第 2 个弯曲/展开关节 | -0.196 ~ 1.61 |
| 10 | rfa2 | rfj2 | 无名指第 3 个弯曲/展开关节 | -0.174 ~ 1.709 |
| 11 | rfa3 | rfj3 | 无名指第 4 个弯曲/展开关节 | -0.227 ~ 1.618 |
| 12 | tha0 | thj0 | 拇指第 1 个弯曲/展开关节 | 0.263 ~ 1.396 |
| 13 | tha1 | thj1 | 拇指第 2 个弯曲/展开关节 | -0.105 ~ 1.163 |
| 14 | tha2 | thj2 | 拇指第 3 个弯曲/展开关节 | -0.189 ~ 1.644 |
| 15 | tha3 | thj3 | 拇指第 4 个弯曲/展开关节 | -0.162 ~ 1.719 |

## 传感器说明

传感器既包括机器人本体 IMU/速度传感器，也可能包括 scene/task fragment 增加的接触传感器。任务文档会说明哪些传感器进入 obs 或 reward。

| ID | 传感器名称 | 维度 |
| --- | --- | --- |
| 0 | ff_contact | 1 |
| 1 | mf_contact | 1 |
| 2 | rf_contact | 1 |
| 3 | th_contact | 1 |

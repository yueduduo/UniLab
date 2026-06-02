---
title: "Unitree Go2 机器人资产与关节说明"
slug: "go2"
robot: "go2"
source_scene: "src/unilab/assets/robots/go2/scene_flat.xml"
---

# Unitree Go2 机器人资产与关节说明

![Unitree Go2 默认姿态](../../images/robots/go2.png)

Go2 覆盖 UniLab 中最完整的四足运动任务族：平地速度跟踪、崎岖地形、handstand 与 footstand。它的 12 个可控关节与足端接触传感器构成了 locomotion reward 和终止条件的主要观测对象。

## 机器人定位

- 用途：四足运动控制、倒立与前足站立任务机器人
- 主场景：`src/unilab/assets/robots/go2/scene_flat.xml`
- 默认 keyframe：`home`
- 资产快照：`robots/go2/assets/`

这份文档只把 `src/unilab/assets/robots/go2/` 下的机器人、场景、mesh、texture 等素材复制到博客目录。motion、checkpoint、cache 不属于机器人结构说明，未复制进文档资产目录。

## 编译模型概览

| 字段 | 数值 | 说明 |
| --- | --- | --- |
| nq | 19 | 广义坐标维度，free joint 的四元数会占 4 维 |
| nv | 18 | 广义速度维度 |
| nu | 12 | 执行器数量，也就是默认 action 维度 |
| njnt | 13 | MuJoCo 编译后的 joint 数，包含 free/object joint |
| nsensor | 38 | 传感器数量 |
| keyframes | home | scene/task 层 keyframe |

## 资产结构

<details>
<summary>已复制文件（默认折叠，点击展开）</summary>

| 已复制文件 |
| --- |
| docs/blog/zh_CN/rl_tasks/robots/go2/assets/assets/base_0.obj |
| docs/blog/zh_CN/rl_tasks/robots/go2/assets/assets/base_1.obj |
| docs/blog/zh_CN/rl_tasks/robots/go2/assets/assets/base_2.obj |
| docs/blog/zh_CN/rl_tasks/robots/go2/assets/assets/base_3.obj |
| docs/blog/zh_CN/rl_tasks/robots/go2/assets/assets/base_4.obj |
| docs/blog/zh_CN/rl_tasks/robots/go2/assets/assets/calf.stl |
| docs/blog/zh_CN/rl_tasks/robots/go2/assets/assets/calf_0.obj |
| docs/blog/zh_CN/rl_tasks/robots/go2/assets/assets/calf_1.obj |
| docs/blog/zh_CN/rl_tasks/robots/go2/assets/assets/calf_mirror.stl |
| docs/blog/zh_CN/rl_tasks/robots/go2/assets/assets/calf_mirror_0.obj |
| docs/blog/zh_CN/rl_tasks/robots/go2/assets/assets/calf_mirror_1.obj |
| docs/blog/zh_CN/rl_tasks/robots/go2/assets/assets/foot.obj |
| docs/blog/zh_CN/rl_tasks/robots/go2/assets/assets/hip_0.obj |
| docs/blog/zh_CN/rl_tasks/robots/go2/assets/assets/hip_1.obj |
| docs/blog/zh_CN/rl_tasks/robots/go2/assets/assets/thigh_0.obj |
| docs/blog/zh_CN/rl_tasks/robots/go2/assets/assets/thigh_1.obj |
| docs/blog/zh_CN/rl_tasks/robots/go2/assets/assets/thigh_mirror_0.obj |
| docs/blog/zh_CN/rl_tasks/robots/go2/assets/assets/thigh_mirror_1.obj |
| docs/blog/zh_CN/rl_tasks/robots/go2/assets/assets/wheel.stl |
| docs/blog/zh_CN/rl_tasks/robots/go2/assets/assets/wheel_convex.stl |
| docs/blog/zh_CN/rl_tasks/robots/go2/assets/go2.xml |
| docs/blog/zh_CN/rl_tasks/robots/go2/assets/go2_mujoco.xml |
| docs/blog/zh_CN/rl_tasks/robots/go2/assets/locomotion_task.xml |
| docs/blog/zh_CN/rl_tasks/robots/go2/assets/scene_flat.xml |

</details>

关键 XML 片段如下。`<include>` 把纯机器人 XML 放入 scene，`<keyframe>` 留在 scene 或 task fragment 中，符合 UniLab 的资产分层约束。

```xml
<!-- src/unilab/assets/robots/go2/scene_flat.xml -->
<mujoco model="go2 scene">
  <include file="go2.xml"/>

  <statistic center="0 0 0.1" extent="0.8"/>

```

## 关节说明

`free` joint 表示浮动基座或任务物体自由度，不直接对应 policy action；`controlled=是` 表示该 joint 被 actuator 直接驱动，是 action 空间的一部分。“中文含义”按机器人运动学命名拆解，用于快速理解每个关节控制的身体部位和运动方向。

| ID | 关节名称 | 中文含义 | 类型 | 有限位 | 关节限制 | 受控 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | <unnamed:0> | 机器人浮动基座自由关节 | free | 否 | 无 | 否 |
| 1 | FL_hip_joint | 左前腿髋外展/内收关节 | hinge | 是 | -1.047 ~ 1.047 | 是 |
| 2 | FL_thigh_joint | 左前腿髋俯仰（大腿）关节 | hinge | 是 | -1.571 ~ 3.491 | 是 |
| 3 | FL_calf_joint | 左前腿膝关节（小腿摆动） | hinge | 是 | -2.723 ~ -0.8378 | 是 |
| 4 | FR_hip_joint | 右前腿髋外展/内收关节 | hinge | 是 | -1.047 ~ 1.047 | 是 |
| 5 | FR_thigh_joint | 右前腿髋俯仰（大腿）关节 | hinge | 是 | -1.571 ~ 3.491 | 是 |
| 6 | FR_calf_joint | 右前腿膝关节（小腿摆动） | hinge | 是 | -2.723 ~ -0.8378 | 是 |
| 7 | RL_hip_joint | 左后腿髋外展/内收关节 | hinge | 是 | -1.047 ~ 1.047 | 是 |
| 8 | RL_thigh_joint | 左后腿髋俯仰（大腿）关节 | hinge | 是 | -0.5236 ~ 4.538 | 是 |
| 9 | RL_calf_joint | 左后腿膝关节（小腿摆动） | hinge | 是 | -2.723 ~ -0.8378 | 是 |
| 10 | RR_hip_joint | 右后腿髋外展/内收关节 | hinge | 是 | -1.047 ~ 1.047 | 是 |
| 11 | RR_thigh_joint | 右后腿髋俯仰（大腿）关节 | hinge | 是 | -0.5236 ~ 4.538 | 是 |
| 12 | RR_calf_joint | 右后腿膝关节（小腿摆动） | hinge | 是 | -2.723 ~ -0.8378 | 是 |

## 执行器说明

执行器表来自 MuJoCo 编译模型的 `actuator` 段。RL action 的每一维最终会映射到这些执行器的控制目标或控制范围。

| ID | 执行器名称 | 驱动关节 | 中文含义 | 控制范围 |
| --- | --- | --- | --- | --- |
| 0 | FR_hip | FR_hip_joint | 右前腿髋外展/内收关节 | 未显式限制 |
| 1 | FR_thigh | FR_thigh_joint | 右前腿髋俯仰（大腿）关节 | 未显式限制 |
| 2 | FR_calf | FR_calf_joint | 右前腿膝关节（小腿摆动） | 未显式限制 |
| 3 | FL_hip | FL_hip_joint | 左前腿髋外展/内收关节 | 未显式限制 |
| 4 | FL_thigh | FL_thigh_joint | 左前腿髋俯仰（大腿）关节 | 未显式限制 |
| 5 | FL_calf | FL_calf_joint | 左前腿膝关节（小腿摆动） | 未显式限制 |
| 6 | RR_hip | RR_hip_joint | 右后腿髋外展/内收关节 | 未显式限制 |
| 7 | RR_thigh | RR_thigh_joint | 右后腿髋俯仰（大腿）关节 | 未显式限制 |
| 8 | RR_calf | RR_calf_joint | 右后腿膝关节（小腿摆动） | 未显式限制 |
| 9 | RL_hip | RL_hip_joint | 左后腿髋外展/内收关节 | 未显式限制 |
| 10 | RL_thigh | RL_thigh_joint | 左后腿髋俯仰（大腿）关节 | 未显式限制 |
| 11 | RL_calf | RL_calf_joint | 左后腿膝关节（小腿摆动） | 未显式限制 |

## 传感器说明

传感器既包括机器人本体 IMU/速度传感器，也可能包括 scene/task fragment 增加的接触传感器。任务文档会说明哪些传感器进入 obs 或 reward。

| ID | 传感器名称 | 维度 |
| --- | --- | --- |
| 0 | gyro | 3 |
| 1 | accelerometer | 3 |
| 2 | local_linvel | 3 |
| 3 | upvector | 3 |
| 4 | global_position | 3 |
| 5 | global_linvel | 3 |
| 6 | global_angvel | 3 |
| 7 | FR_pos | 3 |
| 8 | FL_pos | 3 |
| 9 | RR_pos | 3 |
| 10 | RL_pos | 3 |
| 11 | FR_vel | 3 |
| 12 | FL_vel | 3 |
| 13 | RR_vel | 3 |
| 14 | RL_vel | 3 |
| 15 | FL_foot_contact | 1 |
| 16 | FR_foot_contact | 1 |
| 17 | RL_foot_contact | 1 |
| 18 | RR_foot_contact | 1 |
| 19 | base1_contact | 1 |
| 20 | base2_contact | 1 |
| 21 | base3_contact | 1 |
| 22 | FL_hip_contact | 1 |
| 23 | FR_hip_contact | 1 |
| 24 | RL_hip_contact | 1 |
| 25 | RR_hip_contact | 1 |
| 26 | FL_thigh_contact | 1 |
| 27 | FR_thigh_contact | 1 |
| 28 | RL_thigh_contact | 1 |
| 29 | RR_thigh_contact | 1 |
| 30 | FL_calf_contact1 | 1 |
| 31 | FR_calf_contact1 | 1 |
| 32 | RL_calf_contact1 | 1 |
| 33 | RR_calf_contact1 | 1 |
| 34 | FL_calf_contact2 | 1 |
| 35 | FR_calf_contact2 | 1 |
| 36 | RL_calf_contact2 | 1 |
| 37 | RR_calf_contact2 | 1 |

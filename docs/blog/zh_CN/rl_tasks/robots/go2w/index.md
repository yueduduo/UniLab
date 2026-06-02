---
title: "Unitree Go2W 机器人资产与关节说明"
slug: "go2w"
robot: "go2w"
source_scene: "src/unilab/assets/robots/go2w/scene_flat.xml"
---

# Unitree Go2W 机器人资产与关节说明

![Unitree Go2W 默认姿态](../../images/robots/go2w.png)

Go2W 在 Go2 四足结构上加入四个轮关节，因此 action 维度和速度跟踪约束不同于纯足式四足。文档中单独列出轮关节和轮执行器，便于区分腿部 PD 控制与轮部目标。

## 机器人定位

- 用途：轮足式 joystick 运动控制机器人
- 主场景：`src/unilab/assets/robots/go2w/scene_flat.xml`
- 默认 keyframe：`home`
- 资产快照：`robots/go2w/assets/`

这份文档只把 `src/unilab/assets/robots/go2w/` 下的机器人、场景、mesh、texture 等素材复制到博客目录。motion、checkpoint、cache 不属于机器人结构说明，未复制进文档资产目录。

## 编译模型概览

| 字段 | 数值 | 说明 |
| --- | --- | --- |
| nq | 23 | 广义坐标维度，free joint 的四元数会占 4 维 |
| nv | 22 | 广义速度维度 |
| nu | 16 | 执行器数量，也就是默认 action 维度 |
| njnt | 17 | MuJoCo 编译后的 joint 数，包含 free/object joint |
| nsensor | 55 | 传感器数量 |
| keyframes | home | scene/task 层 keyframe |

## 资产结构

<details>
<summary>已复制文件（默认折叠，点击展开）</summary>

| 已复制文件 |
| --- |
| docs/blog/zh_CN/rl_tasks/robots/go2w/assets/go2w.xml |
| docs/blog/zh_CN/rl_tasks/robots/go2w/assets/go2w_mujoco.xml |
| docs/blog/zh_CN/rl_tasks/robots/go2w/assets/locomotion_task.xml |
| docs/blog/zh_CN/rl_tasks/robots/go2w/assets/scene_flat.xml |

</details>

关键 XML 片段如下。`<include>` 把纯机器人 XML 放入 scene，`<keyframe>` 留在 scene 或 task fragment 中，符合 UniLab 的资产分层约束。

```xml
<!-- src/unilab/assets/robots/go2w/scene_flat.xml -->
<mujoco model="go2w scene">
  <include file="go2w.xml"/>

  <!-- <statistic center="0 0 0.1" extent="0.8"/> -->

```

## 关节说明

`free` joint 表示浮动基座或任务物体自由度，不直接对应 policy action；`controlled=是` 表示该 joint 被 actuator 直接驱动，是 action 空间的一部分。“中文含义”按机器人运动学命名拆解，用于快速理解每个关节控制的身体部位和运动方向。

| ID | 关节名称 | 中文含义 | 类型 | 有限位 | 关节限制 | 受控 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | <unnamed:0> | 机器人浮动基座自由关节 | free | 否 | 无 | 否 |
| 1 | FL_hip_joint | 左前腿髋外展/内收关节 | hinge | 是 | -1.047 ~ 1.047 | 是 |
| 2 | FL_thigh_joint | 左前腿髋俯仰（大腿）关节 | hinge | 是 | -1.571 ~ 3.491 | 是 |
| 3 | FL_calf_joint | 左前腿膝关节（小腿摆动） | hinge | 是 | -2.723 ~ -0.8378 | 是 |
| 4 | FL_wheel_joint | 左前腿轮子滚转关节 | hinge | 否 | 无 | 是 |
| 5 | FR_hip_joint | 右前腿髋外展/内收关节 | hinge | 是 | -1.047 ~ 1.047 | 是 |
| 6 | FR_thigh_joint | 右前腿髋俯仰（大腿）关节 | hinge | 是 | -1.571 ~ 3.491 | 是 |
| 7 | FR_calf_joint | 右前腿膝关节（小腿摆动） | hinge | 是 | -2.723 ~ -0.8378 | 是 |
| 8 | FR_wheel_joint | 右前腿轮子滚转关节 | hinge | 否 | 无 | 是 |
| 9 | RL_hip_joint | 左后腿髋外展/内收关节 | hinge | 是 | -1.047 ~ 1.047 | 是 |
| 10 | RL_thigh_joint | 左后腿髋俯仰（大腿）关节 | hinge | 是 | -0.5236 ~ 4.538 | 是 |
| 11 | RL_calf_joint | 左后腿膝关节（小腿摆动） | hinge | 是 | -2.723 ~ -0.8378 | 是 |
| 12 | RL_wheel_joint | 左后腿轮子滚转关节 | hinge | 否 | 无 | 是 |
| 13 | RR_hip_joint | 右后腿髋外展/内收关节 | hinge | 是 | -1.047 ~ 1.047 | 是 |
| 14 | RR_thigh_joint | 右后腿髋俯仰（大腿）关节 | hinge | 是 | -0.5236 ~ 4.538 | 是 |
| 15 | RR_calf_joint | 右后腿膝关节（小腿摆动） | hinge | 是 | -2.723 ~ -0.8378 | 是 |
| 16 | RR_wheel_joint | 右后腿轮子滚转关节 | hinge | 否 | 无 | 是 |

## 执行器说明

执行器表来自 MuJoCo 编译模型的 `actuator` 段。RL action 的每一维最终会映射到这些执行器的控制目标或控制范围。

| ID | 执行器名称 | 驱动关节 | 中文含义 | 控制范围 |
| --- | --- | --- | --- | --- |
| 0 | FR_hip | FR_hip_joint | 右前腿髋外展/内收关节 | -23.7 ~ 23.7 |
| 1 | FR_thigh | FR_thigh_joint | 右前腿髋俯仰（大腿）关节 | -23.7 ~ 23.7 |
| 2 | FR_calf | FR_calf_joint | 右前腿膝关节（小腿摆动） | -45.43 ~ 45.43 |
| 3 | FL_hip | FL_hip_joint | 左前腿髋外展/内收关节 | -23.7 ~ 23.7 |
| 4 | FL_thigh | FL_thigh_joint | 左前腿髋俯仰（大腿）关节 | -23.7 ~ 23.7 |
| 5 | FL_calf | FL_calf_joint | 左前腿膝关节（小腿摆动） | -45.43 ~ 45.43 |
| 6 | RR_hip | RR_hip_joint | 右后腿髋外展/内收关节 | -23.7 ~ 23.7 |
| 7 | RR_thigh | RR_thigh_joint | 右后腿髋俯仰（大腿）关节 | -23.7 ~ 23.7 |
| 8 | RR_calf | RR_calf_joint | 右后腿膝关节（小腿摆动） | -45.43 ~ 45.43 |
| 9 | RL_hip | RL_hip_joint | 左后腿髋外展/内收关节 | -23.7 ~ 23.7 |
| 10 | RL_thigh | RL_thigh_joint | 左后腿髋俯仰（大腿）关节 | -23.7 ~ 23.7 |
| 11 | RL_calf | RL_calf_joint | 左后腿膝关节（小腿摆动） | -45.43 ~ 45.43 |
| 12 | FR_wheel | FR_wheel_joint | 右前腿轮子滚转关节 | -15 ~ 15 |
| 13 | FL_wheel | FL_wheel_joint | 左前腿轮子滚转关节 | -15 ~ 15 |
| 14 | RR_wheel | RR_wheel_joint | 右后腿轮子滚转关节 | -15 ~ 15 |
| 15 | RL_wheel | RL_wheel_joint | 左后腿轮子滚转关节 | -15 ~ 15 |

## 传感器说明

传感器既包括机器人本体 IMU/速度传感器，也可能包括 scene/task fragment 增加的接触传感器。任务文档会说明哪些传感器进入 obs 或 reward。

| ID | 传感器名称 | 维度 |
| --- | --- | --- |
| 0 | FR_hip_pos | 1 |
| 1 | FR_thigh_pos | 1 |
| 2 | FR_calf_pos | 1 |
| 3 | FL_hip_pos | 1 |
| 4 | FL_thigh_pos | 1 |
| 5 | FL_calf_pos | 1 |
| 6 | RR_hip_pos | 1 |
| 7 | RR_thigh_pos | 1 |
| 8 | RR_calf_pos | 1 |
| 9 | RL_hip_pos | 1 |
| 10 | RL_thigh_pos | 1 |
| 11 | RL_calf_pos | 1 |
| 12 | FR_wheel_pos | 1 |
| 13 | FL_wheel_pos | 1 |
| 14 | RR_wheel_pos | 1 |
| 15 | RL_wheel_pos | 1 |
| 16 | FR_hip_vel | 1 |
| 17 | FR_thigh_vel | 1 |
| 18 | FR_calf_vel | 1 |
| 19 | FL_hip_vel | 1 |
| 20 | FL_thigh_vel | 1 |
| 21 | FL_calf_vel | 1 |
| 22 | RR_hip_vel | 1 |
| 23 | RR_thigh_vel | 1 |
| 24 | RR_calf_vel | 1 |
| 25 | RL_hip_vel | 1 |
| 26 | RL_thigh_vel | 1 |
| 27 | RL_calf_vel | 1 |
| 28 | FR_wheel_vel | 1 |
| 29 | FL_wheel_vel | 1 |
| 30 | RR_wheel_vel | 1 |
| 31 | RL_wheel_vel | 1 |
| 32 | FR_hip_torque | 1 |
| 33 | FR_thigh_torque | 1 |
| 34 | FR_calf_torque | 1 |
| 35 | FL_hip_torque | 1 |
| 36 | FL_thigh_torque | 1 |
| 37 | FL_calf_torque | 1 |
| 38 | RR_hip_torque | 1 |
| 39 | RR_thigh_torque | 1 |
| 40 | RR_calf_torque | 1 |
| 41 | RL_hip_torque | 1 |
| 42 | RL_thigh_torque | 1 |
| 43 | RL_calf_torque | 1 |
| 44 | FR_wheel_torque | 1 |
| 45 | FL_wheel_torque | 1 |
| 46 | RR_wheel_torque | 1 |
| 47 | RL_wheel_torque | 1 |
| 48 | imu_quat | 4 |
| 49 | gyro | 3 |
| 50 | imu_acc | 3 |
| 51 | local_linvel | 3 |
| 52 | upvector | 3 |
| 53 | frame_pos | 3 |
| 54 | frame_vel | 3 |

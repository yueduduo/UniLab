#!/usr/bin/env python3
# pyright: reportAttributeAccessIssue=false
"""Generate blog-ready Chinese documentation for UniLab RL tasks.

The generated files live under ``docs/blog/zh_CN/rl_tasks``. The script is
documentation tooling only: it reads registry/YAML/XML/source files, copies
robot assets into the blog staging tree, and emits Markdown plus inventory
manifests. Existing task Markdown is protected by default because task pages may
contain hand-written code walkthroughs that should not be overwritten by the
generator.
"""

from __future__ import annotations

import argparse
import inspect
import json
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import yaml

REPO_ROOT = Path(__file__).resolve().parents[5]
DOC_ROOT = REPO_ROOT / "docs" / "blog" / "zh_CN" / "rl_tasks"
SRC_ROOT = REPO_ROOT / "src"
ROBOTS_ROOT = SRC_ROOT / "unilab" / "assets" / "robots"

sys.path.insert(0, str(SRC_ROOT))

from unilab.base import registry  # noqa: E402


@dataclass(frozen=True)
class RobotDoc:
    slug: str
    title: str
    role: str
    primary_scene: str
    keyframe: str
    intro: str


@dataclass(frozen=True)
class TaskDoc:
    slug: str
    title: str
    category: str
    robot: str
    registry_names: tuple[str, ...]
    goal: str
    agent: str
    env: str
    obs: str
    action: str
    reward: str
    init: str
    termination: str
    domain_rand: str
    scene_hint: str
    source_hint: str


ROBOTS: dict[str, RobotDoc] = {
    "go1": RobotDoc(
        "go1",
        "Unitree Go1",
        "四足速度跟踪与崎岖地形基线机器人",
        "src/unilab/assets/robots/go1/scene_flat.xml",
        "home",
        "Go1 是 UniLab 中较轻量的四足运动控制基线，主要用于平地/崎岖地形 joystick 指令跟踪。资产目录同时保留纯机器人 XML、MuJoCo 扩展 XML 和 locomotion fragment，用来展示 robot/scene/task 三层边界。",
    ),
    "go2": RobotDoc(
        "go2",
        "Unitree Go2",
        "四足运动控制、倒立与前足站立任务机器人",
        "src/unilab/assets/robots/go2/scene_flat.xml",
        "home",
        "Go2 覆盖 UniLab 中最完整的四足运动任务族：平地速度跟踪、崎岖地形、handstand 与 footstand。它的 12 个可控关节与足端接触传感器构成了 locomotion reward 和终止条件的主要观测对象。",
    ),
    "go2w": RobotDoc(
        "go2w",
        "Unitree Go2W",
        "轮足式 joystick 运动控制机器人",
        "src/unilab/assets/robots/go2w/scene_flat.xml",
        "home",
        "Go2W 在 Go2 四足结构上加入四个轮关节，因此 action 维度和速度跟踪约束不同于纯足式四足。文档中单独列出轮关节和轮执行器，便于区分腿部 PD 控制与轮部目标。",
    ),
    "g1": RobotDoc(
        "g1",
        "Unitree G1",
        "人形行走与全身动作追踪机器人",
        "src/unilab/assets/robots/g1/scene_flat.xml",
        "stand",
        "G1 是 UniLab 中人形 locomotion 与 motion tracking 的核心机器人。其 29 个可控关节覆盖腿、腰和双臂，任务从速度命令行走扩展到翻转、爬箱、靠墙翻转和全身轨迹模仿。",
    ),
    "k1": RobotDoc(
        "k1",
        "Booster K1",
        "人形行走与足球盘带机器人",
        "src/unilab/assets/robots/k1/scene_flat.xml",
        "stand",
        "K1 使用 22 DoF 可控人形模型，既用于稳定行走，也用于带球盘带。足球任务在 scene 中加入球体自由关节，机器人 XML 本身仍保持纯机器人描述。",
    ),
    "go2_arm": RobotDoc(
        "go2_arm",
        "Go2 + Airbot Arm",
        "四足移动操作机器人",
        "src/unilab/assets/robots/go2_arm/scene_flat.xml",
        "home",
        "Go2Arm 将 Go2 四足底盘和 6 DoF 机械臂组合成移动操作任务资产。它的文档重点区分底盘 locomotion action 与机械臂关节 action，以及目标物体距离类 reward。",
    ),
    "allegro_hand": RobotDoc(
        "allegro_hand",
        "Allegro Hand",
        "灵巧手球体旋转与抓取生成任务",
        "src/unilab/assets/robots/allegro_hand/scene.xml",
        "home",
        "Allegro Hand 是 16 DoF 灵巧手，任务场景中 include 手本体和球体资产，用于手内旋转控制与抓取初始化生成。关节表按手指列出，可直接对应 action 维度。",
    ),
    "sharpa_wave": RobotDoc(
        "sharpa_wave",
        "Sharpa Wave Hand",
        "触觉灵巧手旋转与抓取任务",
        "src/unilab/assets/robots/sharpa_wave/scene.xml",
        "home",
        "Sharpa Wave 是带触觉/接触建模的 22 DoF 灵巧手，场景中包含被操作圆柱物体。文档会把手部执行器、物体自由关节和接触传感器分开说明。",
    ),
}


TASKS: dict[str, TaskDoc] = {
    "go1_joystick_flat": TaskDoc(
        "go1_joystick_flat",
        "Go1 Joystick Flat",
        "locomotion",
        "go1",
        ("Go1JoystickFlat",),
        "让 Go1 在平地上跟随 joystick 线速度与角速度命令，同时保持身体姿态、足端接触节律和动作平滑。",
        "Agent 输出 12 维动作，动作经 `action_scale` 缩放后叠加到 keyframe 默认关节角，作为 PD 位置目标。",
        "Env 由 `Go1WalkTask` 承载，遵守 `NpEnv` dict obs contract，并通过 registry 暴露 MuJoCo/Motrix 后端。",
        "观测围绕本体局部速度、角速度、重力方向、命令、关节位置/速度和上一帧动作构造；critic 额外读取特权速度信息。",
        "动作对应四条腿的 hip/thigh/calf actuator；XML 编译得到 `nu=12`。",
        "奖励由 locomotion 公共 reward dispatcher 和 owner YAML 的 `reward.scales` 组合，强调速度跟踪、姿态、足端节律与能耗惩罚。",
        "reset 读取 `home` keyframe，并可叠加 base 速度、关节和物理参数随机化。",
        "主要终止来自身体高度过低、倾斜过大和 episode horizon。",
        "域随机化集中在质量/COM/重力/摩擦/推力/PD gain 等 locomotion 冷路径配置。",
        "src/unilab/assets/robots/go1/scene_flat.xml",
        "src/unilab/envs/locomotion/go1/joystick.py",
    ),
    "go1_joystick_rough": TaskDoc(
        "go1_joystick_rough",
        "Go1 Joystick Rough",
        "locomotion",
        "go1",
        ("Go1JoystickRough",),
        "让 Go1 在崎岖地形上保持 joystick 跟踪能力，并适应高度扫描、地形 spawn 与足端接触扰动。",
        "Agent 与平地任务一致输出 12 维关节目标，但训练信号加入崎岖地形下的稳定性约束。",
        "Env 使用 rough 配置和 scene fragment，把地形/接触传感器留在任务层而不是机器人 XML。",
        "观测在基础 locomotion 观测外引入地形高度扫描或地形相关特权信息，critic 维度随扫描网格变化。",
        "动作仍对应 Go1 12 个腿部执行器，控制语义不因地形变化而改变。",
        "reward 在速度跟踪之外提高姿态、足端拖拽、接触稳定性和高度约束的权重。",
        "reset 通过地形 spawn manager 放置机器人，并从 `home` 姿态开始。",
        "终止条件除倾倒外，还关注崎岖地形相对高度导致的跌落。",
        "域随机化包含地形、摩擦、质量、COM、推力和控制延迟等鲁棒训练项。",
        "src/unilab/assets/robots/go1/scene_flat.xml",
        "src/unilab/envs/locomotion/go1/rough.py",
    ),
    "go2_joystick_flat": TaskDoc(
        "go2_joystick_flat",
        "Go2 Joystick Flat",
        "locomotion",
        "go2",
        ("Go2JoystickFlat",),
        "训练 Go2 在平地上跟踪 x/y/yaw 速度命令，形成可迁移到更复杂 Go2 任务的基础步态。",
        "Agent 输出 12 维腿部目标，通常由 PPO/APPO/FlashSAC/TD3 等算法消费 actor obs。",
        "Env 使用 `Go2WalkTask`，其 reset、action space 和 obs 由 locomotion base 与 Go2 专用配置组合。",
        "actor obs 包含命令、本体姿态、关节误差、关节速度和动作历史；critic 读取额外 base linvel。",
        "动作对应 FR/FL/RR/RL 的 hip、thigh、calf 三个关节。",
        "reward 以 tracking_lin_vel/tracking_ang_vel 为核心，辅以姿态、动作变化、足端接触和能耗项。",
        "初始状态来自 Go2 scene 的 `home` keyframe，默认关节角也由该 keyframe 派生。",
        "身体高度、倾斜角和 episode timeout 构成主要终止面。",
        "domain_rand 可随机摩擦、质量、COM、重力、推力、kp/kd 与初始关节扰动。",
        "src/unilab/assets/robots/go2/scene_flat.xml",
        "src/unilab/envs/locomotion/go2/joystick.py",
    ),
    "go2_joystick_rough": TaskDoc(
        "go2_joystick_rough",
        "Go2 Joystick Rough",
        "locomotion",
        "go2",
        ("Go2JoystickRough",),
        "在崎岖地形上训练 Go2 保持速度命令跟踪，并学习利用足端接触和地形高度信息稳定前进。",
        "Agent 仍输出 12 维腿部目标；算法层不直接感知 backend 私有能力，而是通过 env obs 接收地形信息。",
        "Env 使用 rough 任务配置、terrain spawn 和 locomotion task fragment。",
        "观测在基础 Go2 观测外加入高度扫描；critic 通常保留更多特权地形/速度信息。",
        "动作空间不变，rough 任务只改变观测、场景和奖励约束。",
        "reward 增加 foot_drag、接触与粗糙地形下的姿态稳定惩罚。",
        "reset 在地形上采样 spawn 点，并应用 `home` keyframe 的默认姿态。",
        "终止关注地形相对 base 高度、过大倾斜和异常接触。",
        "域随机化覆盖 terrain、摩擦、质量、COM、推力与控制参数。",
        "src/unilab/assets/robots/go2/scene_flat.xml",
        "src/unilab/envs/locomotion/go2/rough.py",
    ),
    "go2_handstand": TaskDoc(
        "go2_handstand",
        "Go2 HandStand",
        "locomotion",
        "go2",
        ("Go2HandStand",),
        "训练 Go2 从常规四足姿态进入并维持倒立/手倒立式站立，使身体高度、姿态和接触模式满足 handstand 目标。",
        "Agent 输出 Go2 12 维关节目标，策略重点学习非典型支撑相下的姿态调节。",
        "Env 复用 Go2 locomotion backend，但任务类提供 handstand 专用 obs/reward/termination。",
        "观测更强调姿态、重力投影、关节状态和动作历史，弱化 joystick 命令。",
        "动作仍是 12 个腿部 actuator 的位置目标。",
        "reward 包含高度、姿态、支撑接触、动作平滑和能耗等专用项。",
        "初始状态从 `home` keyframe 开始，训练中通过动作序列学习转入目标姿态。",
        "终止由高度、倾斜、接触失败或超时触发。",
        "domain_rand 使用 Go2 物理参数扰动提升非典型姿态下的鲁棒性。",
        "src/unilab/assets/robots/go2/scene_flat.xml",
        "src/unilab/envs/locomotion/go2/handstand.py",
    ),
    "go2_footstand": TaskDoc(
        "go2_footstand",
        "Go2 FootStand",
        "locomotion",
        "go2",
        ("Go2FootStand",),
        "训练 Go2 以前足支撑姿态站立，保持指定身体高度、姿态稳定和后腿接触/离地模式。",
        "Agent 输出 12 维关节目标；owner YAML 使用较长 obs history 强化姿态控制记忆。",
        "Env 使用 `Go2FootStandTask`，是 MuJoCo-only 的足式站立专用任务。",
        "观测堆叠历史帧，包含姿态、角速度、关节误差、关节速度、动作历史和特权信息。",
        "动作经 `action_scale` 叠加到默认角度，控制 Go2 全部 12 个腿部 actuator。",
        "reward 明确包含 height/orientation/contact/tar/rear_feet_contact/energy 等项，约束站立高度、稳定性和能耗。",
        "初始状态来自 Go2 `home` keyframe，reset 时可随机关节 qpos、质量和地面摩擦。",
        "终止包括姿态失败、接触失败、能量阈值和超时。",
        "domain_rand 在 owner YAML 中启用 floor friction、link mass、torso COM、armature 和 reset joint qpos 随机化。",
        "src/unilab/assets/robots/go2/scene_flat.xml",
        "src/unilab/envs/locomotion/go2/footstand.py",
    ),
    "go2w_joystick_flat": TaskDoc(
        "go2w_joystick_flat",
        "Go2W Joystick Flat",
        "locomotion",
        "go2w",
        ("Go2WJoystickFlat",),
        "训练轮足 Go2W 在平地上利用腿部和轮部共同跟踪 joystick 速度命令。",
        "Agent 输出 16 维动作，对应 12 个腿关节和 4 个轮关节。",
        "Env 使用 Go2W 专用 motor control helper，把轮足控制封装在 env 层而不是训练脚本。",
        "观测在 Go2 基础上加入轮关节状态和轮足特定动作历史。",
        "动作空间包含腿部位置目标与轮部目标，XML 编译 `nu=16`。",
        "reward 以速度跟踪为核心，同时惩罚轮足不稳定、动作变化和姿态偏差。",
        "初始姿态来自 Go2W `home` keyframe。",
        "终止关注 base 高度、倾斜和轮足异常状态。",
        "domain_rand 与四足任务类似，但作用到轮足模型的质量、摩擦、推力和控制参数。",
        "src/unilab/assets/robots/go2w/scene_flat.xml",
        "src/unilab/envs/locomotion/go2w/joystick.py",
    ),
    "go2w_joystick_rough": TaskDoc(
        "go2w_joystick_rough",
        "Go2W Joystick Rough",
        "locomotion",
        "go2w",
        ("Go2WJoystickRough",),
        "让 Go2W 在崎岖地形上融合轮部推进与腿部支撑，保持速度命令跟踪和通过性。",
        "Agent 输出 16 维轮足动作，策略需要在地形扰动下协调轮和腿。",
        "Env 使用 rough 场景/fragment 与 Go2W 专用控制逻辑。",
        "观测包含轮足状态、命令、IMU、关节误差、动作历史和地形相关信息。",
        "动作空间保持 16 维，rough 任务主要改变地形观测和奖励。",
        "reward 加强地形稳定、接触、拖拽和姿态项。",
        "reset 在地形 spawn 点应用 `home` keyframe。",
        "终止由跌倒、倾斜、地形相对高度异常和超时触发。",
        "domain_rand 覆盖地形、摩擦、质量、COM、推力和控制参数。",
        "src/unilab/assets/robots/go2w/scene_flat.xml",
        "src/unilab/envs/locomotion/go2w/rough.py",
    ),
    "g1_walk_flat": TaskDoc(
        "g1_walk_flat",
        "G1 Walk Flat",
        "locomotion",
        "g1",
        ("G1WalkFlat",),
        "训练 G1 人形机器人在平地上跟踪前进/横移/转向命令，并保持躯干高度、脚步相位和上肢姿态。",
        "Agent 输出 29 维关节目标，覆盖双腿、腰和双臂。",
        "Env 使用 `G1WalkEnv`，在 `G1BaseEnv` 上增加 gait phase、命令采样和 locomotion reward。",
        "观测包含命令、本体速度/角速度、重力、关节误差/速度、动作历史和 gait phase；critic 额外使用 base linvel。",
        "动作维度由 G1 XML actuator 数决定，scene 编译得到 `nu=29`。",
        "reward 使用 `G1RewardConfig`，包括速度跟踪、base height、feet phase、pose、动作变化和限位惩罚。",
        "初始状态来自 `stand` keyframe，并可采样 gait phase 与 base qvel。",
        "终止通过最小 base 高度和最大倾斜角控制。",
        "domain_rand 可随机 kp/kd、质量、COM、重力、摩擦、推力和 reset joint。",
        "src/unilab/assets/robots/g1/scene_flat.xml",
        "src/unilab/envs/locomotion/g1/joystick.py",
    ),
    "g1_walk_rough": TaskDoc(
        "g1_walk_rough",
        "G1 Walk Rough",
        "locomotion",
        "g1",
        ("G1WalkRough",),
        "训练 G1 在崎岖地形上保持人形行走命令跟踪，重点处理地形相对高度与跌倒风险。",
        "Agent 仍输出 29 维全身关节目标，算法配置主要来自 offpolicy SAC owner。",
        "Env 使用 G1 rough cfg，在平地行走 env 上切换地形场景和终止高度口径。",
        "观测保留 gait phase 与本体状态，并根据 rough 配置提供地形或特权信息。",
        "动作空间与 G1 平地一致。",
        "reward 沿用 G1 locomotion 项，同时在 rough owner YAML 中调整权重。",
        "reset 使用 `stand` keyframe 并在地形上采样初始位置。",
        "终止使用 terrain-relative base height 与倾斜阈值。",
        "domain_rand 覆盖地形、物理参数、推力和控制参数。",
        "src/unilab/assets/robots/g1/scene_rough.xml",
        "src/unilab/envs/locomotion/g1/joystick.py",
    ),
    "k1_walk_flat": TaskDoc(
        "k1_walk_flat",
        "K1 Walk Flat",
        "locomotion",
        "k1",
        ("K1WalkFlat",),
        "训练 K1 人形机器人在平地上跟踪速度命令，同时保持稳定步态和上身姿态。",
        "Agent 输出 22 维关节目标，覆盖头、双臂和双腿可控关节。",
        "Env 使用 `K1WalkEnv` 与 K1 专用 observation constants，采用多帧 history。",
        "观测单帧由命令、IMU、关节状态、动作历史和 gait phase 等组成，再按 history 堆叠。",
        "动作维度由 K1 actuator 数给出，scene 编译得到 `nu=22`。",
        "reward 包含速度跟踪、base height、feet phase、pose、close feet、动作平滑和终止惩罚。",
        "初始状态来自 `stand` keyframe，reset 可扰动 base qvel 和 gait phase。",
        "终止关注 base 高度、倾斜角和超时。",
        "domain_rand 可随机质量、COM、摩擦、重力、推力和控制参数。",
        "src/unilab/assets/robots/k1/scene_flat.xml",
        "src/unilab/envs/locomotion/k1/joystick.py",
    ),
    "k1_soccer_dribble": TaskDoc(
        "k1_soccer_dribble",
        "K1 Soccer Dribble",
        "locomotion",
        "k1",
        ("K1SoccerDribble",),
        "训练 K1 接近足球、保持球在前方并按照目标方向盘带前进。",
        "Agent 输出 K1 22 维关节目标，策略需要同时兼顾自身稳定和球相对位姿。",
        "Env 在 K1 locomotion 基础上加入 ball body/free joint 和球相关 reset/reward。",
        "观测在 K1 行走观测基础上增加球相对位置/速度等信息。",
        "动作仍只控制机器人 22 个 actuator；球由物理接触自然演化。",
        "reward 包含 ball_progress、ball_keep、ball_front、ball_speed_match、ball_lost 等足球专用项。",
        "reset 将机器人放到 `stand` keyframe，并随机/设置球位置和朝球命令。",
        "终止包括机器人跌倒、球丢失距离过大和超时。",
        "domain_rand 可控制机器人物理随机化、球相关初始化扰动和噪声。",
        "src/unilab/assets/robots/k1/scene_soccer_dribble_minimal.xml",
        "src/unilab/envs/locomotion/k1/soccer_dribble.py",
    ),
    "go2_arm_manip_loco": TaskDoc(
        "go2_arm_manip_loco",
        "Go2 Arm Manip-Loco",
        "manip_loco",
        "go2_arm",
        ("Go2ArmManipLoco",),
        "训练带机械臂的 Go2 在移动中接近/操作目标，同时保持底盘稳定和机械臂安全。",
        "Agent 输出 18 维动作，包括 12 个腿部 actuator 与 6 个机械臂关节 actuator。",
        "Env 由 `Go2ArmManipLocoEnv` 管理底盘 locomotion 和 arm/object reward，不把任务规则放到训练脚本。",
        "观测包含底盘状态、命令、机械臂关节状态、目标/物体相对信息和动作历史。",
        "动作空间覆盖底盘与机械臂，scene 编译得到 `nu=18`。",
        "reward 结合 locomotion 跟踪项、object distance、arm collision、动作平滑和姿态稳定项。",
        "初始状态来自 `home` keyframe，机械臂和底盘默认姿态一起复位。",
        "终止关注底盘跌倒、机械臂碰撞/异常和 episode horizon。",
        "domain_rand 覆盖底盘物理参数、摩擦、推力以及移动操作相关扰动。",
        "src/unilab/assets/robots/go2_arm/scene_flat.xml",
        "src/unilab/envs/locomotion/go2_arm/manip_loco.py",
    ),
    "g1_motion_tracking": TaskDoc(
        "g1_motion_tracking",
        "G1 Motion Tracking",
        "motion_tracking",
        "g1",
        ("G1MotionTracking", "G1MotionTrackingSAC"),
        "让 G1 全身跟踪参考 motion clip，在 root、body、末端和关节层面逼近离线动作数据。",
        "Agent 输出 G1 29 维动作；PPO/APPO/SAC 变体共享 motion tracking 目标，但 critic/算法配置不同。",
        "Env 使用 `MotionLoader` 读取 npz motion，并在 reset 时采样 clip 时间。",
        "观测围绕当前机器人状态、参考 motion 相位/目标和动作历史构造；SAC 变体会额外给 critic base linvel。",
        "动作控制 G1 29 个 actuator，通过 action_scale 映射到关节目标。",
        "reward 包含 root pos/ori、body pos/ori、body velocity、joint pos/vel、action_rate 和 joint_limit 等 tracking 项。",
        "初始状态从 motion reference 构造，并叠加 pose/velocity/joint 随机化。",
        "终止由 anchor z/ori 误差、末端 z 误差、undesired contact 和 clip 截断策略决定。",
        "domain_rand 支持 base mass、COM、gravity、push、kp/kd、friction 和 joint default pos 扰动。",
        "src/unilab/assets/robots/g1/scene_flat.xml",
        "src/unilab/envs/motion_tracking/g1/tracking.py",
    ),
    "g1_motion_tracking_deploy": TaskDoc(
        "g1_motion_tracking_deploy",
        "G1 Motion Tracking Deploy",
        "motion_tracking",
        "g1",
        ("G1MotionTrackingDeploy",),
        "为部署/评估口径整理 G1 motion tracking 观测，使策略输入更接近实际部署所需信息。",
        "Agent 输出 29 维动作，但 deploy env 使用更收敛的 actor 观测组织。",
        "Env 继承 motion tracking 流程，重点调整 obs 组合和部署相关接口。",
        "观测减少训练期特权信息，保留部署可获得的本体状态、参考信号和动作历史。",
        "动作空间与 G1 motion tracking 一致。",
        "reward 和终止逻辑沿用 motion tracking 主任务。",
        "reset 仍从 motion reference 初始化，并应用同类随机化。",
        "终止遵循 anchor/末端/接触误差阈值。",
        "domain_rand 与 motion tracking cfg 一致，实际启用项由 owner YAML 控制。",
        "src/unilab/assets/robots/g1/scene_flat.xml",
        "src/unilab/envs/motion_tracking/g1/tracking.py",
    ),
    "g1_flip_tracking": TaskDoc(
        "g1_flip_tracking",
        "G1 Flip Tracking",
        "motion_tracking",
        "g1",
        ("G1FlipTracking", "G1FlipTrackingSAC"),
        "训练 G1 跟踪空翻 motion clip，在大姿态变化中保持 root/body/关节轨迹同步。",
        "Agent 输出 29 维动作，策略要在翻转期间处理高速姿态变化。",
        "Env 是 motion tracking 的 flip 专用配置，默认 motion file 指向 flip 轨迹。",
        "观测结构继承 tracking，但参考 motion 的相位和目标姿态更剧烈。",
        "动作空间与 G1 29 DoF tracking 一致。",
        "reward 沿用 tracking 项，并通过 flip owner YAML 调整各误差项权重。",
        "reset 从 flip clip 采样参考状态，支持 adaptive/mixed sampling。",
        "终止阈值相对普通 tracking 放宽以容纳翻转姿态。",
        "domain_rand 仍来自 tracking cfg，具体启用项由 PPO/APPO/SAC owner 决定。",
        "src/unilab/assets/robots/g1/scene_flat.xml",
        "src/unilab/envs/motion_tracking/g1/flip_tracking.py",
    ),
    "g1_wall_flip_tracking": TaskDoc(
        "g1_wall_flip_tracking",
        "G1 Wall Flip Tracking",
        "motion_tracking",
        "g1",
        ("G1WallFlipTracking", "G1WallFlipTrackingSAC"),
        "训练 G1 跟踪靠墙翻转动作，在场景中利用 wall 几何与参考轨迹完成翻转。",
        "Agent 输出 29 维动作，需协调与墙体交互时的全身姿态。",
        "Env 继承 flip tracking，并切换到带墙场景和 wall flip motion。",
        "观测沿用 motion tracking 结构，参考 motion 提供靠墙动作目标。",
        "动作空间不变。",
        "reward 仍以 tracking 误差为主，并根据 owner YAML 调整 undesired contact 等项。",
        "reset 从 wall flip clip 初始化，场景包含墙体几何。",
        "终止由 anchor/末端误差、接触和超时控制。",
        "domain_rand 与 tracking 系列一致。",
        "src/unilab/assets/robots/g1/scene_flat_with_wall.xml",
        "src/unilab/envs/motion_tracking/g1/flip_tracking.py",
    ),
    "g1_climb_tracking": TaskDoc(
        "g1_climb_tracking",
        "G1 Climb Tracking",
        "motion_tracking",
        "g1",
        ("G1ClimbTracking",),
        "训练 G1 跟踪爬箱/攀爬 motion，在包含台阶/箱体的 scene 中完成参考动作。",
        "Agent 输出 29 维动作，策略目标是跟踪爬升阶段的全身轨迹。",
        "Env 使用 climb scene 和 climb motion file，仍复用 motion tracking 框架。",
        "观测为机器人状态、参考 motion 与动作历史的组合。",
        "动作空间与 G1 tracking 一致。",
        "reward 以 root/body/joint tracking 误差为核心。",
        "reset 从 climb motion 采样初始帧，episode 秒数通常更长。",
        "终止由 tracking 误差、接触约束和超时触发。",
        "domain_rand 使用 tracking cfg 中的物理扰动开关。",
        "src/unilab/assets/robots/g1/scene_climb_20_z_scale_1.xml",
        "src/unilab/envs/motion_tracking/g1/flip_tracking.py",
    ),
    "g1_box_tracking": TaskDoc(
        "g1_box_tracking",
        "G1 Box Tracking",
        "motion_tracking",
        "g1",
        ("G1BoxTracking",),
        "训练 G1 在全身 motion tracking 中同时处理大箱体物体状态，使人体动作和物体轨迹共同满足参考。",
        "Agent 输出 29 维动作；物体状态进入 obs/reward，但不由 action 直接控制。",
        "Env 使用 `G1BoxTrackingEnv` 和 `BoxMotionLoader`，场景包含 largebox。",
        "观测在 tracking 基础上加入物体位置/姿态/速度相关信息。",
        "动作空间仍为 G1 关节 actuator。",
        "reward 除 tracking 项外增加 object pos/ori 等物体误差项。",
        "reset 从 box motion 初始化机器人与物体状态。",
        "终止增加物体位置/姿态误差阈值。",
        "domain_rand 与 tracking 系列一致，物体相关扰动由 box env/config 控制。",
        "src/unilab/assets/robots/g1/scene_flat_with_largebox.xml",
        "src/unilab/envs/motion_tracking/g1/box_tracking.py",
    ),
    "g1_wbt_obs": TaskDoc(
        "g1_wbt_obs",
        "G1 WBT Obs",
        "motion_tracking",
        "g1",
        ("G1WBTObs",),
        "提供 Whole-Body Tracking 观测口径的 SAC 任务，用于对齐全身跟踪训练所需的严格 obs 结构。",
        "Agent 输出 G1 29 维动作，算法侧通常使用 offpolicy SAC owner。",
        "Env 是 tracking SAC 的观测专用子类，强调 obs contract 稳定性。",
        "观测字段更严格地组织为 WBT 所需输入，critic 仍可包含特权项。",
        "动作空间与 G1 tracking 一致。",
        "reward 沿用 motion tracking SAC 配置。",
        "reset 使用 motion reference 和同类随机化。",
        "终止沿用 tracking 阈值。",
        "domain_rand 由 SAC owner YAML 控制。",
        "src/unilab/assets/robots/g1/scene_flat.xml",
        "src/unilab/envs/motion_tracking/g1/tracking_obs.py",
    ),
    "allegro_inhand": TaskDoc(
        "allegro_inhand",
        "Allegro In-Hand Rotation",
        "manipulation",
        "allegro_hand",
        ("AllegroInhandRotation",),
        "训练 Allegro Hand 在手内旋转球体，使球沿指定轴获得目标角速度，同时维持稳定抓持。",
        "Agent 输出 16 维手指关节目标，直接对应 Allegro 16 个 actuator。",
        "Env 使用 `AllegroRotationPPO`，场景 include 手本体和球体。",
        "观测包含归一化手指关节、目标控制、球位置/姿态/角速度和 lag history。",
        "动作经 PD torque/position 逻辑作用到 16 个手指关节。",
        "reward 包含 rotate、obj_linvel、pose_diff、torque、work 等项，权重来自 owner YAML。",
        "reset 可从 grasp cache 采样，也可 procedural 初始化手指和球体。",
        "终止关注球体高度/掉落阈值和 episode horizon。",
        "domain_rand 支持手指噪声、球速度噪声、球 z offset、质量/COM/重力和 push。",
        "src/unilab/assets/robots/allegro_hand/scene.xml",
        "src/unilab/envs/manipulation/allegro_inhand/rotation.py",
    ),
    "allegro_inhand_grasp": TaskDoc(
        "allegro_inhand_grasp",
        "Allegro In-Hand Grasp",
        "manipulation",
        "allegro_hand",
        ("AllegroInhandRotationGrasp",),
        "生成或训练 Allegro Hand 的抓取初始化姿态，为后续手内旋转提供稳定初态。",
        "Agent 控制 16 个手指关节，使球体保持在可旋转抓持区域。",
        "Env 继承 Allegro rotation 基础设施，但使用 grasp generation 任务逻辑。",
        "观测沿用手指状态、球状态和历史帧结构。",
        "动作空间与 Allegro rotation 一致。",
        "reward 侧重抓持稳定、球体保持和动作/力矩约束。",
        "reset 用 grasp cache 或 procedural reset 构造初始抓取。",
        "终止关注球体丢失、高度阈值和超时。",
        "domain_rand 与 Allegro rotation 共享，具体开关由 owner YAML 控制。",
        "src/unilab/assets/robots/allegro_hand/scene.xml",
        "src/unilab/envs/manipulation/allegro_inhand/grasp_gen.py",
    ),
    "sharpa_inhand": TaskDoc(
        "sharpa_inhand",
        "Sharpa In-Hand Rotation",
        "manipulation",
        "sharpa_wave",
        ("SharpaInhandRotation",),
        "训练 Sharpa Wave 灵巧手旋转圆柱物体，并利用接触/触觉相关观测保持稳定操作。",
        "Agent 输出 22 维手部 actuator 目标。",
        "Env 使用 `SharpaInhandRotationEnv`，场景内包含手本体和圆柱物体。",
        "观测由手指关节、动作历史、物体状态和可选触觉/接触信息组成。",
        "动作控制 22 个 `*_ctrl` actuator，物体自由关节由接触动力学驱动。",
        "reward 包含旋转进度、物体稳定、动作/力矩惩罚和接触相关项。",
        "reset 从默认 keyframe 或 grasp cache/初始化范围采样手和物体状态。",
        "终止关注物体掉落、姿态异常和超时。",
        "domain_rand 通过 common reset randomization 与 Sharpa 专用参数作用于初态和物理属性。",
        "src/unilab/assets/robots/sharpa_wave/scene.xml",
        "src/unilab/envs/manipulation/sharpa_inhand/rotation.py",
    ),
    "sharpa_inhand_grasp": TaskDoc(
        "sharpa_inhand_grasp",
        "Sharpa In-Hand Grasp",
        "manipulation",
        "sharpa_wave",
        ("SharpaInhandRotationGrasp",),
        "为 Sharpa Wave 生成稳定抓取状态，使后续旋转任务能从可控接触构型开始。",
        "Agent 输出 22 维手部关节目标，优化稳定接触而非高速旋转。",
        "Env 使用 Sharpa grasp generation 逻辑，复用 Sharpa base 与 scene。",
        "观测沿用 Sharpa 手部/物体状态，并强调抓取质量相关信息。",
        "动作空间与 Sharpa rotation 一致。",
        "reward 聚焦物体保持、接触稳定和动作平滑。",
        "reset 采样手指和物体初态，必要时写入/读取 grasp cache。",
        "终止关注物体掉落、过大姿态误差和超时。",
        "domain_rand 与 Sharpa rotation 共享。",
        "src/unilab/assets/robots/sharpa_wave/scene.xml",
        "src/unilab/envs/manipulation/sharpa_inhand/grasp_gen.py",
    ),
}


@dataclass(frozen=True)
class TaskCodeSection:
    title: str
    anchors: tuple[str, ...]
    notes: tuple[str, ...]
    source: str | None = None


@dataclass(frozen=True)
class TaskCodeGuide:
    focus: tuple[str, ...]
    sections: tuple[TaskCodeSection, ...]


def _sec(
    title: str,
    anchors: tuple[str, ...],
    notes: tuple[str, ...],
    source: str | None = None,
) -> TaskCodeSection:
    return TaskCodeSection(title=title, anchors=anchors, notes=notes, source=source)


TASK_CODE_GUIDES: dict[str, TaskCodeGuide] = {
    "go1_joystick_flat": TaskCodeGuide(
        focus=(
            "Go1 平地任务要读 `Go1JoystickCfg` 和 `Go1WalkTask`：配置层决定 scene/home keyframe，env 层决定 12 维腿部动作如何变成 PD 目标。",
            "这个任务没有地形高度扫描，obs 的重点是命令、IMU、关节误差、关节速度、上一帧动作和四相位步态编码。",
            "reward 重点看速度跟踪与 gait phase；终止主要由 base 高度、姿态倾斜和 episode horizon 共同决定。",
        ),
        sections=(
            _sec("配置与注册", ("Go1JoystickCfg", "Go1JoystickFlat"), ("确认 flat scene、home keyframe 和 Go1 12 DoF 控制配置都在 EnvCfg 中声明。", "registry 把 Go1JoystickFlat 绑定到 Go1WalkTask，训练脚本只消费注册名。")),
            _sec("Obs：平地步态输入", ("obs_groups_spec", "_compute_obs"), ("49 维 actor obs 由 gyro、gravity、关节误差、关节速度、动作历史、命令和 feet phase 组成。", "critic 额外使用本体线速度，平地任务不引入地形高度扫描。")),
            _sec("Action：12 维腿部目标", ("apply_action", "default_angles"), ("Go1 平地任务没有覆写 apply_action，而是继承 locomotion common base 的动作映射。", "action 与 12 个 hip/thigh/calf actuator 一一对应，缩放后叠加默认角。"), source="src/unilab/envs/locomotion/common/base.py"),
            _sec("Reward：速度与步态", ("_init_reward_functions", "tracking_lin_vel"), ("Go1 平地 reward 以 tracking_lin_vel/tracking_ang_vel 为主。", "feet_phase、orientation、action_rate 等项让速度跟踪同时满足步态节律和动作平滑。")),
            _sec("Termination 与随机化", ("update_state", "terminated", "DomainRandConfig"), ("update_state 中同时计算终止、reward 和 obs，是本任务 step 后处理的核心。", "domain randomization 来自 locomotion common 配置，作用在 reset/interval 冷路径。")),
        ),
    ),
    "go1_joystick_rough": TaskCodeGuide(
        focus=(
            "Go1 rough 不是换一个奖励名字，而是在 rough env 中显式加入 terrain、spawn、height scan 和地形相对高度判断。",
            "动作空间仍是 Go1 12 DoF，变化集中在 reset 如何落到地形、obs 如何加入地形信息、reward 如何惩罚拖脚和姿态失稳。",
            "阅读时要把 `Go1JoystickRoughCfg` 和 `Go1JoystickRoughEnv` 同平地任务对比，才能看出 rough 任务新增了哪些风险边界。",
        ),
        sections=(
            _sec("配置与地形", ("Go1JoystickRoughCfg", "Go1RoughTerrainCfg"), ("Rough cfg 声明 terrain generator、rough scene 和地形采样参数。", "这些字段属于任务层，机器人 XML 不承载地形规则。")),
            _sec("Reset：地形 spawn", ("build_reset_plan", "spawn"), ("reset 会按 terrain spawn 重新放置 base，而不是简单复用平地位置。", "qpos/qvel、commands 和 reset randomization 一起写入 ResetPlan。")),
            _sec("Obs：高度扫描", ("obs_groups_spec", "_compute_obs", "height_scan"), ("policy obs 保持紧凑，critic 扩展 height scan 特权信息。", "高度扫描只来自 env/backend 的地形查询结果，训练脚本不直接读地形资产。")),
            _sec("Action：控制语义不变", ("apply_action", "clip_actions"), ("rough 仍输出 12 维腿部位置目标。", "代码中先裁剪 action，再叠加默认角，保证崎岖地形扰动下控制边界稳定。")),
            _sec("Reward/Termination：地形风险", ("_init_reward_functions", "update_state"), ("rough reward 增加 foot_drag、upright/height 等地形稳定相关项。", "终止要看 terrain-relative base height，而不是只看世界坐标高度。")),
        ),
    ),
    "go2_joystick_flat": TaskCodeGuide(
        focus=(
            "Go2 平地任务是 Go2 任务族的基础：后续 rough、handstand、footstand 都能和它比较动作顺序和默认角。",
            "obs 中 feet phase 是步态相位核心，reward 中速度跟踪、姿态和 action_rate 共同塑造平地步态。",
            "这个任务适合从 `Go2JoystickCfg`、`Go2WalkTask._compute_obs`、`_init_reward_functions` 三处读起。",
        ),
        sections=(
            _sec("配置与注册", ("Go2JoystickCfg", "Go2JoystickFlat"), ("EnvCfg 指向 Go2 flat scene，并把 reward/control/noise 默认值留在配置层。", "registry 的 backend 绑定让 MuJoCo/Motrix 差异停留在 backend 适配层。")),
            _sec("Obs：Go2 平地观测", ("obs_groups_spec", "_compute_obs"), ("actor obs 明确注释为 49 维，包含命令、IMU、关节和相位。", "critic 通过额外速度信息帮助 value 学习，但 policy 不直接依赖特权速度。")),
            _sec("Action：三关节四腿", ("apply_action", "default_angles"), ("Go2 平地任务继承 locomotion common base 的 apply_action。", "12 维动作按 FR/FL/RR/RL 的 hip/thigh/calf 顺序控制，默认角来自 keyframe。"), source="src/unilab/envs/locomotion/common/base.py"),
            _sec("Reward：基础 locomotion", ("_init_reward_functions", "tracking_lin_vel"), ("tracking_lin_vel/tracking_ang_vel 是主任务目标。", "orientation、action_rate、feet_air_time 等项把速度命令约束成可站稳的步态。")),
            _sec("Step 后处理", ("update_state", "terminated"), ("update_state 推进相位、读取传感器、计算 done/reward/obs。", "终止主要围绕倾倒、高度和 horizon，不包含 rough 地形逻辑。")),
        ),
    ),
    "go2_joystick_rough": TaskCodeGuide(
        focus=(
            "Go2 rough 是 Go2 平地任务的地形扩展，必须单独看 `Go2JoystickRoughEnv` 中 terrain、height scan 和 rough reward。",
            "动作仍是 12 维腿部目标，策略复杂度来自 obs/reward/termination，而不是训练脚本或 action 解释变化。",
            "这个任务的代码重点是：reset 如何采样地形、critic 如何获得高度扫描、reward 如何门控姿态与足端项。",
        ),
        sections=(
            _sec("配置与地形", ("Go2JoystickRoughCfg", "Go2RoughTerrainCfg"), ("rough cfg 声明 terrain generator 和 scene fragment。", "地形属于 task scene，robot.xml 仍保持纯机器人描述。")),
            _sec("Reset：地形初态", ("build_reset_plan", "spawn"), ("reset 通过 spawn/terrain manager 选择初始 base 位置。", "ResetPlan 同时携带 qpos/qvel、命令和随机化 payload。")),
            _sec("Obs：height scan critic", ("obs_groups_spec", "_compute_obs"), ("policy obs 保持 45 维本体输入，critic 增加 base linvel 与 height scan。", "高度扫描是 rough 稳定性的关键特权信息。")),
            _sec("Action：裁剪后映射", ("apply_action", "clip_actions"), ("动作先按 clip_actions 裁剪，再缩放为关节目标。", "rough 不改变 actuator 顺序，只改变任务环境。")),
            _sec("Reward/Termination", ("_init_reward_functions", "update_state"), ("rough reward 对足端拖拽、姿态和地形高度更敏感。", "终止要结合倾斜和地形相对高度判断跌倒。")),
        ),
    ),
    "go2_handstand": TaskCodeGuide(
        focus=(
            "Handstand 不是 joystick 速度跟踪，它把 Go2 的目标从行走切到倒立/非典型支撑姿态。",
            "obs 弱化命令，重点读 gravity、gyro、关节误差和动作历史；reward 则围绕高度、朝向和接触模式。",
            "源码要看 `Go2HandStandCfg` 与 `Go2HandStandTask`，不要把它和 Go2WalkTask 的 gait phase 混在一起解释。",
        ),
        sections=(
            _sec("配置：倒立任务", ("Go2HandStandCfg", "Go2HandStand"), ("配置类选择 handstand 任务的 scene/control/reward 参数。", "任务目标是非典型姿态稳定，不是 x/y/yaw 命令跟踪。")),
            _sec("Obs：姿态稳定", ("obs_groups_spec", "_compute_obs"), ("42 维 obs 关注 gyro、gravity、关节差值、关节速度和动作历史。", "没有 joystick command 和 gait phase，说明策略输入围绕平衡控制组织。")),
            _sec("Action：12 维姿态调节", ("apply_action", "default_angles"), ("Go2 handstand 复用 locomotion common base 的动作映射。", "动作仍控制 Go2 12 个腿部 actuator，策略通过相对默认角偏移学习维持 handstand 姿态。"), source="src/unilab/envs/locomotion/common/base.py"),
            _sec("Reward：高度/朝向/接触", ("_init_reward_functions", "height"), ("reward 读取 handstand 专用项，不再以速度跟踪作为主目标。", "接触和姿态项决定是否真正站成目标支撑构型。")),
            _sec("Termination", ("update_state", "terminated"), ("终止由高度、倾斜、接触失败和 horizon 组合。", "handstand 的失败定义和普通四足行走不同。")),
        ),
    ),
    "go2_footstand": TaskCodeGuide(
        focus=(
            "Footstand 是 Go2 中最需要单独写的任务：它有专用 reset provider、obs history、energy termination 和前/后足接触逻辑。",
            "动作仍是 12 维，但 reward 明确区分目标前足支撑、后足接触、身体高度和能耗。",
            "阅读时优先看 `Go2FootStandTask`，它不是 Go2WalkTask 的简单参数变体。",
        ),
        sections=(
            _sec("配置：前足站立", ("Go2FootStandCfg", "Go2FootStand"), ("配置中设置 max_episode_seconds、history、reward 和 footstand 专用阈值。", "MuJoCo-only 任务应通过 owner YAML 选择，而不是覆盖 backend 字段。")),
            _sec("Reset：站立初态", ("build_reset_plan", "FootStand"), ("reset 在 handstand 基础上改写站立任务需要的信息。", "随机关节 qpos、质量和摩擦都只在 reset 冷路径发生。")),
            _sec("Obs：历史堆叠", ("obs_groups_spec", "_compute_obs"), ("obs 包含 playground/姿态/动作历史等字段，history 让策略记住姿态变化趋势。", "critic 包含更多特权项帮助 value 估计前足支撑是否稳定。")),
            _sec("Action：裁剪与默认角", ("apply_action", "clip_actions"), ("动作裁剪后乘 action_scale，并叠加 default angles。", "这里每一维仍对应 Go2 actuator 表中的腿部关节。")),
            _sec("Reward/Termination：能量与接触", ("_init_reward_functions", "update_state"), ("height/orientation/contact/tar/rear_feet_contact/energy 是本任务核心。", "termination 不只是摔倒，还包括能量阈值和站立姿态失败。")),
        ),
    ),
    "go2w_joystick_flat": TaskCodeGuide(
        focus=(
            "Go2W 平地任务必须把腿部 12 DoF 和 4 个轮关节分开讲：action 维度变成 16，控制函数也不同于 Go2。",
            "obs 要看轮关节速度/动作历史如何进入策略输入，reward 要看轮足速度跟踪和姿态惩罚如何组合。",
            "源码关键在 `Go2WJoystickEnv.apply_action` 与 `compute_go2w_motor_ctrl` 相关路径。",
        ),
        sections=(
            _sec("配置与轮足模型", ("Go2WJoystickCfg", "Go2WJoystickFlat"), ("cfg 指向 Go2W scene，模型包含 12 个腿关节和 4 个轮 actuator。", "轮足控制保持在 env/backend 边界内，训练脚本不解释轮子。")),
            _sec("Reset：轮足初态", ("build_reset_plan", "zero_actions"), ("reset 初始化 commands、动作历史和轮足状态。", "Go2W 的 16 维 action history 必须和 actuator 顺序一致。")),
            _sec("Obs：轮足状态", ("obs_groups_spec", "_compute_obs"), ("obs 维度大于 Go2，因为包含轮关节相关状态。", "critic 继续携带额外速度/特权信息。")),
            _sec("Action：腿 + 轮", ("apply_action", "compute_go2w_motor_ctrl"), ("腿部按位置目标控制，轮部由 Go2W 专用控制 helper 转换。", "这就是为什么不能把 Go2W 简单当成 16 维 Go2。")),
            _sec("Reward/Termination", ("_init_reward_functions", "update_state"), ("reward 仍以速度跟踪为主，但加入轮足稳定和动作平滑约束。", "终止关注 base 高度、倾斜和轮足异常。")),
        ),
    ),
    "go2w_joystick_rough": TaskCodeGuide(
        focus=(
            "Go2W rough 同时有轮足控制和地形通过性，必须单独读 rough env，而不是套 Go2W flat 或 Go2 rough。",
            "动作 16 维不变，新增复杂度来自 terrain reset、height scan critic 和 rough reward gate。",
            "这个任务最重要的是区分轮部推进、腿部支撑和地形观测三条线。",
        ),
        sections=(
            _sec("配置与地形", ("Go2WJoystickRoughCfg", "Go2WRoughTerrainCfg"), ("rough cfg 声明轮足地形任务的 terrain generator。", "scene/task fragment 承载地形，robot asset 不写任务逻辑。")),
            _sec("Reset：轮足地形 spawn", ("build_reset_plan", "spawn"), ("reset 同时处理地形 spawn 和 16 维轮足动作缓存。", "随机化只在 reset payload 中传递给 backend。")),
            _sec("Obs：轮足 + height scan", ("obs_groups_spec", "_compute_obs"), ("policy 观测保留轮足本体状态，critic 增加 height scan。", "rough 下高度扫描帮助 value 判断轮足是否处于危险地形。")),
            _sec("Action：专用轮足控制", ("apply_action", "compute_go2w_motor_ctrl"), ("Go2W rough 继承 Go2W flat 的轮足 apply_action，不在 rough.py 中重复实现。", "腿部和轮部控制语义不同，必须看 Go2W 专用映射。"), source="src/unilab/envs/locomotion/go2w/joystick.py"),
            _sec("Reward/Termination", ("_init_reward_functions",), ("rough.py 本地覆写 reward map，并用 upright gate 调整地形稳定项。", "终止逻辑继承 Go2WJoystickEnv.update_state，结合倾斜、地形相对高度和 horizon。")),
        ),
    ),
    "g1_walk_flat": TaskCodeGuide(
        focus=(
            "G1 平地行走是 29 DoF 人形 locomotion：不能用四足 gait 解释，必须读 gait_phase、上身 pose 和脚相位 reward。",
            "action 覆盖双腿、腰、双臂，默认角来自 stand keyframe，策略输出是全身相对角度目标。",
            "reward 的 pose/upper_body_pose/feet_phase 是 G1 特有阅读重点。",
        ),
        sections=(
            _sec("配置：G1WalkFlat", ("G1WalkFlatCfg", "G1WalkEnvCfg"), ("cfg 指向 G1 flat scene，并指定 G1WalkRewardConfig。", "curriculum 和 control_config 在 env cfg 层表达，不放训练脚本。")),
            _sec("Obs：全身本体 + gait phase", ("obs_groups_spec", "_compute_obs"), ("actor obs 为 98 维：gyro、gravity、29 维关节差、29 维速度、29 维动作、命令和相位。", "critic 增加 base linvel，总维度 101。")),
            _sec("Action：29 DoF 全身目标", ("apply_action", "gait_phase"), ("apply_action 更新 gait_phase 和动作历史。", "ctrl = action_scale * actions + default_angles，表示相对 stand 姿态的全身目标角。")),
            _sec("Reward：人形步态", ("_init_reward_functions", "upper_body_pose"), ("reward map 包含 tracking、base_height、pose、upper_body_pose、feet_phase 等。", "上肢姿态和脚相位让 G1 走路时不只是追速度。")),
            _sec("Termination/DR", ("update_state", "G1DomainRandConfig"), ("update_state 用 tilt 和 terrain-relative base height 判断终止。", "kp/kd、质量、COM、摩擦、重力等随机化来自 G1DomainRandConfig。")),
        ),
    ),
    "g1_walk_rough": TaskCodeGuide(
        focus=(
            "G1 rough 与 flat 共用 G1WalkEnv 源码，但任务差异在 owner/scene 和 terrain-relative 终止口径，文档必须显式说明这一点。",
            "它不是新的 action 解释，仍是 29 DoF；重点是崎岖地形下高度、姿态和 reward 权重变化。",
            "源码阅读要把 `G1WalkRough` registry/config 与同文件的 G1WalkEnv 行为对应起来。",
        ),
        sections=(
            _sec("配置：G1WalkRough", ("G1WalkRoughCfg", "G1WalkRough"), ("rough cfg 选择 scene_rough，并沿用 G1WalkEnv 行为。", "任务差异主要来自配置、scene 和 owner YAML 权重。")),
            _sec("Obs：G1 rough 输入", ("obs_groups_spec", "_compute_obs"), ("obs 结构和 G1 flat 保持一致，便于复用策略结构。", "rough 的环境风险通过地形相对高度和 reward/termination 进入学习。")),
            _sec("Action：不变的 29 DoF", ("apply_action", "default_angles"), ("action 仍是全身关节目标。", "不能因为 rough 场景存在就引入训练脚本层面的动作分支。")),
            _sec("Reward：rough 权重", ("_init_reward_functions", "PenaltyCurriculum"), ("源码 reward map 与 flat 相近，具体权重由 rough owner YAML 决定。", "penalty curriculum 影响惩罚项强度，帮助崎岖地形训练稳定。")),
            _sec("Termination：地形相对高度", ("update_state", "_terrain_relative_base_height"), ("终止看 tilt 和 terrain-relative base height。", "这一点是 rough 与 flat 文档必须分开的关键。")),
        ),
    ),
    "k1_walk_flat": TaskCodeGuide(
        focus=(
            "K1 Walk 是 22 DoF 人形行走，obs 使用堆叠历史，和 G1 的 29 DoF 单帧结构不同。",
            "动作包括头/臂/腿等 K1 actuator，但 reward 重点仍是行走稳定、脚相位和姿态。",
            "K1 的 `_compute_obs`、`_init_reward_functions` 和 `apply_action` 都应单独阅读。",
        ),
        sections=(
            _sec("配置：K1WalkFlat", ("K1WalkFlatCfg", "K1WalkEnvCfg"), ("cfg 指向 K1 flat scene，并绑定 K1RewardConfig。", "K1 使用自己的 actuator order 和 observation constants。")),
            _sec("Obs：历史堆叠", ("obs_groups_spec", "_compute_obs"), ("obs 为 K1_OBS_STACKED_DIM，来自多帧本体/命令/action history。", "critic 在堆叠 obs 外追加 base linvel。")),
            _sec("Action：22 维 K1 目标", ("apply_action", "default_angles"), ("动作缓存 last/current actions，并缩放到 22 个 actuator。", "球或外物不在本任务中出现。")),
            _sec("Reward：K1 步态", ("_init_reward_functions", "feet_phase"), ("reward map 包含速度跟踪、pose、feet_phase、close feet、action_rate 等。", "这些项约束 K1 以人形步态跟踪命令。")),
            _sec("Termination/DR", ("update_state", "K1WalkDomainRandomizationProvider"), ("update_state 同步计算 done、reward 和 obs。", "domain randomization provider 负责 K1 reset/物理扰动。")),
        ),
    ),
    "k1_soccer_dribble": TaskCodeGuide(
        focus=(
            "K1 足球盘带必须单独写：action 只控制机器人，球的运动来自 free joint 和接触动力学。",
            "obs 在 K1 行走历史帧基础上加入 ball relative state；reward 也从行走扩展到 ball_progress、ball_keep、ball_front、ball_lost。",
            "这个任务的 reset provider 会初始化机器人和球，并生成朝球/带球命令，是理解任务目标的关键。",
        ),
        sections=(
            _sec("配置：足球场景", ("K1SoccerDribbleFlatCfg", "K1SoccerDribbleCfg"), ("cfg 指向 scene_soccer_dribble_minimal，场景中包含足球 free joint。", "K1SoccerDribbleEnv 继承 K1WalkEnv，但加入球相关状态。")),
            _sec("Reset：机器人 + 球", ("build_reset_plan", "_build_extra_info_updates"), ("reset 复制机器人初态并通过 spawn 记录 episode start。", "info_updates 中写入 commands、动作历史和球任务额外信息。")),
            _sec("Obs：球相对状态", ("obs_groups_spec", "_compute_obs", "_ball_state"), ("obs 在 K1 stack 基础上增加球相对位置/速度等字段。", "critic 继续添加 base linvel，帮助估计盘带稳定性。")),
            _sec("Action：只控机器人", ("apply_action", "default_angles"), ("K1 足球任务继承 K1WalkEnv 的 apply_action，动作维度仍是 K1 22 个 actuator。", "球不在 action 中，盘带效果完全由接触物理产生。"), source="src/unilab/envs/locomotion/k1/joystick.py"),
            _sec("Reward/Termination：盘带目标", ("_init_reward_functions", "update_state"), ("reward map 追加 ball_progress、ball_keep、ball_front、ball_speed_match、ball_lost 等。", "termination 同时看机器人跌倒和球丢失距离。")),
        ),
    ),
    "go2_arm_manip_loco": TaskCodeGuide(
        focus=(
            "Go2Arm 是移动操作，不是简单 locomotion：action 同时覆盖 12 个腿关节和 6 个机械臂关节。",
            "obs 要区分底盘状态、机械臂状态、目标/物体相对信息和历史帧。",
            "reward 同时约束移动稳定、末端目标、碰撞和动作平滑，不能只按四足速度跟踪解释。",
        ),
        sections=(
            _sec("配置：底盘 + 机械臂", ("Go2ArmManipLocoCfg", "_default_go2_arm_scene"), ("cfg 指向组合 scene，robot 资产由 Go2 底盘和 Airbot arm 构成。", "任务规则留在 manip_loco env，训练脚本只选择注册名。")),
            _sec("Reset：目标与历史", ("build_reset_plan", "target"), ("reset 在底盘 home 姿态上初始化 arm、目标和历史缓存。", "目标/物体信息进入 info_updates，后续 obs/reward 共用。")),
            _sec("Obs：移动操作输入", ("obs_groups_spec", "_compute_obs"), ("obs 结合底盘本体、arm joint、目标相对量和 history。", "actor/critic 维度由 history 与目标字段共同决定。")),
            _sec("Action：18 维混合控制", ("apply_action", "arm"), ("前 12 维用于腿部，后 6 维用于机械臂。", "控制映射必须保持 actuator 顺序，不能在训练脚本里拆分业务规则。")),
            _sec("Reward/Termination", ("_init_reward_functions", "update_state"), ("reward 同时包含 locomotion 稳定项和 arm/object 距离项。", "终止关注底盘跌倒、机械臂异常/碰撞和 episode horizon。")),
        ),
    ),
    "g1_motion_tracking": TaskCodeGuide(
        focus=(
            "G1 Motion Tracking 的核心不是 joystick，而是从 motion npz 采样参考帧并构造机器人/参考误差。",
            "reset 从 motion reference 生成 qpos/qvel，obs 同时包含机器人当前状态和参考 motion anchor/body 目标。",
            "reward 是 root/body/joint/action 多层 tracking 误差，不应按 locomotion 速度跟踪来写。",
        ),
        sections=(
            _sec("配置：MotionLoader", ("G1MotionTrackingEnvCfg", "G1MotionTrackingCfg"), ("cfg 声明 motion_file、body_names、anchor 和 tracking 阈值。", "MotionLoader 在 env 初始化/冷路径读取 motion 数据。")),
            _sec("Reset：参考帧初态", ("build_reset_plan", "motion_loader.get_motion_at_frame"), ("reset 采样 motion frame，并用参考动作构造 qpos/qvel。", "randomization 可叠加在参考初态上，而不是替代 motion 目标。")),
            _sec("Obs：机器人 + 参考目标", ("obs_groups_spec", "_compute_obs"), ("actor obs 包含 command、motion anchor、linvel/gyro、joint 和 action。", "critic 额外加入 body pos/ori 等 tracking 特权项。")),
            _sec("Action：延迟与默认角偏移", ("apply_action", "simulate_action_latency"), ("动作可按配置模拟 latency，之后叠加 default_dof_pos_bias。", "这是 sim2real/部署前常见控制口径。")),
            _sec("Reward/Termination", ("_init_reward_functions", "update_state"), ("reward map 是多层 motion tracking：root、body、joint、velocity、action_rate。", "termination 看 anchor z/ori、ee z、undesired contact 和 clip 结束。")),
        ),
    ),
    "g1_motion_tracking_deploy": TaskCodeGuide(
        focus=(
            "Deploy 任务和普通 tracking 共享运动跟踪目标，但重点是部署可用观测组织。",
            "应读 `G1MotionTrackingDeployEnv`，看它如何覆盖/整理 actor 输入，而不是重复普通 tracking 文案。",
            "reward 和 reset 可沿用 tracking，但文档必须说明 deploy 的差异点在 obs contract。",
        ),
        sections=(
            _sec("配置：Deploy 注册", ("G1MotionTrackingDeployEnvCfg", "G1MotionTrackingDeploy"), ("deploy cfg 单独注册，便于 owner YAML 使用部署口径。", "它不改变任务目标，只改变环境暴露的观测组织。")),
            _sec("Reset：沿用 motion reference", ("build_reset_plan", "motion_loader.get_motion_at_frame"), ("reset 仍从 motion clip 采样参考初态。", "domain randomization 和普通 tracking 一致，由 owner YAML 控制启用项。")),
            _sec("Obs：部署 actor 输入", ("G1MotionTrackingDeployEnv", "_compute_obs"), ("deploy env 重点调整 actor obs，减少训练期不可部署的特权项。", "critic/训练辅助信息仍由 env contract 管理。")),
            _sec("Action：29 DoF tracking", ("apply_action", "simulate_action_latency"), ("动作口径与普通 tracking 一致。", "部署任务不引入新的 actuator 或训练脚本分支。")),
            _sec("Reward/Termination", ("_init_reward_functions", "update_state"), ("reward 和 termination 继承 tracking 主逻辑。", "文档中应把 deploy 差异写在 obs，而不是虚构新的 reward。")),
        ),
    ),
    "g1_flip_tracking": TaskCodeGuide(
        focus=(
            "Flip tracking 的独立性来自 flip motion profile 和大姿态变化阈值，不是新的 action 空间。",
            "要读 `G1FlipTrackingCfg`，确认 motion file、episode 秒数和终止阈值如何为翻转调整。",
            "reward 仍是 motion tracking 误差，但翻转对 root orientation/body velocity 的敏感度更高。",
        ),
        sections=(
            _sec("配置：Flip clip", ("G1FlipTrackingEnvCfg", "G1FlipTrackingCfg"), ("cfg profile 指向 flip motion clip，并设置适配翻转的阈值。", "它注册到同一 tracking env 框架，避免复制业务逻辑。")),
            _sec("Reset：翻转参考帧", ("build_reset_plan", "motion_frames"), ("reset 从 flip clip 采样帧，qpos/qvel 跟随翻转姿态。", "adaptive sampling 决定训练从哪些困难帧开始。"), source="src/unilab/envs/motion_tracking/g1/tracking.py"),
            _sec("Obs：剧烈参考姿态", ("obs_groups_spec", "_compute_obs"), ("obs 结构继承 tracking，但 reference command 的变化更剧烈。", "策略必须根据 motion anchor/body target 预测翻转阶段。"), source="src/unilab/envs/motion_tracking/g1/tracking.py"),
            _sec("Action：29 DoF", ("apply_action", "default_dof_pos_bias"), ("action 与普通 tracking 一样映射到 29 DoF。", "翻转任务不通过额外 action 控制 root。"), source="src/unilab/envs/motion_tracking/g1/tracking.py"),
            _sec("Reward/Termination", ("_init_reward_functions", "update_state"), ("reward 仍按 root/body/joint tracking 误差计算。", "终止阈值由 flip cfg 放宽或调整，以容纳大姿态变化。"), source="src/unilab/envs/motion_tracking/g1/tracking.py"),
        ),
    ),
    "g1_wall_flip_tracking": TaskCodeGuide(
        focus=(
            "Wall flip 不是普通 flip：scene 中有 wall，motion clip 表达靠墙交互。",
            "必须把场景几何、undesired contact 和 reference motion 一起解释，不能只写空翻。",
            "源码 profile 在 flip_tracking.py，执行主体仍是 G1MotionTrackingEnv。",
        ),
        sections=(
            _sec("配置：带墙场景", ("G1WallFlipTrackingEnvCfg", "G1WallFlipTrackingCfg"), ("cfg 选择 wall flip motion 和带墙 scene。", "墙体属于 task scene，不写进 robot.xml。")),
            _sec("Reset：靠墙参考帧", ("build_reset_plan", "motion_frames"), ("reset 从 wall flip clip 构造初态。", "参考状态包含靠墙动作阶段的 root/body 姿态。"), source="src/unilab/envs/motion_tracking/g1/tracking.py"),
            _sec("Obs：wall flip 目标", ("obs_groups_spec", "_compute_obs"), ("obs 仍是 tracking 结构，但 reference target 来自 wall flip。", "策略通过参考 anchor/body 信息学习与墙交互时机。"), source="src/unilab/envs/motion_tracking/g1/tracking.py"),
            _sec("Action：全身追踪", ("apply_action", "simulate_action_latency"), ("29 DoF action 只控制关节，不直接控制墙体交互。", "墙体接触由物理仿真产生。"), source="src/unilab/envs/motion_tracking/g1/tracking.py"),
            _sec("Reward/Termination", ("_init_reward_functions", "undesired_contact"), ("reward 仍以 tracking 误差为核心。", "termination/contact 项决定哪些墙体/身体接触是允许或失败。"), source="src/unilab/envs/motion_tracking/g1/tracking.py"),
        ),
    ),
    "g1_climb_tracking": TaskCodeGuide(
        focus=(
            "Climb tracking 使用 climb scene 和爬箱 motion，任务目标是跟踪上升过程，不是平地或翻转。",
            "配置 profile 决定 scene_climb、motion clip 和 episode 长度。",
            "执行主体复用 tracking env，因此文档应重点写 climb profile 带来的参考动作/场景差异。",
        ),
        sections=(
            _sec("配置：爬箱 profile", ("G1ClimbTrackingEnvCfg", "G1ClimbTrackingCfg"), ("cfg 指向 climb_20_z_scale_1 motion/scene。", "爬箱几何在 task scene 中表达。")),
            _sec("Reset：爬升参考状态", ("build_reset_plan", "motion_frames"), ("reset 从爬箱 motion 采样初态。", "qpos/qvel 反映爬升阶段，而非普通站立。"), source="src/unilab/envs/motion_tracking/g1/tracking.py"),
            _sec("Obs：爬升目标", ("obs_groups_spec", "_compute_obs"), ("obs 包含 reference anchor/body target，使策略知道当前应处于爬升哪一段。", "critic 使用 body 特权信息衡量整体跟踪误差。"), source="src/unilab/envs/motion_tracking/g1/tracking.py"),
            _sec("Action：29 DoF 爬升控制", ("apply_action", "default_dof_pos_bias"), ("动作仍控制 G1 29 个关节。", "爬升高度来自参考轨迹和物理接触，不是 action 中的额外维度。"), source="src/unilab/envs/motion_tracking/g1/tracking.py"),
            _sec("Reward/Termination", ("_init_reward_functions", "update_state"), ("reward 以 tracking 误差约束 root/body/joint。", "终止关注 anchor/末端误差和不期望接触。"), source="src/unilab/envs/motion_tracking/g1/tracking.py"),
        ),
    ),
    "g1_box_tracking": TaskCodeGuide(
        focus=(
            "Box tracking 在 G1 motion tracking 外加入 largebox 物体状态，物体不是 action 控制对象。",
            "obs/reward/termination 都要看 box 扩展：object pos/ori/velocity 如何进入学习目标。",
            "reset 需要同时初始化机器人 reference 和 box reference，这是和普通 tracking 最大的不同。",
        ),
        sections=(
            _sec("配置：large box", ("G1BoxTrackingEnvCfg", "G1BoxTrackingCfg"), ("cfg 指向 largebox scene 和 box motion loader。", "物体状态属于任务层，机器人 XML 不包含 box 任务逻辑。")),
            _sec("Reset：机器人 + 箱体", ("build_reset_plan", "BoxMotionLoader"), ("reset 采样 motion，并构造机器人与箱体初态。", "info_updates 保存 box reference 供 obs/reward 使用。")),
            _sec("Obs：加入物体状态", ("obs_groups_spec", "_compute_obs"), ("obs 在 tracking 基础上扩展 object position/orientation/velocity。", "critic 需要更完整的物体误差信息。")),
            _sec("Action：只控 G1", ("apply_action", "simulate_action_latency"), ("action 仍为 29 DoF 关节目标。", "箱体运动来自接触动力学和参考跟踪，不在 action 中。"), source="src/unilab/envs/motion_tracking/g1/tracking.py"),
            _sec("Reward/Termination：物体误差", ("_init_reward_functions", "_reward_object"), ("reward 追加 object pos/ori 等箱体跟踪项。", "终止增加物体误差阈值，避免只看机器人自身是否跟踪。")),
        ),
    ),
    "g1_wbt_obs": TaskCodeGuide(
        focus=(
            "G1WBTObs 的价值在观测口径，不是新动作或新机器人：它为 SAC/WBT 对齐 deploy-like obs/history。",
            "要读 tracking_obs.py 中的 cfg、reset provider 和 `_compute_obs`，因为它重写了观测结构。",
            "reward 在 tracking 基础上扩展，obs contract 是这个任务文档的重点。",
        ),
        sections=(
            _sec("配置：WBT obs", ("G1WBTObsCfg", "G1WBTObs"), ("cfg 单独注册 WBT 观测任务，并设置 obs/history/domain randomization 字段。", "它继承 SAC tracking cfg，但不是普通 PPO tracking 页面。")),
            _sec("Reset：WBT 随机化", ("build_reset_plan", "WBT"), ("reset 在 motion reference 上叠加 WBT 专用随机化。", "默认关节偏置、摩擦等扰动写入 ResetPlan/info。")),
            _sec("Obs：严格观测结构", ("_compute_obs",), ("_compute_obs 是本任务核心，组织 proprio history 和 critic 特权信息。", "文档应优先解释字段来源和维度，而不是只列 reward。")),
            _sec("Action：SAC tracking action", ("apply_action", "simulate_action_latency"), ("动作沿用 G1 tracking 29 DoF 目标。", "SAC 只改变算法/obs 口径，不改变 actuator。"), source="src/unilab/envs/motion_tracking/g1/tracking.py"),
            _sec("Reward/Termination", ("_init_reward_functions",), ("reward 在父类 tracking 基础上追加/调整 WBT 项。", "termination 继承 tracking.py 的风险边界，本任务不在 tracking_obs.py 重写 update_state。")),
        ),
    ),
    "allegro_inhand": TaskCodeGuide(
        focus=(
            "Allegro rotation 是手内旋转任务：action 控制 16 个手指关节，球体 free joint 由接触物理驱动。",
            "obs 要看手指关节、目标旋转轴/速度、球姿态/角速度和 lag history。",
            "reward 不是 locomotion tracking，而是 rotate、物体稳定、torque/work 等操作任务项。",
        ),
        sections=(
            _sec("配置：Allegro rotation", ("AllegroRotationPPOCfg", "AllegroInhandRotation"), ("cfg 指向 Allegro hand scene，并设置球体/目标旋转参数。", "手和球的资产在 scene 中 include，机器人 XML 不写任务 reward。")),
            _sec("Reset：抓取/球初态", ("build_reset_plan", "grasp"), ("reset 可从 grasp cache 或随机初态构造手指和球体状态。", "球体 qpos/qvel 与手指 qpos 同时进入 ResetPlan。")),
            _sec("Obs：手指 + 球", ("obs_groups_spec", "_compute_obs"), ("obs 按 lag steps 堆叠手指状态、球状态、目标控制和动作历史。", "球姿态/角速度是旋转任务的核心输入。")),
            _sec("Action：16 维手指控制", ("apply_action", "clipped_actions"), ("action 在 AllegroBaseEnv 中裁剪并映射到手指目标。", "球不在 action 中，旋转来自接触。"), source="src/unilab/envs/manipulation/allegro_inhand/base.py"),
            _sec("Reward/Termination", ("_init_reward_functions", "update_state"), ("reward 包含 rotate、obj_linvel、pose_diff、torque、work 等项。", "termination 关注球体掉落/高度阈值和 horizon。")),
        ),
    ),
    "allegro_inhand_grasp": TaskCodeGuide(
        focus=(
            "Allegro grasp generation 的目标是得到稳定抓取初态，而不是高速旋转。",
            "它继承 rotation 基础设施，但 apply_action/update_state 有抓取生成专用行为。",
            "文档要强调抓取稳定、球体保持和 cache/procedural reset，而不是照抄 rotation reward。",
        ),
        sections=(
            _sec("配置：Grasp cfg", ("AllegroRotationGraspCfg", "AllegroInhandRotationGrasp"), ("grasp cfg 覆盖 rotation 默认参数，用于抓取生成。", "注册名单独存在，便于 owner YAML 选择 grasp 任务。")),
            _sec("Reset：抓取初态", ("build_reset_plan", "grasp"), ("reset 复用/生成抓取状态。", "核心是让球从稳定可控姿态开始，而不是追求旋转速度。"), source="src/unilab/envs/manipulation/allegro_inhand/rotation.py"),
            _sec("Obs：沿用手/球状态", ("obs_groups_spec", "_compute_obs"), ("obs 仍包含手指和球体状态。", "抓取任务更关注球是否保持在手内。"), source="src/unilab/envs/manipulation/allegro_inhand/rotation.py"),
            _sec("Action：抓取生成控制", ("apply_action", "del actions"), ("grasp_gen 中 action 逻辑不同于普通 PPO rotation。", "需要看它如何处理/忽略策略动作以服务 cache 生成。")),
            _sec("Reward/Termination", ("update_state", "grasp"), ("update_state 在父类 rotation 后追加抓取生成判断。", "终止关注球体丢失、抓取失败和 horizon。")),
        ),
    ),
    "sharpa_inhand": TaskCodeGuide(
        focus=(
            "Sharpa rotation 是 22 DoF 灵巧手旋转圆柱体，和 Allegro 的球体旋转不同。",
            "obs 包含手指、物体、动作历史以及 Sharpa 可选 tactile/contact 信息；action 控制 22 个 hand actuator。",
            "reward 需要看旋转进度、物体稳定和接触/动作惩罚，不应按 Allegro 直接替换名字。",
        ),
        sections=(
            _sec("配置：Sharpa rotation", ("SharpaInhandRotationCfg", "SharpaInhandRotation"), ("cfg 设置 critic_info_dim、obs lag、物体和 Sharpa 专用参数。", "场景 include Sharpa 手和圆柱物体。")),
            _sec("Reset：手与圆柱体", ("build_reset_plan", "SharpaInhandRotationDRProvider"), ("reset provider 构造手指、物体 pose 和随机化 payload。", "grasp/cache 或随机初始化都只在 reset 冷路径处理。")),
            _sec("Obs：手/物体/history", ("obs_groups_spec", "_compute_obs_from_inputs"), ("obs 维度由 policy frame 和 obs_lag_steps 决定。", "critic 可额外接收物体/接触相关信息。")),
            _sec("Action：22 维手部目标", ("apply_action", "actions_np"), ("action 裁剪后映射到 Sharpa 22 个 actuator。", "圆柱体 free joint 由接触动力学驱动，不在 action 中。")),
            _sec("Reward/Termination", ("_compute_reward", "update_state"), ("reward 关注旋转、物体稳定、动作/力矩和接触项。", "Sharpa rotation 直接在 `_compute_reward` 中遍历 reward_config.scales，不使用 `_init_reward_functions` 分发表。")),
        ),
    ),
    "sharpa_inhand_grasp": TaskCodeGuide(
        focus=(
            "Sharpa grasp generation 的目标是收集/保持稳定抓取状态，为 rotation 提供可用初态。",
            "它有自己的 grasp DR provider 和 apply_action/update_state 覆盖，不能只沿用 rotation 文案。",
            "文档应强调 cache 收集、物体保持和接触稳定，而不是旋转速度。",
        ),
        sections=(
            _sec("配置：Sharpa grasp", ("SharpaInhandGraspEnvCfg", "SharpaInhandRotationGraspCfg"), ("grasp cfg 缩短 episode 并调整抓取生成参数。", "注册名 SharpaInhandRotationGrasp 指向抓取任务。")),
            _sec("Reset：grasp cache", ("build_reset_plan", "SharpaInhandGraspDRProvider"), ("reset provider 会保留原抓取任务行为并收集成功 pre-reset 状态。", "这和 rotation 的普通随机初态不同。")),
            _sec("Obs：抓取质量输入", ("obs_groups_spec", "_compute_obs_from_inputs"), ("obs 沿用 Sharpa 手/物体/history 结构。", "抓取任务更关注物体是否在稳定接触区域。"), source="src/unilab/envs/manipulation/sharpa_inhand/rotation.py"),
            _sec("Action：抓取生成动作", ("apply_action", "Grasp-cache"), ("grasp_gen 覆盖 apply_action，不使用普通策略随机动作收集 cache。", "这是生成稳定抓取初态的关键差异。")),
            _sec("Reward/Termination", ("update_state", "success"), ("update_state 在父类基础上追加成功抓取记录/终止逻辑。", "termination 关注物体掉落、接触失败和 horizon。")),
        ),
    ),
}


CATEGORY_TITLES = {
    "locomotion": "运动控制",
    "motion_tracking": "动作追踪",
    "manipulation": "灵巧操作",
    "manip_loco": "移动操作",
}


REWARD_EXPLANATIONS: dict[str, str] = {
    "tracking_lin_vel": "线速度命令跟踪奖励，鼓励机器人实际平面速度贴近 joystick/命令速度。",
    "tracking_ang_vel": "角速度命令跟踪奖励，鼓励 yaw 角速度贴近转向命令。",
    "lin_vel_z": "竖直速度惩罚，限制 base 上下弹跳。",
    "ang_vel_xy": "横滚/俯仰角速度惩罚，抑制身体剧烈摇摆。",
    "base_height": "身体高度项，使 base 高度保持在任务目标附近。",
    "orientation": "姿态项，通常基于重力投影或目标朝向惩罚倾斜。",
    "action_rate": "动作变化惩罚，抑制相邻控制步之间的目标突变。",
    "action_rate_l2": "动作变化 L2 惩罚， motion tracking 中用于约束动作平滑。",
    "pose": "默认姿态或参考姿态约束，防止无关关节偏离可用构型。",
    "feet_phase": "步态相位奖励，鼓励左右脚按期望相位摆动/支撑。",
    "feet_phase_contact": "足端接触相位奖励，鼓励支撑相接触、摆动相离地。",
    "feet_phase_contrast": "步态相位对比项，增加左右脚交替节律。",
    "feet_double_stance": "双脚同时支撑惩罚/约束，避免拖步或不自然站立。",
    "penalty_close_feet_xy": "双脚在水平面过近惩罚，降低绊脚风险。",
    "height": "目标高度奖励，用于 footstand/handstand 等姿态任务。",
    "contact": "接触项，约束指定足端或身体部位的接触模式。",
    "termination": "失败终止惩罚，在触发终止时给负奖励。",
    "termination_bad": "足球盘带任务中的失败终止惩罚，强调跌倒或丢球代价。",
    "dof_pos_limits": "关节位置限位惩罚，防止接近 XML joint range 边界。",
    "torques": "力矩惩罚，降低控制输出过大。",
    "dof_acc": "关节加速度惩罚，抑制高频振荡。",
    "tar": "目标前腿姿态项，FootStand 中约束前腿到目标角度。",
    "rear_feet_contact": "后足接触奖励/约束，FootStand 中用于稳定后腿接触模式。",
    "rear_leg_symmetry": "后腿对称性约束，避免左右后腿构型过度不一致。",
    "front_leg_motion": "前腿运动惩罚，抑制支撑姿态下前腿无效摆动。",
    "upright_stability": "竖直稳定项，鼓励身体朝向目标支撑方向。",
    "knee_clearance": "膝部离地/高度约束，防止膝盖碰地。",
    "stay_still": "静止约束，姿态任务中减少 base 漂移。",
    "energy": "能量消耗惩罚，降低力矩与速度共同造成的控制代价。",
    "forward_progress": "前向进度奖励，鼓励沿目标方向推进。",
    "under_speed": "低速惩罚，命令非零时避免策略停滞。",
    "alive": "存活奖励，鼓励保持非终止状态。",
    "ball_progress": "足球沿目标方向前进的奖励。",
    "ball_keep": "保持球在机器人附近的奖励。",
    "ball_front": "保持球在机器人前方的奖励，避免球跑到侧后方。",
    "ball_speed_match": "球速与目标速度匹配奖励。",
    "ball_approach": "机器人接近足球的奖励。",
    "ball_moving": "鼓励足球移动，避免只站在球旁。",
    "ball_still": "球静止惩罚。",
    "ball_lost": "丢球惩罚，球离机器人过远时触发。",
    "ball_over_speed": "球过快惩罚，避免不可控大力踢球。",
    "ball_orbit": "绕球打转惩罚，鼓励直接推进而不是原地绕行。",
    "feet_inward_yaw": "脚内扣惩罚，改善足球任务中的脚尖朝向。",
    "feet_step_travel": "足端步幅/位移奖励，鼓励有效迈步接触球。",
    "object_distance": "移动操作中末端/目标物体距离项。",
    "object_distance_l2": "移动操作中目标距离 L2 项。",
    "arm_collision": "机械臂碰撞惩罚。",
    "motion_global_root_pos": "参考 motion 的 root 全局位置跟踪奖励。",
    "motion_global_root_ori": "参考 motion 的 root 全局朝向跟踪奖励。",
    "motion_body_pos": "参考 motion 的各 body 位置跟踪奖励。",
    "motion_body_ori": "参考 motion 的各 body 朝向跟踪奖励。",
    "motion_body_lin_vel": "参考 motion 的 body 线速度跟踪奖励。",
    "motion_body_ang_vel": "参考 motion 的 body 角速度跟踪奖励。",
    "motion_ee_body_pos_z": "末端 body 高度跟踪奖励。",
    "motion_joint_pos": "参考 motion 的关节位置跟踪奖励。",
    "motion_joint_vel": "参考 motion 的关节速度跟踪奖励。",
    "joint_limit": "关节限位惩罚，避免追踪动作时撞到机械限位。",
    "undesired_contacts": "非期望接触惩罚，例如手、膝、身体等部位异常触地。",
    "rotate": "手内旋转主奖励，鼓励物体沿目标轴旋转。",
    "obj_linvel": "物体线速度惩罚，避免物体被甩飞或平移过大。",
    "pose_diff": "手部姿态偏差惩罚，保持抓持构型稳定。",
    "torque": "手部力矩惩罚，降低过大的执行器输出。",
    "work": "机械功惩罚，约束力矩与运动共同造成的能耗。",
    "object_pos": "物体位置保持奖励/惩罚，防止被操作物体偏离手心。",
}


OBS_FAMILY_ROWS: dict[str, list[list[str]]] = {
    "quadruped_locomotion": [
        ["command", "通常 3 维", "`commands` / joystick 采样", "期望 x/y 线速度和 yaw 角速度，是策略要跟踪的外部目标。"],
        ["gyro", "3 维", "`get_gyro()` / sensor.gyro", "机身角速度，帮助策略判断身体旋转和姿态变化。"],
        ["gravity", "3 维", "局部重力投影", "把世界重力投到 body 坐标系，用于表示 roll/pitch 倾斜。"],
        ["dof_pos - default", "nu 维", "关节角与 keyframe 默认角差", "告诉策略每个关节离默认站姿有多远。"],
        ["dof_vel", "nu 维", "backend dof velocity", "关节速度，用于阻尼和判断腿部摆动速度。"],
        ["last_actions", "nu 维", "info['last_actions']", "上一控制步动作，帮助策略形成平滑控制。"],
        ["critic privileged", "任务相关", "`critic` obs group", "训练 critic 可读取更多速度或地形信息，actor 不一定可见。"],
    ],
    "humanoid_locomotion": [
        ["command", "3 维", "`Commands`", "期望前进、横移和转向速度。"],
        ["gait_phase", "2 或更多", "`gait_phase` / phase target", "左右腿步态相位，决定摆动脚和支撑脚的节律。"],
        ["gyro / gravity", "各 3 维", "IMU 传感器", "反映躯干角速度和倾斜方向，是防摔核心输入。"],
        ["dof_pos - default", "nu 维", "keyframe `stand` 差值", "全身关节相对默认站姿的偏移。"],
        ["dof_vel", "nu 维", "backend dof velocity", "全身关节速度。"],
        ["last_actions", "nu 维", "动作历史", "让策略观察自身上一帧控制目标。"],
        ["critic base linvel", "3 维", "critic 特权观测", "训练 critic 使用真实 base 线速度，提高 value 学习稳定性。"],
    ],
    "footstand": [
        ["linvel", "3 维", "`_FOOTSTAND_FRAME_OBS_DIM` 注释", "base 局部线速度，用于判断是否在姿态任务中漂移。"],
        ["gyro", "3 维", "`sensor.gyro`", "躯干角速度，判断倒立/前足站立是否稳定。"],
        ["gravity", "3 维", "`_get_local_gravity()`", "局部重力方向，是姿态误差的主要观测。"],
        ["diff", "12 维", "当前关节角与目标/默认角差", "腿部 12 个关节的姿态误差。"],
        ["dof_vel", "12 维", "关节速度", "腿部运动速度。"],
        ["last_action", "12 维", "上一帧动作", "动作平滑和控制滞后建模。"],
        ["history stack", "45 × history", "`obs_history_len`", "把单帧 45 维堆叠为短时序观测，默认至少 15 帧。"],
        ["critic tail", "49 维", "`_FOOTSTAND_PRIVILEGED_TAIL_DIM`", "critic 附加当前特权状态，用于训练但不直接部署。"],
    ],
    "motion_tracking": [
        ["current robot state", "随 G1 nu 变化", "`G1MotionTrackingEnv`", "当前 root、关节、末端和 body 状态。"],
        ["reference root", "位置/朝向/速度", "`MotionLoader`", "参考 motion 中 root 的目标位置、朝向和速度。"],
        ["reference body", "body_names × 状态", "`body_names`", "参考 motion 中关键 body 的位置、朝向、线速度和角速度。"],
        ["reference joints", "nu 维", "`motion_data.joint_pos`", "参考 motion 的关节角。"],
        ["phase / sampling", "clip 时间相关", "`MotionSampler`", "当前 episode 对应 motion clip 的时间位置。"],
        ["last_actions", "nu 维", "info 动作历史", "帮助策略保持动作连续。"],
        ["critic extras", "任务相关", "SAC/Deploy 变体", "SAC 变体可给 critic 额外 base linvel 或全身特权信息。"],
    ],
    "manipulation": [
        ["hand dof pos", "nu 维", "手部关节角", "表示每根手指当前弯曲/张开状态。"],
        ["target / prev ctrl", "nu 维", "上一帧控制目标", "让策略知道当前 actuator 目标和动作历史。"],
        ["object pose", "3+4 维", "物体位置与四元数", "被操作物体在手中的位置和朝向。"],
        ["object velocity", "3 或 6 维", "物体线速度/角速度", "判断旋转速度和是否被甩出。"],
        ["contact / tactile", "任务相关", "contact sensors / tactile obs", "手指接触和触觉信息，帮助稳定抓持。"],
        ["obs history", "lag steps × frame", "`obs_lag_steps`", "把短时间内手-物体状态串起来，改善部分可观测问题。"],
        ["critic info", "任务相关", "`critic_info_dim` / priv_info", "训练 critic 使用的特权状态，例如物体误差、摩擦或 scale 信息。"],
    ],
}


def _joint_chinese_name(robot_slug: str, joint_name: str, joint_type: str) -> str:
    if joint_type == "free":
        if "ball" in joint_name:
            return "足球或球体的 6 自由度自由关节"
        if "object" in joint_name:
            return "被操作物体的 6 自由度自由关节"
        return "机器人浮动基座自由关节"

    quadruped_prefix = {
        "FR": "右前腿",
        "FL": "左前腿",
        "RR": "右后腿",
        "RL": "左后腿",
    }
    for prefix, limb in quadruped_prefix.items():
        if joint_name.startswith(f"{prefix}_"):
            if "hip" in joint_name:
                return f"{limb}髋外展/内收关节"
            if "thigh" in joint_name:
                return f"{limb}髋俯仰（大腿）关节"
            if "calf" in joint_name:
                return f"{limb}膝关节（小腿摆动）"
            if "wheel" in joint_name:
                return f"{limb}轮子滚转关节"

    if re.match(r"joint[1-6]$", joint_name):
        return f"机械臂第 {joint_name[-1]} 轴关节"

    lower = joint_name.lower()
    side = ""
    if lower.startswith("left_") or lower.startswith("left"):
        side = "左"
    elif lower.startswith("right_") or lower.startswith("right"):
        side = "右"

    body_map = {
        "hip": "髋",
        "knee": "膝",
        "ankle": "踝",
        "waist": "腰",
        "shoulder": "肩",
        "elbow": "肘",
        "wrist": "腕",
        "head": "头部",
        "neck": "颈部",
    }
    axis_map = {
        "pitch": "俯仰",
        "roll": "侧摆",
        "yaw": "偏航",
    }
    for body_key, body_cn in body_map.items():
        if body_key in lower:
            axis = next((axis_cn for axis_key, axis_cn in axis_map.items() if axis_key in lower), "")
            return f"{side}{body_cn}{axis}关节" if axis else f"{side}{body_cn}关节"

    allegro_match = re.match(r"([fmr]f|th)j(\d+)", lower)
    if allegro_match:
        finger = {"ff": "食指", "mf": "中指", "rf": "无名指", "th": "拇指"}[allegro_match.group(1)]
        return f"{finger}第 {int(allegro_match.group(2)) + 1} 个弯曲/展开关节"

    sharpa_finger = {
        "thumb": "拇指",
        "index": "食指",
        "middle": "中指",
        "ring": "无名指",
        "pinky": "小指",
    }
    for key, finger in sharpa_finger.items():
        if key in lower:
            if "cmc" in lower:
                joint_cn = "腕掌"
            elif "mcp" in lower:
                joint_cn = "掌指"
            elif "pip" in lower:
                joint_cn = "近端指间"
            elif "dip" in lower:
                joint_cn = "远端指间"
            elif "_ip" in lower or lower.endswith("ip"):
                joint_cn = "指间"
            else:
                joint_cn = "手指"
            if "fe" in lower:
                axis = "屈伸"
            elif "aa" in lower:
                axis = "外展/内收"
            else:
                axis = ""
            return f"{side}{finger}{joint_cn}{axis}关节"

    if robot_slug == "k1":
        return f"K1 机器人关节：{joint_name}"
    return f"机器人关节：{joint_name}"


def _obs_family(task: TaskDoc) -> str:
    if task.slug == "go2_footstand":
        return "footstand"
    if task.category == "motion_tracking":
        return "motion_tracking"
    if task.category == "manipulation":
        return "manipulation"
    if task.robot in {"g1", "k1"}:
        return "humanoid_locomotion"
    return "quadruped_locomotion"


def _obs_breakdown_table(task: TaskDoc) -> str:
    return _table(["观测字段", "维度/规模", "代码来源", "中文说明"], OBS_FAMILY_ROWS[_obs_family(task)])


def _controlled_actuator_rows(task: TaskDoc, robot_inventory: dict[str, dict[str, Any]]) -> list[list[Any]]:
    model = robot_inventory[task.robot]["model"]
    joint_by_name = {joint["name"]: joint for joint in model["joints"]}
    rows: list[list[Any]] = []
    for index, actuator in enumerate(model["actuators"]):
        joint = joint_by_name.get(actuator["joint"])
        cn = (
            _joint_chinese_name(task.robot, joint["name"], joint["type"])
            if joint is not None
            else f"执行器目标：{actuator['joint']}"
        )
        ctrl = (
            f"{actuator['ctrlrange'][0]:.4g} ~ {actuator['ctrlrange'][1]:.4g}"
            if actuator["ctrllimited"]
            else "XML 未显式限制，实际由 env 缩放/默认角和 backend 控制"
        )
        rows.append([index, actuator["name"], actuator["joint"], cn, ctrl])
    return rows


def _action_detail_text(task: TaskDoc) -> str:
    if task.robot in {"go1", "go2"}:
        return (
            "四足任务的 action 逐维对应四条腿的 hip/thigh/calf actuator。"
            "env 在 `apply_action` 中把策略输出乘以 `action_scale`，再加到 keyframe 默认角上；"
            "因此策略学习的是“相对默认站姿的目标角偏移”，不是直接力矩。"
        )
    if task.robot == "go2w":
        return (
            "Go2W 的 action 包含 12 个腿部关节和 4 个轮关节。腿部仍是位置目标，"
            "轮部由 Go2W 专用控制逻辑解释，文档表中把轮关节单独标出。"
        )
    if task.robot == "g1":
        return (
            "G1 的 action 覆盖 29 个可控关节：腿、腰和双臂都参与控制。"
            "行走任务更强调腿部和腰部稳定，motion tracking 则让上肢也跟随参考动作。"
        )
    if task.robot == "k1":
        return (
            "K1 的 action 覆盖 22 个可控关节。足球盘带任务不直接控制足球，"
            "球体运动完全来自脚与球的物理接触。"
        )
    if task.robot == "go2_arm":
        return "移动操作任务的 action 同时控制四足底盘和 6 DoF 机械臂，策略需要在移动稳定和末端目标之间折中。"
    return "灵巧手任务的 action 逐维对应手指 actuator，物体自由关节不在 action 中，由接触动力学被动演化。"


def _reward_explanation(name: str) -> str:
    if name in REWARD_EXPLANATIONS:
        return REWARD_EXPLANATIONS[name]
    for key, explanation in REWARD_EXPLANATIONS.items():
        if key in name or name in key:
            return explanation
    if "contact" in name:
        return "接触相关奖励/惩罚，约束指定 body 或 geom 的接触状态。"
    if "vel" in name:
        return "速度相关奖励/惩罚，用于跟踪目标速度或抑制不希望的运动。"
    if "pos" in name:
        return "位置相关奖励/惩罚，用于跟踪目标位置或避免偏离安全区域。"
    if "ori" in name or "orientation" in name:
        return "朝向相关奖励/惩罚，用于约束姿态或参考朝向。"
    return "仓库 owner YAML 中声明的奖励项；具体实现见本页源码片段和 env 的 reward dispatch。"


def _reward_rows(reward_cfg: dict[str, Any]) -> list[list[Any]]:
    scales = reward_cfg.get("scales", {})
    rows: list[list[Any]] = []
    for name, scale in scales.items():
        if isinstance(scale, (int, float)):
            direction = "奖励" if scale > 0 else "惩罚" if scale < 0 else "记录/关闭"
        else:
            direction = "配置项"
        rows.append([name, scale, direction, _reward_explanation(str(name))])
    return rows


def _config_rows(config: dict[str, Any], prefix: str = "") -> list[list[Any]]:
    rows: list[list[Any]] = []
    for key, value in sorted(config.items()):
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            rows.extend(_config_rows(value, full_key))
        else:
            rows.append([full_key, value, _config_explanation(full_key, value)])
    return rows


def _config_explanation(key: str, value: Any) -> str:
    lower = key.lower()
    if "action_scale" in lower:
        return "策略输出到目标关节角的缩放系数。"
    if "clip_actions" in lower:
        return "动作裁剪范围，防止策略输出过大。"
    if "noise" in lower or "scale_" in lower:
        return "观测噪声或传感器噪声缩放，用于提升鲁棒性。"
    if "randomize" in lower:
        return "域随机化开关。"
    if "range" in lower:
        return "随机采样范围或控制范围。"
    if "threshold" in lower:
        return "终止或奖励判定阈值。"
    if "height" in lower:
        return "高度目标或高度约束。"
    if "sigma" in lower:
        return "指数型 tracking reward 的宽度参数。"
    if isinstance(value, bool):
        return "布尔开关。"
    return "任务 owner YAML 中的配置字段。"


def _source_symbols(path_str: str, topic: str) -> list[list[str]]:
    path = REPO_ROOT / path_str
    text = path.read_text(encoding="utf-8")
    patterns = {
        "obs": ("obs_groups_spec", "_get_obs", "_compute_obs", "_build_single_frame_obs"),
        "action": ("apply_action", "_init_action_space", "compute_go2w_motor_ctrl", "_dof_to_ctrl_order"),
        "reward": ("_init_reward_functions", "_compute_reward", "_reward_", "RewardConfig"),
    }
    rows: list[list[str]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not (stripped.startswith("def ") or stripped.startswith("class ") or stripped.startswith("@property")):
            continue
        if any(token in stripped for token in patterns[topic]):
            symbol = stripped.split("(")[0].replace("def ", "").replace("class ", "")
            rows.append([symbol, _symbol_explanation(symbol, topic)])
    return rows[:12]


def _symbol_explanation(symbol: str, topic: str) -> str:
    lower = symbol.lower()
    if "obs_groups_spec" in lower:
        return "声明 obs dict 中各观测组的维度，是 wrapper/learner 对齐观测的契约。"
    if "compute_obs" in lower or "get_obs" in lower or "single_frame" in lower:
        return "读取 backend 传感器和 info 字段，拼接 actor/critic 观测。"
    if "action_space" in lower:
        return "从 backend actuator control range 推导 action space。"
    if "apply_action" in lower:
        return "把策略输出缩放为执行器控制目标。"
    if "rewardconfig" in lower:
        return "声明 reward 可用参数和默认 scale。"
    if "init_reward_functions" in lower:
        return "把 reward 名称映射到源码函数，owner YAML 的 scale 会按这些 key 分发。"
    if "_reward_" in lower:
        return "单个 reward 项的实现函数，通常返回每个 env 的 reward 向量。"
    return f"{topic} 相关源码锚点。"


def rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def required_path(data: dict[str, Any], keys: tuple[str, ...], source: Path) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            joined = ".".join(keys)
            raise KeyError(f"{rel(source)} missing required key: {joined}")
        current = current[key]
    return current


def load_owner_configs() -> dict[str, list[dict[str, Any]]]:
    owners: dict[str, list[dict[str, Any]]] = {}
    for path in sorted((REPO_ROOT / "conf").glob("**/task/**/*.yaml")):
        parts = path.parts
        if "params_bk" in parts:
            continue
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        training = data.get("training", {})
        if "task_name" not in training or "sim_backend" not in training:
            # Hydra profile variants (for example HORA overlays) inherit a full owner
            # through defaults and intentionally do not restate the task identity.
            continue
        task_name = required_path(data, ("training", "task_name"), path)
        sim_backend = required_path(data, ("training", "sim_backend"), path)
        slug = path.parent.name
        if path.parent.parent.name in {"sac", "td3", "flashsac"}:
            slug = path.parent.name
        owners.setdefault(slug, []).append(
            {
                "path": rel(path),
                "task_name": task_name,
                "sim_backend": sim_backend,
                "algo_group": _algo_group_from_path(path),
                "data": data,
            }
        )
    return owners


def _algo_group_from_path(path: Path) -> str:
    rel_parts = path.relative_to(REPO_ROOT).parts
    if rel_parts[:3] == ("conf", "offpolicy", "task"):
        return rel_parts[3]
    return rel_parts[1]


def load_registry() -> dict[str, dict[str, Any]]:
    registry.ensure_registries()
    raw = registry.list_registered_envs()
    out: dict[str, dict[str, Any]] = {}
    for name, meta in sorted(raw.items()):
        cfg_cls = registry._envs[name].env_cfg_cls  # type: ignore[attr-defined]
        env_sources = {}
        for backend, env_cls in registry._envs[name].env_cls_dict.items():  # type: ignore[attr-defined]
            source = inspect.getsourcefile(env_cls)
            env_sources[backend] = rel(Path(source)) if source else ""
        cfg_source = inspect.getsourcefile(cfg_cls)
        out[name] = {
            "config_class": meta["config_class"],
            "available_backends": meta["available_backends"],
            "config_source": rel(Path(cfg_source)) if cfg_source else "",
            "env_sources": env_sources,
        }
    return out


def _mj_name(model: mujoco.MjModel, obj: mujoco.mjtObj, idx: int) -> str:
    name = mujoco.mj_id2name(model, obj, idx)
    return name if name else f"<unnamed:{idx}>"


def _joint_type_name(value: int) -> str:
    names = {
        int(mujoco.mjtJoint.mjJNT_FREE): "free",
        int(mujoco.mjtJoint.mjJNT_BALL): "ball",
        int(mujoco.mjtJoint.mjJNT_SLIDE): "slide",
        int(mujoco.mjtJoint.mjJNT_HINGE): "hinge",
    }
    return names[value]


def compile_model_info(xml_path: Path) -> dict[str, Any]:
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    actuator_joint_ids: set[int] = set()
    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        if joint_id >= 0:
            actuator_joint_ids.add(joint_id)

    joints: list[dict[str, Any]] = []
    for joint_id in range(model.njnt):
        limited = bool(model.jnt_limited[joint_id])
        range_low, range_high = model.jnt_range[joint_id]
        joints.append(
            {
                "id": joint_id,
                "name": _mj_name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id),
                "type": _joint_type_name(int(model.jnt_type[joint_id])),
                "limited": limited,
                "range": [float(range_low), float(range_high)] if limited else None,
                "controlled": joint_id in actuator_joint_ids,
            }
        )

    actuators: list[dict[str, Any]] = []
    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        joint_name = _mj_name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) if joint_id >= 0 else ""
        ctrl_low, ctrl_high = model.actuator_ctrlrange[actuator_id]
        actuators.append(
            {
                "id": actuator_id,
                "name": _mj_name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id),
                "joint": joint_name,
                "ctrllimited": bool(model.actuator_ctrllimited[actuator_id]),
                "ctrlrange": [float(ctrl_low), float(ctrl_high)],
            }
        )

    sensors: list[dict[str, Any]] = []
    for sensor_id in range(model.nsensor):
        sensors.append(
            {
                "id": sensor_id,
                "name": _mj_name(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_id),
                "dim": int(model.sensor_dim[sensor_id]),
            }
        )

    keyframes = [_mj_name(model, mujoco.mjtObj.mjOBJ_KEY, key_id) for key_id in range(model.nkey)]
    return {
        "xml": rel(xml_path),
        "nq": int(model.nq),
        "nv": int(model.nv),
        "nu": int(model.nu),
        "njnt": int(model.njnt),
        "nsensor": int(model.nsensor),
        "keyframes": keyframes,
        "joints": joints,
        "actuators": actuators,
        "sensors": sensors,
    }


def build_robot_inventory() -> dict[str, dict[str, Any]]:
    inventory: dict[str, dict[str, Any]] = {}
    for slug, robot in ROBOTS.items():
        scene = REPO_ROOT / robot.primary_scene
        info = compile_model_info(scene)
        asset_files = [
            rel(path)
            for path in sorted((ROBOTS_ROOT / slug).rglob("*"))
            if path.is_file() and not path.name.startswith("tmp")
        ]
        inventory[slug] = {
            "doc": robot.__dict__,
            "model": info,
            "asset_files": asset_files,
        }
    return inventory


def copy_robot_assets(robot_inventory: dict[str, dict[str, Any]]) -> dict[str, Any]:
    manifest: dict[str, Any] = {"robots": {}}
    for slug in robot_inventory:
        src = ROBOTS_ROOT / slug
        dst = DOC_ROOT / "robots" / slug / "assets"
        shutil.copytree(
            src,
            dst,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("tmp*.xml", "__pycache__", "*.pyc"),
        )
        copied = [rel(path) for path in sorted(dst.rglob("*")) if path.is_file()]
        manifest["robots"][slug] = {
            "source": rel(src),
            "destination": rel(dst),
            "files": copied,
        }
    return manifest


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    def cell(value: Any) -> str:
        text = "" if value is None else str(value)
        return text.replace("|", "\\|").replace("\n", "<br>")

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(cell(value) for value in row) + " |")
    return "\n".join(lines)


def _yaml_block(value: Any) -> str:
    if value is None:
        return "null\n"
    return yaml.safe_dump(value, allow_unicode=True, sort_keys=False)


def _source_snippet(path_str: str, needles: tuple[str, ...], context: int = 5) -> str:
    path = REPO_ROOT / path_str
    lines = path.read_text(encoding="utf-8").splitlines()
    hit = -1
    for needle in needles:
        for index, line in enumerate(lines):
            if needle in line:
                hit = index
                break
        if hit >= 0:
            break
    if hit < 0:
        return f"```text\n# 未在 {path_str} 中找到片段：{', '.join(needles)}\n```"
    start = max(0, hit - context)
    end = min(len(lines), hit + context + 1)
    snippet = "\n".join(lines[start:end])
    return f"```python\n# {path_str}\n{snippet}\n```"


def _line_indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _find_code_block(path_str: str, needles: tuple[str, ...], max_lines: int = 90) -> tuple[str, str] | None:
    path = REPO_ROOT / path_str
    lines = path.read_text(encoding="utf-8").splitlines()
    hit = -1
    matched = ""
    for needle in needles:
        for index, line in enumerate(lines):
            stripped = line.strip()
            if (
                stripped.startswith(f"def {needle}")
                or stripped.startswith(f"class {needle}")
                or stripped.startswith(f"async def {needle}")
            ):
                hit = index
                matched = needle
                break
        if hit >= 0:
            break
    if hit < 0:
        for needle in needles:
            for index, line in enumerate(lines):
                stripped = line.strip()
                if stripped.startswith(("import ", "from ")):
                    continue
                if needle in stripped:
                    hit = index
                    matched = needle
                    break
            if hit >= 0:
                break
    if hit < 0:
        return None

    start = hit
    if lines[hit].lstrip().startswith("@"):
        scan = hit + 1
        while scan < len(lines) and lines[scan].lstrip().startswith("@"):
            scan += 1
        if scan < len(lines) and lines[scan].lstrip().startswith(("def ", "class ", "async def ")):
            hit = scan
    elif not lines[hit].lstrip().startswith(("def ", "class ", "async def ")):
        current_indent = _line_indent(lines[hit])
        for scan in range(hit - 1, -1, -1):
            stripped = lines[scan].strip()
            if not stripped:
                continue
            if _line_indent(lines[scan]) < current_indent and stripped.startswith(
                ("def ", "class ", "async def ")
            ):
                hit = scan
                break
    start = hit
    while start > 0 and lines[start - 1].lstrip().startswith("@"):
        start -= 1

    indent = _line_indent(lines[hit])
    end = min(len(lines), start + max_lines)
    for index in range(hit + 1, min(len(lines), hit + max_lines)):
        stripped = lines[index].strip()
        if not stripped:
            continue
        if _line_indent(lines[index]) <= indent and (
            stripped.startswith("def ")
            or stripped.startswith("class ")
            or stripped.startswith("@registry.")
            or stripped.startswith("@dataclass")
        ):
            end = index
            break
    snippet = "\n".join(lines[start:end])
    return matched, f"```python\n# {path_str}\n{snippet}\n```"


def _source_topic_snippet(path_str: str, needles: tuple[str, ...], max_lines: int = 90) -> str:
    block = _find_code_block(path_str, needles, max_lines=max_lines)
    if block is None:
        return f"```text\n# 未在 {path_str} 中找到代码块：{', '.join(needles)}\n```"
    return block[1]


def _task_execution_flow_rows(task: TaskDoc, owner_path: str) -> list[list[str]]:
    return [
        [
            "1. Hydra owner",
            owner_path,
            "训练命令先 compose owner YAML。这里确定 `training.task_name`、后端身份、obs_groups、reward scale 和 env override。",
        ],
        [
            "2. Registry 构造 Env",
            task.source_hint,
            "`@registry.envcfg` 注册配置类，`@registry.env` 注册后端实现类；训练脚本只调用 registry，不直接 new 任务类。",
        ],
        [
            "3. Reset/初始状态",
            task.scene_hint,
            "env 从 scene keyframe 或 motion/grasp cache 取初态，再通过 reset randomization 改写 qpos/qvel/info。",
        ],
        [
            "4. Obs dict",
            task.source_hint,
            "env 读取 backend 传感器和 info，返回 dict；learner 根据 `obs_groups_spec` 和 owner `algo.obs_groups` 选择 actor/critic 输入。",
        ],
        [
            "5. Action 到控制目标",
            task.source_hint,
            "policy 输出先按 action scale / 控制顺序映射到 actuator，再由 backend step 推进仿真。",
        ],
        [
            "6. Reward dispatch",
            task.source_hint,
            "源码把 reward key 映射到函数，owner YAML 的 scale 决定每项奖励/惩罚是否启用以及权重。",
        ],
        [
            "7. Termination",
            task.source_hint,
            "env 根据高度、姿态、接触、参考轨迹误差、物体状态或能量阈值生成 done/terminated。",
        ],
    ]


def _topic_explanation(topic: str, task: TaskDoc, matched: str) -> list[str]:
    explanations = {
        "注册与配置": [
            f"`{matched}` 是任务进入 registry/Hydra 体系的入口。",
            "这里的字段属于 owner layer：任务身份、默认 scene、reward_config、domain_rand 和控制参数都应从配置注入，而不是在训练脚本里硬编码。",
            "读这段代码时要关注 dataclass 字段名，因为 owner YAML 的 `env:` 和 `reward:` 最终会覆盖这些字段。",
        ],
        "Reset 与初始状态": [
            f"`{matched}` 负责 episode 开始时的状态来源。",
            "关键是 qpos/qvel/info 三类数据：qpos/qvel 写入仿真状态，info 保存 commands、last_actions、obs history、motion reference 等策略需要的上下文。",
            "如果这里出现 domain randomization，它也应该只在 reset 冷路径改写物理量或初态。",
        ],
        "Obs 构造": [
            f"`{matched}` 展示 actor/critic 观测如何从 backend 传感器和 info 拼出来。",
            "`obs_groups_spec` 是维度契约；实际数组拼接通常在 `_compute_obs`、`_get_obs` 或 `_build_single_frame_obs` 中完成。",
            "文档中的观测字段表就是按这段代码的拼接顺序和语义拆出来的。",
        ],
        "Action 映射": [
            f"`{matched}` 是 policy action 进入仿真控制器的关键路径。",
            "常见模式是 `actions * action_scale + default_angles`；轮足、移动操作和灵巧手可能还会做控制顺序转换或 PD 目标计算。",
            "这段代码决定了 action 的物理含义，因此动作表按 actuator 顺序列出每一维。",
        ],
        "Reward 分发": [
            f"`{matched}` 说明 reward key 如何连接到具体函数。",
            "owner YAML 只给 scale，源码函数才定义每个 reward 的计算方式；scale 为 0 时通常表示保留接口但当前任务关闭。",
            "阅读 reward 时要同时看 `_init_reward_functions`、单项 `_reward_*` 函数和 owner YAML 的 `reward.scales`。",
        ],
        "Termination 终止": [
            f"`{matched}` 是失败/截断条件的实现入口。",
            "这里通常会组合高度、姿态、接触、能量、参考轨迹误差或物体状态阈值。",
            "终止条件会影响 reward 中的 termination penalty，也影响 reset 何时重新采样初态。",
        ],
        "Domain Randomization": [
            f"`{matched}` 说明随机化参数如何进入 backend。",
            "随机化应在 init/reset/interval 等冷路径执行，不在 step 热路径解析资产或探测 backend 私有能力。",
            "字段名通常和 owner YAML 的 `env.domain_rand` 对齐，文档中的随机化表按这些字段解释。",
        ],
    }
    return explanations[topic]


def _explain_code_line(stripped: str, topic: str) -> str | None:
    if stripped.startswith("@registry.envcfg"):
        return "把 EnvCfg 注册到 UniLab registry，Hydra compose 后会通过这个名字找到配置类。"
    if stripped.startswith("registry.register_env"):
        return "把任务名、Env 类和 backend 绑定起来；训练入口只需要任务名和 backend 即可实例化环境。"
    if stripped.startswith("class ") and "Cfg" in stripped:
        return "定义任务配置类；owner YAML 的 `env:` 字段会覆盖这里的默认字段。"
    if stripped.startswith("class ") and "Reward" in stripped:
        return "定义 reward 参数集合；这些参数会被单项 reward 函数读取。"
    if "model_file=" in stripped or "SceneCfg(" in stripped:
        return "声明任务使用的 scene/XML 入口，资产路径只在配置/初始化冷路径解析。"
    if stripped.startswith("reward_config"):
        return "指定本任务使用的 reward 配置类型，避免训练脚本解释 reward 细节。"
    if stripped.startswith("domain_rand"):
        return "指定本任务的域随机化配置，reset/interval 随机化会读取这些字段。"
    if stripped.startswith("def build_reset_plan"):
        return "构造 reset plan：批量生成 qpos、qvel、info_updates 和随机化 payload。"
    if "motion_loader.get_motion_at_frame" in stripped:
        return "从参考动作数据集中读取当前采样帧，用于 motion tracking 初始状态和目标。"
    if "qpos" in stripped and "np.tile" in stripped:
        return "把默认 qpos 复制到所有需要 reset 的环境，形成批量初始姿态。"
    if "qvel" in stripped and ("zeros_like" in stripped or "np.zeros" in stripped):
        return "初始化速度，通常 reset 时从静止或参考 motion 速度开始。"
    if "info_updates" in stripped:
        return "准备写入 state.info 的 episode 上下文，例如命令、动作历史、参考轨迹或物体状态。"
    if "ResetPlan(" in stripped:
        return "把 reset 所需的状态和随机化 payload 交给 runner/backend 统一应用。"
    if "randomization=" in stripped:
        return "reset 阶段携带的物理随机化数据；为空时本次 reset 不额外改写物理参数。"
    if stripped.startswith("def obs_groups_spec"):
        return "声明 obs dict 中每个观测组的维度，是 env、wrapper、learner 对齐的契约。"
    if 'return {"obs"' in stripped or 'return {"policy"' in stripped:
        return "返回观测组维度；actor/critic 会按这些 key 读取对应数组。"
    if "np.concatenate" in stripped and "Action" in topic:
        return "按 actuator 顺序拼接不同控制目标；轮足任务中通常是腿部位置目标加轮部速度目标。"
    if "np.concatenate" in stripped:
        return "把多个传感器/状态字段按固定顺序拼接成策略输入向量。"
    if "get_local_linvel" in stripped:
        return "读取机体坐标系下的 base 线速度，常用于 critic 特权观测或速度奖励。"
    if "get_gyro" in stripped:
        return "读取 IMU 角速度，是姿态稳定和倾倒检测的重要输入。"
    if "get_dof_pos" in stripped:
        return "读取可控关节位置，用于观测、姿态奖励和 action 偏移计算。"
    if "get_dof_vel" in stripped:
        return "读取关节速度，用于观测和速度/平滑惩罚。"
    if "state.info[" in stripped and "=" in stripped:
        return "更新 episode 级缓存；这些字段会被后续 obs/reward/action 逻辑复用。"
    if stripped.startswith(('"', "'")) and ":" in stripped and "Reward" in topic:
        return "reward 映射项：左侧是 owner YAML 中的 reward key，右侧是实际计算函数。"
    if stripped.startswith(('"', "'")) and "," in stripped:
        return "读取或写入这一项任务状态字段，通常会进入 info、obs 或 reset payload。"
    if stripped.startswith(("axis=", "dtype=")):
        return "指定数组拼接/转换的参数，保证输出维度和数值类型符合 env contract。"
    if "np.asarray" in stripped:
        return "把中间结果转换成 NumPy 数组，并对齐 UniLab 当前使用的数值 dtype。"
    if "np.zeros" in stripped:
        return "构造零初始化数组，通常用于 reset 初态、动作历史或默认缓存。"
    if "np.broadcast_to" in stripped:
        return "把单个默认状态扩展成并行环境批量形状，方便一次 reset 多个 env。"
    if "np.column_stack" in stripped:
        return "按列拼接多个一维信号，形成策略需要的相位或目标向量。"
    if "gait_phase" in stripped and "=" in stripped:
        return "更新步态相位缓存，后续 obs 和步态 reward 会读取它。"
    if "commands" in stripped and "=" in stripped:
        return "写入或读取速度/目标命令，policy 通过该字段知道当前要完成的目标。"
    if stripped.startswith("def apply_action"):
        return "策略输出进入控制器的入口函数，返回 backend actuator control。"
    if "simulate_action_latency" in stripped:
        return "根据配置决定是否用上一帧动作模拟真实控制延迟。"
    if "actions *" in stripped and "action_scale" in stripped:
        return "把归一化策略动作缩放成关节角偏移。"
    if "default_angles" in stripped:
        return "以默认站姿/参考姿态为中心，action 学习的是相对偏移而非绝对力矩。"
    if stripped.startswith("return ctrl") or stripped.startswith("return np.asarray(ctrl"):
        return "返回最终控制目标，backend step 会把它写入 actuator control。"
    if stripped.startswith("def _init_reward_functions"):
        return "建立 reward key 到函数的映射表，owner YAML 的 scale 通过 key 找到实现函数。"
    if "self._reward_fns.update" in stripped:
        return "在父类 reward 映射基础上追加当前任务专属奖励项。"
    if stripped.startswith("def _reward_"):
        return "单项 reward 实现；通常返回每个并行 env 的 reward 向量。"
    if "np.exp" in stripped:
        return "指数型 tracking reward：误差越小越接近 1，误差增大后快速衰减。"
    if "np.linalg.norm" in stripped:
        return "计算向量距离/范数，常用于位置误差、速度误差或球距离。"
    if "np.clip" in stripped:
        return "裁剪数值范围，避免超出物理/数学有效区间。"
    if "terminated" in stripped and "=" in stripped:
        return "生成终止布尔量；为 True 的环境会在 runner 生命周期中 reset。"
    if "np.logical_or" in stripped or "np.logical_and" in stripped:
        return "组合多个失败条件，任何一个满足或同时满足时触发终止/惩罚。"
    if "max_tilt_rad" in stripped or "max_tilt_deg" in stripped:
        return "把最大倾斜角阈值用于摔倒检测。"
    if "min_base_height" in stripped:
        return "检查机体高度，低于阈值通常表示跌倒或异常姿态。"
    if "randomize_" in stripped or "friction_range" in stripped:
        return "读取域随机化开关或采样范围。"
    if "base_kp" in stripped or "base_kd" in stripped:
        return "保存默认 PD 增益，随机化时以默认值为基准缩放。"
    return None


WEAK_SOURCE_ANCHORS = {
    "default_angles",
    "simulate_action_latency",
    "clip_actions",
    "action_scale",
    "PenaltyCurriculum",
    "WBT",
    "Grasp-cache",
    "success",
}


def _fallback_code_comment(stripped: str, topic: str) -> str:
    if stripped.startswith("#"):
        return "保留源码中的注释，用来说明这一段实现的上下文。"
    if stripped.startswith(("from ", "import ")):
        return "导入本段代码依赖的类型、工具函数或任务组件。"
    if stripped.startswith("@"):
        return "装饰器会在类或函数定义前附加注册、dataclass 等元信息。"
    if stripped.startswith("def "):
        return "定义本节要解释的函数入口，后续代码都围绕这个职责展开。"
    if stripped.startswith("class "):
        return "定义本节要解释的类，任务配置或环境行为会从这里开始扩展。"
    if stripped.startswith(("if ", "elif ")):
        return "根据当前状态或配置选择不同执行路径。"
    if stripped == "else:":
        return "处理上一个条件不满足时的备用执行路径。"
    if stripped.startswith("for "):
        return "遍历一组配置、环境编号或 reward 项，逐个执行同一类处理。"
    if stripped.startswith("while "):
        return "在条件满足时重复执行，通常用于搜索、采样或状态推进。"
    if stripped.startswith("return "):
        return "返回本函数计算出的结果，交给 env、runner 或 backend 的下一阶段使用。"
    if stripped.startswith("assert "):
        return "声明这里必须满足的运行时不变量，失败时说明配置或状态不符合预期。"
    if stripped.startswith("del "):
        return "显式丢弃当前逻辑不使用的变量，说明这个任务不依赖该输入。"
    if stripped in {")", ")", "},", "}", "],", "]", "):"}:
        return "结束上一段函数调用、容器字面量或代码块。"
    if stripped.endswith("("):
        return "开始一次函数调用或对象构造，后续缩进行会继续填写参数。"
    if stripped.endswith("{"):
        return "开始构造字典，后续行会写入 key 和对应的任务状态。"
    if stripped.endswith("["):
        return "开始构造列表，后续行会按顺序写入观测、动作或配置字段。"
    if "=" in stripped and "==" not in stripped and "!=" not in stripped:
        lhs = stripped.split("=", 1)[0].strip()
        return f"计算或更新 `{lhs}`，供本节后续逻辑继续使用。"
    if ":" in stripped:
        return "声明字段、分支或映射关系，具体语义由本任务配置和上下文决定。"
    return f"执行 `{topic}` 小节中的一步具体逻辑，和上下文共同完成该任务行为。"


def _code_line_comment(stripped: str, topic: str) -> str:
    return _explain_code_line(stripped, topic) or _fallback_code_comment(stripped, topic)


def _annotate_python_snippet(snippet: str, topic: str) -> str:
    lines = snippet.splitlines()
    if len(lines) < 2 or not lines[0].startswith("```python"):
        return snippet

    annotated: list[str] = [lines[0]]
    for line in lines[1:]:
        if line == "```":
            annotated.append(line)
            continue
        annotated.append(line)
        stripped = line.strip()
        if not stripped:
            continue
        indent = line[: len(line) - len(line.lstrip(" "))]
        annotated.append(f"{indent}# 中文注释：{_code_line_comment(stripped, topic)}")
    return "\n".join(annotated)


def _section_source_blocks(section: TaskCodeSection, default_source: str) -> tuple[str, str]:
    source_path = section.source or default_source
    snippets: list[str] = []
    seen: set[str] = set()
    missing: list[str] = []
    for anchor in section.anchors:
        if snippets and anchor in WEAK_SOURCE_ANCHORS:
            continue
        block = _find_code_block(source_path, (anchor,), max_lines=90)
        if block is None:
            missing.append(anchor)
            continue
        _, snippet = block
        if snippet in seen:
            continue
        snippets.append(snippet)
        seen.add(snippet)
        if len(snippets) >= 3:
            break
    if not snippets:
        snippets.append(
            "```text\n"
            f"# 未在 {source_path} 中找到本任务指定锚点：{', '.join(section.anchors)}\n"
            "```"
        )
    if missing and len(snippets) < 3:
        snippets.append(
            "```text\n"
            f"# 这些手写锚点未命中，需要人工复核：{', '.join(missing)}\n"
            "```"
        )
    return source_path, "\n\n".join(snippets)


def _code_walkthrough_section(
    task: TaskDoc, owner_path: str, registry_info: dict[str, dict[str, Any]]
) -> str:
    if task.slug not in TASK_CODE_GUIDES:
        raise KeyError(f"missing TaskCodeGuide for {task.slug}")
    guide = TASK_CODE_GUIDES[task.slug]

    blocks = []
    for section in guide.sections:
        source_path, snippet = _section_source_blocks(section, task.source_hint)
        manual_rows = [[note] for note in section.notes]
        blocks.append(
            f"""### {section.title}

{_table(["本任务人工导读"], manual_rows)}

下面是人工选择的源码证据片段。逐行解释需要在任务 Markdown 中人工撰写，不能由生成器自动套模板。

{snippet}
"""
        )
    focus_rows = [[item] for item in guide.focus]
    section_rows = [
        [section.title, section.source or task.source_hint, ", ".join(section.anchors)]
        for section in guide.sections
    ]
    registered_rows = [
        [
            name,
            registry_info[name]["config_class"],
            ", ".join(registry_info[name]["available_backends"]),
        ]
        for name in task.registry_names
    ]
    return f"""## 代码级执行流

这部分不再用通用关键词套模板，而是为 `{task.slug}` 单独写源码导读。源码片段只负责提供证据；每节说明都围绕本任务自己的配置、状态、观测、动作和 reward 语义展开。

{_table(["阶段", "证据位置", "代码级说明"], _task_execution_flow_rows(task, owner_path))}

### 本任务阅读重点

{_table(["阅读重点"], focus_rows)}

### 本任务注册入口

{_table(["注册名", "配置类", "可用后端"], registered_rows)}

### 本任务源码锚点

{_table(["小节", "源码文件", "明确锚点"], section_rows)}

## 关键源码逐段解释（按本任务手写）

{"".join(blocks)}
"""


def _xml_snippet(path_str: str, needles: tuple[str, ...], context: int = 3) -> str:
    path = REPO_ROOT / path_str
    lines = path.read_text(encoding="utf-8").splitlines()
    hit = -1
    for index, line in enumerate(lines):
        if any(needle in line for needle in needles):
            hit = index
            break
    if hit < 0:
        return f"```xml\n<!-- 未在 {path_str} 中找到片段：{', '.join(needles)} -->\n```"
    start = max(0, hit - context)
    end = min(len(lines), hit + context + 1)
    snippet = "\n".join(lines[start:end])
    return f"```xml\n<!-- {path_str} -->\n{snippet}\n```"


def _owner_rows(owners: list[dict[str, Any]]) -> list[list[Any]]:
    rows = []
    for owner in owners:
        rows.append([owner["algo_group"], owner["sim_backend"], owner["task_name"], owner["path"]])
    return rows


def _task_action_dim(task: TaskDoc, robot_inventory: dict[str, dict[str, Any]]) -> int | str:
    info = robot_inventory[task.robot]["model"]
    return info["nu"]


def _select_primary_owner(task: TaskDoc, owners_by_slug: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    owners = owners_by_slug[task.slug]
    for owner in owners:
        if owner["sim_backend"] == "mujoco":
            return owner
    return owners[0]


def _task_registry_summary(task: TaskDoc, registry_info: dict[str, dict[str, Any]]) -> str:
    rows = []
    for env_name in task.registry_names:
        reg = registry_info[env_name]
        rows.append(
            [
                env_name,
                ", ".join(reg["available_backends"]),
                reg["config_class"],
                reg["config_source"],
            ]
        )
    return _table(["注册名", "后端", "配置类", "配置源码"], rows)


def write_robot_docs(robot_inventory: dict[str, dict[str, Any]], asset_manifest: dict[str, Any]) -> None:
    for slug, payload in robot_inventory.items():
        robot = ROBOTS[slug]
        model = payload["model"]
        doc_path = DOC_ROOT / "robots" / slug / "index.md"
        doc_path.parent.mkdir(parents=True, exist_ok=True)

        joint_rows = []
        for joint in model["joints"]:
            range_text = "无" if joint["range"] is None else f"{joint['range'][0]:.4g} ~ {joint['range'][1]:.4g}"
            joint_rows.append(
                [
                    joint["id"],
                    joint["name"],
                    _joint_chinese_name(slug, joint["name"], joint["type"]),
                    joint["type"],
                    "是" if joint["limited"] else "否",
                    range_text,
                    "是" if joint["controlled"] else "否",
                ]
            )
        actuator_rows = [
            [
                actuator["id"],
                actuator["name"],
                actuator["joint"],
                _joint_chinese_name(
                    slug,
                    actuator["joint"],
                    next(
                        (
                            joint["type"]
                            for joint in model["joints"]
                            if joint["name"] == actuator["joint"]
                        ),
                        "hinge",
                    ),
                ),
                (
                    f"{actuator['ctrlrange'][0]:.4g} ~ {actuator['ctrlrange'][1]:.4g}"
                    if actuator["ctrllimited"]
                    else "未显式限制"
                ),
            ]
            for actuator in model["actuators"]
        ]
        sensor_rows = [
            [sensor["id"], sensor["name"], sensor["dim"]] for sensor in model["sensors"]
        ]
        files = asset_manifest["robots"][slug]["files"]
        asset_rows = [[path] for path in files[:80]]
        if len(files) > 80:
            asset_rows.append([f"... 其余 {len(files) - 80} 个文件见 `_generated/asset_manifest.json`"])

        content = f"""---
title: "{robot.title} 机器人资产与关节说明"
slug: "{slug}"
robot: "{slug}"
source_scene: "{robot.primary_scene}"
---

# {robot.title} 机器人资产与关节说明

![{robot.title} 默认姿态](../../images/robots/{slug}.png)

{robot.intro}

## 机器人定位

- 用途：{robot.role}
- 主场景：`{robot.primary_scene}`
- 默认 keyframe：`{robot.keyframe}`
- 资产快照：`robots/{slug}/assets/`

这份文档只把 `src/unilab/assets/robots/{slug}/` 下的机器人、场景、mesh、texture 等素材复制到博客目录。motion、checkpoint、cache 不属于机器人结构说明，未复制进文档资产目录。

## 编译模型概览

{_table(["字段", "数值", "说明"], [
    ["nq", model["nq"], "广义坐标维度，free joint 的四元数会占 4 维"],
    ["nv", model["nv"], "广义速度维度"],
    ["nu", model["nu"], "执行器数量，也就是默认 action 维度"],
    ["njnt", model["njnt"], "MuJoCo 编译后的 joint 数，包含 free/object joint"],
    ["nsensor", model["nsensor"], "传感器数量"],
    ["keyframes", ", ".join(model["keyframes"]), "scene/task 层 keyframe"],
])}

## 资产结构

<details>
<summary>已复制文件（默认折叠，点击展开）</summary>

{_table(["已复制文件"], asset_rows)}

</details>

关键 XML 片段如下。`<include>` 把纯机器人 XML 放入 scene，`<keyframe>` 留在 scene 或 task fragment 中，符合 UniLab 的资产分层约束。

{_xml_snippet(robot.primary_scene, ("<include", "<keyframe"))}

## 关节说明

`free` joint 表示浮动基座或任务物体自由度，不直接对应 policy action；`controlled=是` 表示该 joint 被 actuator 直接驱动，是 action 空间的一部分。“中文含义”按机器人运动学命名拆解，用于快速理解每个关节控制的身体部位和运动方向。

{_table(["ID", "关节名称", "中文含义", "类型", "有限位", "关节限制", "受控"], joint_rows)}

## 执行器说明

执行器表来自 MuJoCo 编译模型的 `actuator` 段。RL action 的每一维最终会映射到这些执行器的控制目标或控制范围。

{_table(["ID", "执行器名称", "驱动关节", "中文含义", "控制范围"], actuator_rows)}

## 传感器说明

传感器既包括机器人本体 IMU/速度传感器，也可能包括 scene/task fragment 增加的接触传感器。任务文档会说明哪些传感器进入 obs 或 reward。

{_table(["ID", "传感器名称", "维度"], sensor_rows)}
"""
        doc_path.write_text(content, encoding="utf-8")


def write_task_docs(
    owners_by_slug: dict[str, list[dict[str, Any]]],
    registry_info: dict[str, dict[str, Any]],
    robot_inventory: dict[str, dict[str, Any]],
    *,
    overwrite_existing: bool = False,
) -> None:
    for slug, task in TASKS.items():
        owners = owners_by_slug[slug]
        owner = _select_primary_owner(task, owners_by_slug)
        data = owner["data"]
        env_cfg = data.get("env", {})
        reward_cfg = data.get("reward", {})
        algo_cfg = data.get("algo", {})
        action_dim = _task_action_dim(task, robot_inventory)
        output = DOC_ROOT / "tasks" / task.category / f"{slug}.md"
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists() and not overwrite_existing:
            print(
                f"skip existing task doc with possible manual annotations: "
                f"{output.relative_to(REPO_ROOT)}"
            )
            continue

        domain_rand = env_cfg.get("domain_rand", {})
        control_cfg = env_cfg.get("control_config", {})
        obs_groups = algo_cfg.get("obs_groups", {})
        obs_symbol_rows = _source_symbols(task.source_hint, "obs")
        action_symbol_rows = _source_symbols(task.source_hint, "action")
        reward_symbol_rows = _source_symbols(task.source_hint, "reward")
        action_rows = _controlled_actuator_rows(task, robot_inventory)
        reward_rows = _reward_rows(reward_cfg)
        control_rows = _config_rows(control_cfg)
        domain_rand_rows = _config_rows(domain_rand)
        code_walkthrough = _code_walkthrough_section(task, owner["path"], registry_info)

        content = f"""---
title: "{task.title}"
slug: "{slug}"
category: "{task.category}"
robot: "{task.robot}"
registered_envs: [{", ".join(task.registry_names)}]
---

# {task.title}

## 任务目标

{task.goal}

![{task.title} 场景渲染](../../images/tasks/{slug}.png)

> 图片由 `_tools/render_static_images.py` 使用 `{task.scene_hint}` 的 MuJoCo scene 离屏渲染生成；如果当前环境无法启用 EGL/OSMesa，生成 manifest 会记录失败原因。

## 证据索引

{_table(["项目", "内容"], [
    ["任务类别", CATEGORY_TITLES[task.category]],
    ["机器人", task.robot],
    ["代表 scene", task.scene_hint],
    ["代表源码", task.source_hint],
    ["代表 owner YAML", owner["path"]],
    ["动作维度", action_dim],
])}

### Registry 与 Owner

{_task_registry_summary(task, registry_info)}

{_table(["算法组", "后端", "training.task_name", "owner YAML"], _owner_rows(owners))}

{code_walkthrough}

## Agent

{task.agent}

训练入口通过 owner YAML 决定注册 env、后端身份和算法参数；CLI 应使用 `--sim` 选择 backend owner，而不是单独 override `training.sim_backend`。

```yaml
# {owner["path"]}
{_yaml_block({"training": data["training"], "algo": {"obs_groups": obs_groups, "num_envs": algo_cfg.get("num_envs"), "max_iterations": algo_cfg.get("max_iterations")}})}```

## Env

{task.env}

Env 必须遵守 `NpEnv` contract：`reset()` 返回 `(obs_dict, info_dict)`，`obs` 是 dict，`obs_groups_spec` 决定 learner 使用哪些观测组。

{_source_snippet(task.source_hint, ("registry.envcfg",) + task.registry_names)}

## Obs

{task.obs}

### Obs 字段级拆解

下面的表格把源码里的观测拼接逻辑拆成语义字段。维度如果依赖 history、terrain scan 或 motion body 数量，文档会写成“规模”而不是硬编码，避免和配置漂移。

{_obs_breakdown_table(task)}

owner YAML 中的 `algo.obs_groups` 说明算法从 env obs dict 中读取哪些组。没有写入 owner 的字段保持 env cfg 默认值，文档不臆造未配置项。

```yaml
# {owner["path"]}
algo:
  obs_groups:
{_indent_yaml(obs_groups, 4)}
```

代码侧观测通常在 `_get_obs`、`_build_obs`、`obs_groups_spec` 或 base env 中组装：

{_source_snippet(task.source_hint, ("obs_groups_spec", "_get_obs", "build_obs", "get_obs"), context=6)}

源码锚点拆解：

{_table(["源码符号", "中文说明"], obs_symbol_rows)}

## Action

{task.action}

### Action 逐维拆解

{_action_detail_text(task)}

当前代表 scene 编译出的 action 维度为 `{action_dim}`。下表来自 MuJoCo 编译模型的 actuator 顺序，也就是策略输出维度和执行器维度的对应关系。

{_table(["Action 维度", "执行器", "驱动关节", "中文含义", "控制/限制说明"], action_rows)}

控制参数来自 owner YAML 的 `env.control_config`，缺省值由对应 env cfg/dataclass 提供。

{_table(["配置字段", "当前值", "中文说明"], control_rows)}

```yaml
# {owner["path"]}
env:
  control_config:
{_indent_yaml(control_cfg, 4)}
```

动作到执行器的实际映射由 backend 的 actuator control range 和 env 的 `apply_action`/控制函数完成：

{_source_snippet(task.source_hint, ("apply_action", "action_scale", "compute_go2w_motor_ctrl", "_init_action_space"), context=6)}

源码锚点拆解：

{_table(["源码符号", "中文说明"], action_symbol_rows)}

## Reward

{task.reward}

### Reward 逐项拆解

reward scale 以 owner YAML 的 `reward.scales` 为真源；源码中的 `RewardConfig` 描述可用字段，训练脚本只做 Hydra 组装和注入。正数通常表示奖励，负数通常表示惩罚，0 表示该项在当前 owner 中关闭或仅保留接口。

{_table(["Reward key", "Scale", "方向", "中文含义"], reward_rows)}

```yaml
# {owner["path"]}
reward:
{_indent_yaml(reward_cfg, 2)}
```

源码中的 reward 入口：

{_source_snippet(task.source_hint, ("RewardConfig", "_compute_reward", "run_reward_dispatch", "_reward"), context=6)}

源码锚点拆解：

{_table(["源码符号", "中文说明"], reward_symbol_rows)}

## 初始状态

{task.init}

机器人默认姿态来自 scene/task fragment 中的 keyframe；任务 reset 在此基础上加入命令、motion reference、物体状态或随机化。

{_xml_snippet(task.scene_hint, ("<keyframe", "keyframe"))}

{_source_snippet(task.source_hint, ("reset", "build_reset_plan", "_reset", "get_keyframe_qpos"), context=6)}

## 终止条件

{task.termination}

终止条件通常由 env 源码中的 `_compute_termination(s)`、reward config 阈值和 `max_episode_seconds` 共同决定。

```yaml
# {owner["path"]}
env:
{_indent_yaml(_termination_related_env(env_cfg), 2)}
reward:
{_indent_yaml(_termination_related_reward(reward_cfg), 2)}
```

{_source_snippet(task.source_hint, ("_compute_termination", "_compute_terminations", "terminated", "termination"), context=8)}

## 域随机化

{task.domain_rand}

域随机化只在 reset/interval 等冷路径触发，不在 step 热路径解析资产或探测 backend 私有能力。

### 域随机化字段拆解

{_table(["配置字段", "当前值", "中文说明"], domain_rand_rows)}

```yaml
# {owner["path"]}
env:
  domain_rand:
{_indent_yaml(domain_rand, 4)}
```

{_source_snippet(task.source_hint, ("DomainRand", "DomainRandomization", "build_reset_plan", "domain_rand"), context=6)}
"""
        output.write_text(content, encoding="utf-8")


def _indent_yaml(value: Any, spaces: int) -> str:
    dumped = _yaml_block(value).rstrip()
    if dumped == "null":
        dumped = "{}"
    prefix = " " * spaces
    return "\n".join(prefix + line for line in dumped.splitlines()) + "\n"


def _termination_related_env(env_cfg: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "max_episode_seconds",
        "energy_termination_threshold",
        "anchor_pos_z_threshold",
        "anchor_ori_threshold",
        "ee_body_pos_z_threshold",
        "truncate_on_clip_end",
    ]
    return {key: env_cfg[key] for key in keys if key in env_cfg}


def _termination_related_reward(reward_cfg: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "min_base_height",
        "max_tilt_deg",
        "reset_z_threshold",
        "ball_lost_distance",
        "ball_lost_distance_hard",
    ]
    return {key: reward_cfg[key] for key in keys if key in reward_cfg}


def write_index(
    owners_by_slug: dict[str, list[dict[str, Any]]],
    registry_info: dict[str, dict[str, Any]],
    robot_inventory: dict[str, dict[str, Any]],
) -> None:
    robot_rows = [
        [
            robot.title,
            robot.role,
            f"[文档](robots/{slug}/index.md)",
            f"`{robot.primary_scene}`",
        ]
        for slug, robot in ROBOTS.items()
    ]
    task_rows = []
    for slug, task in TASKS.items():
        backends = sorted({owner["sim_backend"] for owner in owners_by_slug[slug]})
        algos = sorted({owner["algo_group"] for owner in owners_by_slug[slug]})
        task_rows.append(
            [
                task.title,
                CATEGORY_TITLES[task.category],
                task.robot,
                ", ".join(algos),
                ", ".join(backends),
                f"[文档](tasks/{task.category}/{slug}.md)",
            ]
        )

    content = f"""---
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

{_table(["机器人", "定位", "文档", "主场景"], robot_rows)}

## 任务索引

{_table(["任务", "类别", "机器人", "算法组", "后端", "文档"], task_rows)}

## 生成命令

```bash
uv run docs/blog/zh_CN/rl_tasks/_tools/generate_rl_docs.py
uv run docs/blog/zh_CN/rl_tasks/_tools/render_static_images.py
```

## 事实源摘要

{_table(["事实源", "数量"], [
    ["Registry env", len(registry_info)],
    ["文档任务", len(TASKS)],
    ["机器人", len(robot_inventory)],
    ["Owner YAML", sum(len(items) for items in owners_by_slug.values())],
])}

## 阅读建议

先从机器人页理解资产和关节，再进入任务页阅读 Agent/Env/Obs/Action/Reward/初始状态/终止条件/域随机化。任务页中的 YAML 片段代表当前 owner 配置，源码片段展示对应 env 的 owner layer 实现。
"""
    (DOC_ROOT / "README.md").write_text(content, encoding="utf-8")

    for category, title in CATEGORY_TITLES.items():
        category_tasks = [task for task in TASKS.values() if task.category == category]
        rows = [
            [
                task.title,
                task.robot,
                ", ".join(task.registry_names),
                f"[文档]({task.slug}.md)",
            ]
            for task in category_tasks
        ]
        path = DOC_ROOT / "tasks" / category / "README.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            f"# {title}任务\n\n{_table(['任务', '机器人', 'Registry', '文档'], rows)}\n",
            encoding="utf-8",
        )


def write_asset_manifest_md(asset_manifest: dict[str, Any]) -> None:
    rows = []
    for slug, payload in asset_manifest["robots"].items():
        rows.append([slug, payload["source"], payload["destination"], len(payload["files"])])
    content = f"""# 机器人资产复制清单

本清单由 `_tools/generate_rl_docs.py` 生成。复制范围限定在 `src/unilab/assets/robots/<robot>/` 下的机器人 XML、scene/fragment XML、mesh、texture 和任务场景素材；不会复制 motion、checkpoint、cache。

{_table(["机器人", "来源", "目标", "文件数"], rows)}
"""
    (DOC_ROOT / "_generated" / "asset_manifest.md").write_text(content, encoding="utf-8")


def write_inventory_files(
    owners_by_slug: dict[str, list[dict[str, Any]]],
    registry_info: dict[str, dict[str, Any]],
    robot_inventory: dict[str, dict[str, Any]],
    asset_manifest: dict[str, Any],
) -> None:
    generated = DOC_ROOT / "_generated"
    generated.mkdir(parents=True, exist_ok=True)

    owners_public = {
        slug: [
            {
                "path": owner["path"],
                "task_name": owner["task_name"],
                "sim_backend": owner["sim_backend"],
                "algo_group": owner["algo_group"],
            }
            for owner in items
        ]
        for slug, items in sorted(owners_by_slug.items())
    }
    inventory = {
        "tasks": {slug: task.__dict__ for slug, task in TASKS.items()},
        "owners": owners_public,
        "registry": registry_info,
        "robots": robot_inventory,
    }
    (generated / "inventory.json").write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (generated / "asset_manifest.json").write_text(
        json.dumps(asset_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_asset_manifest_md(asset_manifest)


def validate_metadata(
    owners_by_slug: dict[str, list[dict[str, Any]]],
    registry_info: dict[str, dict[str, Any]],
) -> None:
    missing_slugs = sorted(set(TASKS) - set(owners_by_slug))
    if missing_slugs:
        raise ValueError(f"Missing owner YAML for task slugs: {missing_slugs}")
    missing_registry = sorted(
        {
            env_name
            for task in TASKS.values()
            for env_name in task.registry_names
            if env_name not in registry_info
        }
    )
    if missing_registry:
        raise ValueError(f"Missing registry entries: {missing_registry}")


def validate_links() -> dict[str, Any]:
    markdown_files = sorted(DOC_ROOT.rglob("*.md"))
    missing: list[dict[str, str]] = []
    link_re = re.compile(r"\]\(([^)#][^)]+)\)")
    for md_path in markdown_files:
        text = md_path.read_text(encoding="utf-8")
        for match in link_re.finditer(text):
            target = match.group(1)
            if "://" in target:
                continue
            if target.startswith("mailto:"):
                continue
            target_path = (md_path.parent / target).resolve()
            if not target_path.exists():
                missing.append({"file": rel(md_path), "target": target})
    report = {"markdown_files": len(markdown_files), "missing_links": missing}
    (DOC_ROOT / "_generated" / "link_check.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate UniLab RL task/robot blog documentation.",
    )
    parser.add_argument(
        "--overwrite-task-docs",
        action="store_true",
        help=(
            "Overwrite existing task Markdown files. By default existing task pages are "
            "kept to protect hand-written code annotations."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    DOC_ROOT.mkdir(parents=True, exist_ok=True)
    owners_by_slug = load_owner_configs()
    registry_info = load_registry()
    validate_metadata(owners_by_slug, registry_info)
    robot_inventory = build_robot_inventory()
    asset_manifest = copy_robot_assets(robot_inventory)
    write_inventory_files(owners_by_slug, registry_info, robot_inventory, asset_manifest)
    write_robot_docs(robot_inventory, asset_manifest)
    write_task_docs(
        owners_by_slug,
        registry_info,
        robot_inventory,
        overwrite_existing=args.overwrite_task_docs,
    )
    write_index(owners_by_slug, registry_info, robot_inventory)
    link_report = validate_links()
    missing_non_images = [
        item
        for item in link_report["missing_links"]
        if not item["target"].lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".webp"))
    ]
    if missing_non_images:
        raise RuntimeError(f"Missing generated doc links: {missing_non_images}")
    print(
        "generated docs: tasks={tasks} robots={robots} markdown={markdown}".format(
            tasks=len(TASKS),
            robots=len(ROBOTS),
            markdown=link_report["markdown_files"],
        )
    )


if __name__ == "__main__":
    main()

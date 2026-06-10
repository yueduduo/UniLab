# K1VelCmdCompat 键盘摇杆（UniLab 内测）
cd ~/projects/UniLab && uv run scripts/play_interactive.py \
  --config-path /home/m/projects/UniLab/conf/offpolicy \
  --config-name config \
  algo=flashsac task=flashsac/k1_vel_cmd_compat/mujoco \
  algo.load_run=2026-06-10_12-12-35_mujoco \
  +interactive.action_mode=policy +interactive.keyboard=true

# 导出 MOS-SIM 部署用 TorchScript（FlashSAC ckpt → 78→22 .pt）
cd ~/projects/UniLab && uv run scripts/export_k1_dribble_compat_deploy.py \
  --checkpoint logs/flash_sac/K1VelCmdCompat/2026-06-10_19-33-40_mujoco/model_10000.pt \
  --output MOS-SIM-perf-pipelined-capture/simulation/motrixsim/assets/policies/k1_vel_cmd_model_10000.pt

# MOS-SIM 比赛场景部署（策略名含 vel_cmd 时自动放开 yaw、cmd 范围 [-1,1]）
conda activate motrixsim0508
cd ~/projects/UniLab/MOS-SIM-perf-pipelined-capture/simulation/motrixsim
python sim2sim_runner.py --robot-type k1 --team-size 1 --no-k1-legged-gym \
  --policy assets/policies/k1_vel_cmd_model_10000.pt --real-time
# vel_cmd 策略默认把球放在 +X 侧点球点 (3.0, 0)；左点球点加 --initial-ball-penalty-side left

# headless 快速验
python tools/test_k1_unilab_headless.py --policy assets/policies/k1_vel_cmd_model_20000.pt \
  --vx 0 --vy 0 --yaw 0 --seconds 16
python tools/test_k1_unilab_headless.py --policy assets/policies/k1_vel_cmd_model_20000.pt \
  --vx 0.5 --vy 0 --yaw 0 --seconds 16

# 看mujoco带球（旧）

cd ~/projects/UniLab && uv run scripts/play_interactive.py \
  --config-path /home/m/projects/UniLab/conf/offpolicy \
  --config-name config \
  algo=flashsac task=flashsac/k1_vel_cmd_compat/mujoco \
  algo.load_run=2026-06-10_12-12-35_mujoco \
  +interactive.action_mode=policy +interactive.keyboard=true \
  +interactive.soccer_dribble_debug=true

# 录制视频
  uv run eval --algo flashsac --task k1_soccer_dribble --sim mujoco \
  --render-mode record \
  algo.load_run=/home/m/projects/UniLab/logs/flash_sac/K1SoccerDribble/2026-06-06_22-25-33_mujoco/model_11000.pt \
  training.export_onnx=false training.play_env_num=1 training.play_steps=800


  cd ~/projects/UniLab && uv run eval --algo flashsac --task k1_soccer_dribble --sim mujoco \
  --render-mode record \
  algo.load_run=/home/m/projects/UniLab/logs/flash_sac/K1SoccerDribble/2026-06-07_15-21-58_mujoco/model_10000.pt \
  training.play_env_num=16 training.play_steps=800

# 点球训练（warm-start: K1SoccerDribble model_13000）
cd ~/projects/UniLab && uv run train --algo flashsac --task k1_soccer_penalty_kick --sim mujoco
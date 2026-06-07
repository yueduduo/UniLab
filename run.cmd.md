# 看mujoco带球

cd ~/projects/UniLab && uv run scripts/play_interactive.py \
  --config-path /home/m/projects/UniLab/conf/offpolicy \
  --config-name config \
  algo=flashsac task=flashsac/k1_soccer_dribble/mujoco \
  algo.load_run=2026-06-07_10-24-48_mujoco \
  +interactive.action_mode=policy +interactive.keyboard=true \
  +interactive.soccer_dribble_debug=true

# 录制视频
  uv run eval --algo flashsac --task k1_soccer_dribble --sim mujoco \
  --render-mode record \
  algo.load_run=/home/m/projects/UniLab/logs/flash_sac/K1SoccerDribble/2026-06-06_22-25-33_mujoco/model_11000.pt \
  training.export_onnx=false training.play_env_num=1 training.play_steps=800
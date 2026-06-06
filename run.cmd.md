# 看mujoco带球

cd ~/projects/UniLab && uv run scripts/play_interactive.py \
  --config-path /home/m/projects/UniLab/conf/offpolicy \
  --config-name config \
  algo=flashsac task=flashsac/k1_soccer_dribble/mujoco \
  algo.load_run=2026-06-06_18-51-00_mujoco \
  +interactive.action_mode=policy +interactive.keyboard=true \
  +interactive.soccer_dribble_debug=true
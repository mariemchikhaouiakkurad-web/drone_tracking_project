#!/usr/bin/env bash
# start_vision.sh - JETSON CONTAINER: camera receiver + YOLO/DyHiT + fusion (+ latency trace + image stream).
# tmux session "vision": windows cam | track | fusion | trace | stream    (watch: tmux attach -t vision)
# Usage: start_vision.sh [detector_every]
set -u
. "$(dirname "$0")/common.sh"
[ "$INSIDE" = 1 ] || { echo "[FAIL] run in the container (or: jetson/start_all.sh from the host)"; exit 1; }
need tmux tmux
N=${1:-$DETECTOR_EVERY}
tmux kill-session -t vision 2>/dev/null
pkill -f net_camera_node.py; pkill -f thermal_tracking.launch; pkill -f "dalbd_fusion/state_machine"; pkill -f stream_server_node.py; sleep 1
V="$(ros_env $VISION_DOMAIN)"
win vision cam "$V; cd $WS/tools/net_camera && python3 net_camera_node.py --ros-args -p host:=$LAPTOP_IP -p port:=$CAM_PORT -p image_topic:=image -p camera_info_topic:=camera_info"
win vision track "$V; ros2 launch drone_target_tracker thermal_tracking.launch.py detector_every:=$N video_device:=/dev/null_cam"
sleep 2
win vision fusion "$V; ros2 run dalbd_fusion state_machine --ros-args --params-file \$(ros2 pkg prefix dalbd_fusion)/share/dalbd_fusion/config/fusion.yaml"
if [ "$STREAM_TOOLS" = 1 ]; then
  sleep 8
  win vision trace "$V; ros2 run dalbd_fusion latency_trace"
  win vision stream "$V; cd $WS/tools/net_camera && python3 stream_server_node.py --ros-args -p topic:=/target/debug_image -p port:=9400 -p quality:=80"
fi
echo "[ OK ] vision started (detector_every=$N). tmux session 'vision'."

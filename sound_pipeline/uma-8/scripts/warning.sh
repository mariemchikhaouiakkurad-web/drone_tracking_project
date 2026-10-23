#!/bin/bash
# warning.sh - drone warning (end of the pipeline): /target/direction -> /drone_warning + console
#   uma-8/scripts/warning.sh [--ros-args -p hold_s:=1.5 -p min_prob:=0.5]
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
source /opt/ros/humble/setup.bash
SW=$HOME/ros2_ws/install; [ -d /workspaces/isaac_ros-dev ] && SW=/workspaces/isaac_ros-dev/sound_ws/install_container; [ -f "$SW/setup.bash" ] && source "$SW/setup.bash"
export PYTHONUNBUFFERED=1
LOG="/home/akkurad/dataset/odas_logs"; mkdir -p "$LOG"
python3 "$ROOT/warning/drone_warning_node.py" "$@" 2>&1 | tee "$LOG/warning_$(date +%Y%m%d_%H%M%S).log"

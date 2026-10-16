#!/bin/bash
# warning.sh - drone warning (end of the pipeline): /target/direction -> /drone_warning + console
#   uma-8/scripts/warning.sh [--ros-args -p hold_s:=1.5 -p min_prob:=0.5]
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
source /opt/ros/humble/setup.bash
[ -f "$HOME/ros2_ws/install/setup.bash" ] && source "$HOME/ros2_ws/install/setup.bash"
export PYTHONUNBUFFERED=1
LOG="$HOME/dataset/odas_logs"; mkdir -p "$LOG"
python3 "$ROOT/warning/drone_warning_node.py" "$@" 2>&1 | tee "$LOG/warning_$(date +%Y%m%d_%H%M%S).log"

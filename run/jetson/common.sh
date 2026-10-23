# common.sh - sourced by every Jetson script (host or container)
if [ -d /workspaces/isaac_ros-dev ]; then WS=/workspaces/isaac_ros-dev; INSIDE=1
else WS=$HOME/workspaces/isaac_ros-dev; INSIDE=0; fi
if [ -d "$WS/src/DALBD" ]; then DALBD=$WS/src/DALBD; else DALBD=$HOME/DALBD; fi
RUN=$DALBD/run
. "$RUN/run.conf"
LOGS=$WS/fusion_logs
mkdir -p "$LOGS"

ros_env() {  # $1 = domain
  local s="source /opt/ros/humble/setup.bash"
  if [ "$INSIDE" = 1 ]; then s="$s; source $WS/install/setup.bash"
  else s="$s; [ -f $HOME/ros2_ws/install/setup.bash ] && source $HOME/ros2_ws/install/setup.bash"; fi
  echo "$s; export ROS_DOMAIN_ID=$1 ROS_LOCALHOST_ONLY=0"
}

win() {  # win <session> <window> <command>   (window stays open after the command ends, to read errors)
  local f="$LOGS/.run_$1_$2.sh"
  printf '%s\n' "$3" 'echo; echo "[exited] press Enter for a shell"; read; exec bash' > "$f"
  if tmux has-session -t "$1" 2>/dev/null; then tmux new-window -t "$1" -n "$2" "bash -l $f"
  else tmux new-session -d -s "$1" -n "$2" "bash -l $f"; fi
}

need() { command -v "$1" >/dev/null || { echo "[FAIL] '$1' missing: sudo apt install -y $2"; exit 1; }; }

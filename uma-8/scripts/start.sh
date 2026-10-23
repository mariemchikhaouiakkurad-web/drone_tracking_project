#!/bin/bash
# start.sh - the whole sound pipeline in one command (UMA-8 streamed from the laptop into the Jetson Loopback).
#
#   Loopback -> arecord -> /tmp/odas_raw.fifo -> ODAS (uma-8/config/uma8.cfg) -> /raw/sst, /raw/sss
#   -> nodes/direction_gate_node.py     known point (track id + direction cone) dropped
#        /gate/sss (known channel silenced) -> nodes/sss_target_classifier_node.py (MobileNet) -> /classifier/tracks
#        /gate/sst (known track removed)    -> nodes/target_direction_node.py -> /target/direction, /target/markers (RViz)
#   -> warning/drone_warning_node.py (uma-8/scripts/warning.sh) -> /drone_warning
#
#   uma-8/scripts/start.sh          10 s calibration of the known point (known source ALONE) + start + watchdog
#   uma-8/scripts/start.sh load     reuse the saved known point (uma-8/config/known_point.json)
#   uma-8/scripts/start.sh stop     stop everything
#   CAL_S=15 uma-8/scripts/start.sh longer calibration       DEV=plughw:...  other capture device
# Logs: ~/dataset/odas_logs/{odas_server,arecord,direction_gate,classifier,target_direction}.log
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CFG="$ROOT/uma-8/config/uma8.cfg"
POINT="$ROOT/uma-8/config/known_point.json"
YAML="$ROOT/uma-8/config/drone_classifier.yaml"
FIFO=/tmp/odas_raw.fifo
DEV="${DEV:-plughw:CARD=Loopback,DEV=1}"
CAL_S="${CAL_S:-10}"
WATCH_S="${WATCH_S:-20}"
LOG="/home/akkurad/dataset/odas_logs"; mkdir -p "$LOG"
if [ -f /opt/ros/humble/setup.bash ]; then source /opt/ros/humble/setup.bash; else echo "ERROR: ROS 2 Humble not found"; exit 1; fi
SW=$HOME/ros2_ws/install; [ -d /workspaces/isaac_ros-dev ] && SW=/workspaces/isaac_ros-dev/sound_ws/install_container; [ -f "$SW/setup.bash" ] && source "$SW/setup.bash"
set -u
export PYTHONUNBUFFERED=1 RCUTILS_LOGGING_BUFFERED_STREAM=0 RCUTILS_COLORIZED_OUTPUT=0
cd "$ROOT"
say()  { echo "[$(date +%H:%M:%S)] $*"; }
rate() { local r i; for i in 1 2; do r=$(timeout 8 ros2 topic hz "$1" 2>/dev/null | awk '/average rate/{print $3; exit}'); [ -n "$r" ] && { echo "$r"; return; }; done; }
stream_ok() { for s in /proc/asound/Loopback/pcm*p/sub*/status; do grep -q "state: RUNNING" "$s" 2>/dev/null && return 0; done; return 1; }

# ------------------------------------------------------------------ ODAS
stop_odas() {
  pkill -f "arecord.*$DEV" 2>/dev/null
  pkill -9 -f odas_server_node 2>/dev/null
  pkill -9 -x odas_core_node 2>/dev/null
  rm -f "$FIFO"
  sleep 1
}

start_odas() {
  stop_odas
  mkfifo "$FIFO"
  nohup ros2 run odas_ros odas_server_node.py --ros-args -r __ns:=/raw \
        -p configuration_path:="$CFG" -p frame_id:=odas > "$LOG/odas_server.log" 2>&1 &
  local i n p
  for i in $(seq 1 40); do
    n=0; for p in 9000 9001 10200; do ss -ltn | grep -q ":$p " && n=$((n + 1)); done
    [ "$n" -ge 3 ] && break; sleep 0.5
  done
  [ "$n" -ge 3 ] || { say "ERROR: ODAS server not listening on 9000/9001/10200"; tail -8 "$LOG/odas_server.log"; return 1; }
  # odas_server_node starts ONE odas_core_node itself; it opens the FIFO for reading
  nohup arecord -D "$DEV" -q -c 8 -r 48000 -f S32_LE -t raw > "$FIFO" 2> "$LOG/arecord.log" &
  for i in $(seq 1 20); do [ "$(pgrep -cx odas_core_node)" -ge 1 ] && break; sleep 0.5; done
  [ "$(pgrep -cx odas_core_node)" -ge 1 ] || { say "ERROR: odas_core_node not running"; tail -8 "$LOG/odas_server.log"; return 1; }
  sleep 2
  pgrep -f "arecord.*$DEV" >/dev/null || { say "ERROR: arecord died:"; cat "$LOG/arecord.log"; return 1; }
  if timeout 8 ros2 topic echo /raw/sst --once >/dev/null 2>&1; then say "ODAS up: /raw/sst publishing"
  else say "ERROR: no /raw/sst"; tail -8 "$LOG/odas_server.log"; return 1; fi
}

# ------------------------------------------------------------------ nodes
start_gate() {
  pkill -f direction_gate_node 2>/dev/null; sleep 1
  local load=false; [ "${1:-}" = load ] && load=true
  nohup python3 "$ROOT/nodes/direction_gate_node.py" --ros-args -p load:=$load -p cal_s:="${CAL_S}.0" \
        > "$LOG/direction_gate.log" 2>&1 &
  say "direction gate started (log: $LOG/direction_gate.log)"
}

wait_point() {
  local t0=$SECONDS
  if [ "${1:-}" != load ]; then
    say "CALIBRATION ${CAL_S} s: play ONLY the known source now"
    until grep -q "KNOWN POINT" "$LOG/direction_gate.log" 2>/dev/null; do
      sleep 1; [ $((SECONDS - t0)) -gt $((CAL_S + 40)) ] && { say "ERROR: no known point (is the known source playing loud enough?)"; return 1; }
    done
  else
    [ -f "$POINT" ] || { say "ERROR: no saved point ($POINT): run without 'load'"; return 1; }
    sleep 2
  fi
  grep -E "KNOWN POINT|WARN|loaded" "$LOG/direction_gate.log" | tail -2
}

cone_of_point() { python3 -c "import json;print(float(json.load(open('$POINT'))['cone_deg']))" 2>/dev/null || echo 20.0; }

start_direction() {
  pkill -f target_direction_node 2>/dev/null; sleep 1
  nohup python3 "$ROOT/nodes/target_direction_node.py" --ros-args \
        -p sst_topic:=/gate/sst -p known_topic:=/gate/known \
        -p known_cone_deg:="$(cone_of_point)" -p cone_3d:=true -p known_lock_s:=0.0 \
        -p opposite_cone_deg:=0.0 -p learn_s:=0.0 -p labels:=false \
        -p known_dirs_file:=/tmp/known_dirs_point.json > "$LOG/target_direction.log" 2>&1 &
  say "target_direction_node on /gate/sst -> /target/direction, /target/markers (log: $LOG/target_direction.log)"
}

start_classifier() {
  pkill -f sss_target_classifier 2>/dev/null; sleep 1
  nohup python3 "$ROOT/nodes/sss_target_classifier_node.py" --ros-args -p params:="$YAML" \
        -p sss_topic:=/gate/sss -p sst_topic:=/gate/sst > "$LOG/classifier.log" 2>&1 &
  say "MobileNet on /gate/sss (log: $LOG/classifier.log)"
}

stop_all() {
  pkill -f direction_gate_node 2>/dev/null
  pkill -f sss_target_classifier 2>/dev/null
  pkill -f target_direction_node 2>/dev/null
  stop_odas
}

check() {
  local raw rsss gsss
  raw=$(rate /raw/sst); rsss=$(rate /raw/sss); gsss=$(rate /gate/sss)
  say "stream $(stream_ok && echo OK || echo DOWN) | /raw/sst ${raw:-0} Hz | /raw/sss ${rsss:-0} Hz | /gate/sss ${gsss:-0} Hz | classifier $(pgrep -f sss_target_classifier >/dev/null && echo up || echo DOWN)"
  [ -n "$raw" ] && [ -n "$rsss" ] && [ -n "$gsss" ]
}

# ------------------------------------------------------------------ main
case "${1:-}" in
  stop) stop_all; say "stopped"; exit 0 ;;
  ""|load) ;;
  *) echo "usage: $0 [load|stop]"; exit 1 ;;
esac
stream_ok || { say "ERROR: no audio in Loopback: start the JETSON receiver, then the LAPTOP sender"; exit 1; }
say "audio stream from the laptop: OK"
stop_all
start_odas || { stop_all; exit 1; }
start_gate "${1:-}"
wait_point "${1:-}" || { stop_all; exit 1; }
start_direction
start_classifier
sleep 5
check && say "pipeline OK: MobileNet gets /gate/sss (known point dropped)" || say "WARN: not fully up yet, the watchdog will retry"
subs=$(timeout 5 ros2 topic info /gate/sss 2>/dev/null | awk '/Subscription count/{print $3}')
[ "${subs:-0}" -ge 1 ] && say "MobileNet reads /gate/sss" || say "WARN: nobody reads /gate/sss (see $LOG/classifier.log)"

say "watchdog every ${WATCH_S}s (Ctrl+C stops the watchdog only; stop all: $0 stop)"
fails=0
while true; do
  sleep "$WATCH_S"
  stream_ok || { say "WARN: laptop stream DOWN"; continue; }
  pgrep -f direction_gate_node >/dev/null   || { say "gate died -> restart (saved point)"; start_gate load; }
  pgrep -f sss_target_classifier >/dev/null || { say "classifier died -> restart"; start_classifier; }
  pgrep -f target_direction_node >/dev/null || { say "target_direction died -> restart"; start_direction; }
  if check; then fails=0; continue; fi
  fails=$((fails + 1))
  if [ "$fails" -ge 3 ] || [ "$(pgrep -cx odas_core_node)" -lt 1 ] || ! pgrep -f "arecord.*$DEV" >/dev/null; then
    say "chain down -> restart ODAS + gate (saved point) + nodes"
    start_odas && start_gate load && start_direction && start_classifier
    fails=0
  fi
done

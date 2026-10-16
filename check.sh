#!/bin/bash
# check.sh - live check of the sound pipeline (run while uma-8/scripts/start.sh is up).
source /opt/ros/humble/setup.bash
[ -f "$HOME/ros2_ws/install/setup.bash" ] && source "$HOME/ros2_ws/install/setup.bash"
fail=0
ok()  { echo "  [OK]   $*"; }
bad() { echo "  [FAIL] $*"; fail=1; }
echo "== laptop link + audio"
ping -c1 -W1 192.168.10.2 >/dev/null 2>&1 && ok "laptop 192.168.10.2 reachable" || bad "laptop 192.168.10.2 not reachable"
run=0; for s in /proc/asound/Loopback/pcm*p/sub*/status; do grep -q "state: RUNNING" "$s" 2>/dev/null && run=1; done
[ $run = 1 ] && ok "Loopback stream RUNNING" || bad "no stream in Loopback (start the receiver + laptop sender)"
echo "== processes"
n=$(pgrep -cx odas_core_node); [ "$n" = 1 ] && ok "1 odas_core_node" || bad "$n odas_core_node (expected 1)"
pgrep -f "arecord -D" >/dev/null && ok "arecord -> /tmp/odas_raw.fifo" || bad "arecord not running"
for p in direction_gate_node sss_target_classifier_node target_direction_node; do
  c=$(pgrep -fc "$p.py"); [ "$c" = 1 ] && ok "$p" || bad "$p running $c times (expected 1)"
done
pgrep -f drone_warning_node.py >/dev/null && ok "drone_warning_node" || echo "  [--]   drone_warning_node not running (uma-8/scripts/warning.sh)"
echo "== topics (rate, publishers)"
for t in /raw/sst /raw/sss /gate/sst /gate/sss /gate/known /target/direction /target/markers; do
  r=$(timeout 6 ros2 topic hz "$t" 2>/dev/null | awk '/average rate/{print $3; exit}')
  np=$(ros2 topic info "$t" 2>/dev/null | awk '/Publisher count/{print $3}')
  if [ -n "$r" ] && [ "${np:-0}" = 1 ]; then ok "$(printf '%-20s %6s Hz' "$t" "$r")"
  else bad "$(printf '%-20s rate %s, publishers %s' "$t" "${r:-0}" "${np:-0}")"; fi
done
np=$(ros2 topic info /classifier/tracks 2>/dev/null | awk '/Publisher count/{print $3}')
[ "${np:-0}" = 1 ] && ok "/classifier/tracks has 1 publisher (MobileNet)" || bad "/classifier/tracks publishers: ${np:-0}"
echo "== known point"
f="$(cd "$(dirname "$0")" && pwd)/uma-8/config/known_point.json"
[ -f "$f" ] && ok "$(cat "$f")" || bad "no $f (run start.sh without 'load')"
echo; [ $fail = 0 ] && echo "check OK" || echo "check FAILED"
exit $fail

#!/usr/bin/env bash
# status.sh - JETSON HOST or container: is every part alive? (topic rates in both domains + power)
set -u
. "$(dirname "$0")/common.sh"
rate() {  # rate <domain> <topic>
  local r
  r=$(bash -c "$(ros_env $1); timeout 6 ros2 topic hz $2 2>/dev/null | grep -m1 'average rate' | awk '{print \$3}'")
  printf "  %-24s domain %-3s %s\n" "$2" "$1" "${r:+$r Hz}${r:-NO DATA}"
}
echo "== sound";  rate $SOUND_DOMAIN /drone_warning; rate $SOUND_DOMAIN /raw/sst
echo "== vision"; rate $VISION_DOMAIN /image; rate $VISION_DOMAIN /detections_output; rate $VISION_DOMAIN /target/bearing
echo "== fusion"; rate $VISION_DOMAIN /fused/state
echo "== board";  tail -1 "$LOGS/tegrastats.log" 2>/dev/null | grep -oE "VDD_IN [0-9]+mW|GR3D_FREQ [0-9]+%|tj@[0-9.]+C" | tr '\n' ' '; echo

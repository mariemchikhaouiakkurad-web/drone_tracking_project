#!/bin/bash
# sss_rec.sh - record the sound MobileNet gets, before and after the known-point drop -> ~/dataset/sss_rec/
#   uma-8/scripts/sss_rec.sh                record only (Ctrl+C to stop and save)
#   uma-8/scripts/sss_rec.sh listen both    + stream to the laptop: LEFT = before, RIGHT = after
#        LAPTOP first: nc -l 6000 | aplay -t raw -f S16_LE -r 16000 -c 2
#   uma-8/scripts/sss_rec.sh listen after   only the after-drop sound  (LAPTOP: ... -c 1)
#   uma-8/scripts/sss_rec.sh listen before  only the before sound      (LAPTOP: ... -c 1)
# Then: python3 tools/sss_analyze.py   (report + spectrograms of the last take)
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
source /opt/ros/humble/setup.bash
SW=$HOME/ros2_ws/install; [ -d /workspaces/isaac_ros-dev ] && SW=/workspaces/isaac_ros-dev/sound_ws/install_container; [ -f "$SW/setup.bash" ] && source "$SW/setup.bash"
export PYTHONUNBUFFERED=1
ARGS=(-p topic:=/raw/sss -p sst_topic:=/raw/sst -p gate:=true)
if [ "${1:-}" = listen ]; then
  python3 "$ROOT/tools/sss_recorder.py" --ros-args -p listen:="${2:-both}" "${ARGS[@]}" | nc 192.168.10.2 6000
else
  python3 "$ROOT/tools/sss_recorder.py" --ros-args -p listen:=none "${ARGS[@]}"
fi

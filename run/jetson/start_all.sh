#!/usr/bin/env bash
# start_all.sh - JETSON HOST: everything in one command.  Usage: start_all.sh [detector_every]
set -u
DIR="$(cd "$(dirname "$0")" && pwd)"
. "$DIR/common.sh"
[ "$INSIDE" = 0 ] || { echo "[FAIL] run on the Jetson HOST"; exit 1; }
"$DIR/host_prep.sh"
docker ps --format '{{.Names}}' | grep -qx "$CONTAINER" || exit 1
CIN="/workspaces/isaac_ros-dev/src/DALBD/run/jetson"
if [ "$SOUND_WHERE" = container ]; then
  docker exec -u root "$CONTAINER" usermod -aG audio "$CONTAINER_USER"   # sound cards (user is recreated at each container start)
  docker exec -u "$CONTAINER_USER" "$CONTAINER" bash -lc "$CIN/start_sound.sh"
else
  "$DIR/start_sound.sh"
fi
docker exec -u "$CONTAINER_USER" "$CONTAINER" bash -lc "$CIN/start_vision.sh ${1:-}"
echo
echo "Watch:  sound  -> $( [ "$SOUND_WHERE" = container ] && echo "docker exec -it -u $CONTAINER_USER $CONTAINER tmux attach -t sound" || echo "tmux attach -t sound")"
echo "        vision -> docker exec -it -u $CONTAINER_USER $CONTAINER tmux attach -t vision"
echo "Check:  $DIR/status.sh        Stop: $DIR/stop_all.sh"

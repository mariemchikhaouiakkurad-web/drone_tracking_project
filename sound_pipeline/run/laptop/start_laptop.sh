#!/usr/bin/env bash
# start_laptop.sh - LAPTOP: thermal camera sender + UMA-8 audio sender (both reconnect by themselves)
# tmux session "laptop": windows cam | audio | view      (watch: tmux attach -t laptop)
set -u
DIR="$(cd "$(dirname "$0")/.." && pwd)"
. "$DIR/run.conf"
command -v tmux >/dev/null || { echo "sudo apt install -y tmux"; exit 1; }
NETCAM=${NETCAM:-$HOME/net_camera}
tmux kill-session -t laptop 2>/dev/null; pkill -f cam_sender.py; pkill -f "arecord -D plughw:CARD=$UMA8_CARD"; sleep 1
DEV=$CAM_DEVICE
if [ "$DEV" = auto ]; then
  for d in /dev/video*; do v4l2-ctl -d "$d" --list-formats-ext 2>/dev/null | grep -q 640x512 && { DEV=${d#/dev/video}; break; }; done
fi
[ "$DEV" = auto ] && { echo "[FAIL] thermal camera (640x512) not found: plug it directly into the laptop"; exit 1; }
arecord -l | grep -q "$UMA8_CARD" || { echo "[FAIL] UMA-8 (card $UMA8_CARD) not found"; exit 1; }
EXTRA=""; [ "$CAM_MODE" = raw ] && EXTRA="--raw --skip-dup 0.5"
tmux new-session -d -s laptop -n cam "bash -lc 'cd $NETCAM && python3 cam_sender.py --device $DEV --width 640 --height 512 $EXTRA; read'"
tmux new-window -t laptop -n audio "bash -lc 'echo sending UMA-8 audio to $JETSON_IP:$AUDIO_PORT (retries every 2 s); \
while true; do arecord -D plughw:CARD=$UMA8_CARD,DEV=0 -t raw -f S32_LE -c 8 -r 48000 --buffer-time=40000 --period-time=8000 \
| nc -N $JETSON_IP $AUDIO_PORT; echo [audio] no receiver / disconnected, retry; sleep 2; done'"
tmux new-window -t laptop -n view "bash -lc 'sleep 20; cd $NETCAM && python3 view_stream.py --host $JETSON_IP --port 9400; read'"
echo "[ OK ] laptop: camera /dev/video$DEV ($CAM_MODE) + UMA-8 audio. tmux attach -t laptop"

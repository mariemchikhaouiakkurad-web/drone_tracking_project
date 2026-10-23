#!/usr/bin/env bash
# start_sound.sh - JETSON (host or container, see SOUND_WHERE): audio receiver + ODAS/classifier + drone warning.
# tmux session "sound": windows rx | odas | warning     (watch: tmux attach -t sound)
set -u
. "$(dirname "$0")/common.sh"
need tmux tmux; need nc netcat-openbsd; need aplay alsa-utils
tmux kill-session -t sound 2>/dev/null; pkill -f "nc -l $AUDIO_PORT"; pkill -f "aplay -D hw:Loopback"; sleep 1
win sound rx "echo 'audio receiver: waiting for the laptop on port $AUDIO_PORT (restarts after each disconnect)'; \
while true; do nc -l $AUDIO_PORT | aplay -D hw:Loopback,0 -t raw -f S32_LE -c 8 -r 48000 -B 100000 -F 10000; \
echo '[rx] laptop disconnected, waiting again'; sleep 1; done"
sleep 1
win sound odas "$(ros_env $SOUND_DOMAIN); cd $DALBD/uma-8/scripts && ./start.sh"
sleep 3
win sound warning "$(ros_env $SOUND_DOMAIN); cd $DALBD/uma-8/scripts && ./warning.sh"
echo "[ OK ] sound started (tmux session 'sound': rx, odas, warning). Start the laptop audio if not running."

#!/usr/bin/env bash
# stop_all.sh - JETSON HOST: stop sound + vision (host and container). Keeps tegrastats unless --all.
set -u
. "$(dirname "$0")/common.sh"
tmux kill-session -t sound 2>/dev/null
# patterns written as [x]yz so pkill does not match (and kill) the bash -lc that runs it
docker exec -u "$CONTAINER_USER" "$CONTAINER" bash -lc "tmux kill-session -t vision 2>/dev/null; tmux kill-session -t sound 2>/dev/null; \
pkill -f '[n]et_camera_node.py'; pkill -f '[t]hermal_tracking.launch'; pkill -f '[s]tate_machine'; pkill -f '[l]atency_trace'; pkill -f '[s]tream_server_node.py'; pkill -f '[c]omponent_container'; \
pkill -f '[n]c -l $AUDIO_PORT'; pkill -f '[a]play -D hw:Loopback'; pkill -f '[a]record -D'; pkill -f '[o]das_server_node'; pkill -x odas_core_node; pkill -f '[c]lassifier_node'; pkill -f '[w]arning_node'; true" 2>/dev/null
pkill -f "nc -l $AUDIO_PORT"; pkill -f "aplay -D hw:Loopback"; pkill -f odas_server_node; pkill -x odas_core_node; pkill -f odaslive; true
[ "${1:-}" = --all ] && sudo pkill -f "tegrastats.*--logfile"
echo "[ OK ] stopped"

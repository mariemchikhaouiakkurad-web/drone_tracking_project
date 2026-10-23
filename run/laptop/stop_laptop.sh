#!/usr/bin/env bash
tmux kill-session -t laptop 2>/dev/null; pkill -f cam_sender.py; pkill -f "arecord -D plughw"; pkill -f view_stream.py; echo "[ OK ] laptop stopped"

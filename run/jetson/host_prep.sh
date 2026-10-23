#!/usr/bin/env bash
# host_prep.sh - JETSON HOST: power mode, clocks, Loopback card, PulseAudio off, power/temperature logger.
set -u
. "$(dirname "$0")/common.sh"
[ "$INSIDE" = 0 ] || { echo "[FAIL] run on the Jetson HOST (akkurad@...), not in the container"; exit 1; }
echo "== power mode";   sudo nvpmodel -q | head -1; sudo jetson_clocks && echo "clocks locked"
echo "== network buffers (large ROS images: without this /image drops from 30 to ~10 Hz)"
sudo sysctl -q -w net.core.rmem_max=2147483647 net.core.rmem_default=8388608 net.core.wmem_max=2147483647 \
  net.ipv4.ipfrag_time=3 net.ipv4.ipfrag_high_thresh=134217728 && echo "buffers set"
[ -f /etc/sysctl.d/60-dalbd-ros.conf ] || printf "net.core.rmem_max=2147483647\nnet.core.rmem_default=8388608\nnet.core.wmem_max=2147483647\nnet.ipv4.ipfrag_time=3\nnet.ipv4.ipfrag_high_thresh=134217728\n" | sudo tee /etc/sysctl.d/60-dalbd-ros.conf >/dev/null
echo "== Loopback";     sudo modprobe snd-aloop
[ -f /etc/modules-load.d/snd-aloop.conf ] || echo snd-aloop | sudo tee /etc/modules-load.d/snd-aloop.conf >/dev/null
sudo chmod a+rw /dev/snd/*   # every user (also the container user admin) can open the sound cards; reset at each boot
echo "== PulseAudio off (it grabs the Loopback card)"
systemctl --user stop pulseaudio.socket pulseaudio.service 2>/dev/null; systemctl --user mask pulseaudio.socket pulseaudio.service 2>/dev/null
echo "== tegrastats logger -> $LOGS/tegrastats.log"
if pgrep -f "tegrastats.*--logfile" >/dev/null; then echo "already running"
else sudo nohup tegrastats --interval 1000 --logfile "$LOGS/tegrastats.log" >/dev/null 2>&1 & sleep 1; echo "started"; fi
echo "== container"
if docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then echo "$CONTAINER running"
  docker exec -u root "$CONTAINER" bash -c "chmod a+rw /dev/snd/*; usermod -aG audio $CONTAINER_USER" && echo "sound cards open for $CONTAINER_USER"
else echo "[WARN] container not running: open one terminal with
  cd ~/workspaces/isaac_ros-dev/src/yolo-pipline/isaac_ros_common && ./scripts/run_dev.sh -d ~/workspaces/isaac_ros-dev -b"; fi

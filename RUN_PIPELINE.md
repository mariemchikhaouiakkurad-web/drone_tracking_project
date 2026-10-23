# DALBD — How to run the pipeline

The **laptop** only reads the thermal camera and the UMA-8 and sends them to the Jetson.
The **Jetson** does all the processing, inside the Isaac ROS container.

There are 3 kinds of terminals. Check the prompt before pasting:

| Name in this file | Where | Prompt | How to open it |
|---|---|---|---|
| **HOST** | Jetson, outside the container | `akkurad@tegra-ubuntu` | on the laptop: `ssh akkurad@192.168.10.1` |
| **CONTAINER** | Jetson, inside the container | `admin@tegra-ubuntu` | in a HOST terminal: `docker exec -it -u admin isaac_ros_dev-aarch64-container bash` |
| **LAPTOP** | the laptop | `mariem@...` | a normal terminal |

Open the terminals **in the order of this file**. Every block is complete: copy the whole block.

---

## 1. Prepare the Jetson — HOST 1

```bash
docker ps --format '{{.Names}}'
```
- If you see `isaac_ros_dev-aarch64-container` → go on.
- If not → start the container (keep this terminal open afterwards, it becomes a container shell):
```bash
cd ~/workspaces/isaac_ros-dev/src/yolo-pipline/isaac_ros_common && ./scripts/run_dev.sh -d ~/workspaces/isaac_ros-dev -b
```

Then, in a **new HOST terminal (HOST 2)**:
```bash
~/DALBD/run/jetson/stop_all.sh
~/DALBD/run/jetson/host_prep.sh
docker exec -u root isaac_ros_dev-aarch64-container bash -c "chmod a+rw /dev/snd/*; usermod -aG audio admin"
```

---

## 2. Check the laptop devices — LAPTOP 1

Laptop on the **charger** and in **performance** mode (on battery the laptop slows down):
```bash
grep -H . /sys/class/power_supply/ADP0/online      # must be 1
powerprofilesctl set performance
```

```bash
v4l2-ctl --list-devices
arecord -l | grep SPK
```
- Thermal camera = `Camera` → note its number (`/dev/video4` → **4**).
- UMA-8 = card `SPK`. If it is missing: replug it on another USB port and run `sudo dmesg -w` (you must see `micArray RAW SPK`).

---

## 3. Sound

**CONTAINER 1 — receive the UMA-8 audio**
```bash
while true; do nc -l 5005 | aplay -D hw:Loopback,0 -t raw -f S32_LE -c 8 -r 48000 -B 100000 -F 10000; echo "[rx] waiting again"; sleep 1; done
```

**LAPTOP 1 — send the UMA-8 audio**
```bash
arecord -D plughw:CARD=SPK,DEV=0 -t raw -f S32_LE -c 8 -r 48000 --buffer-time=40000 --period-time=8000 | nc -N 192.168.10.1 5005
```

**CONTAINER 2 — sound pipeline (ODAS + drone classifier)**
```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=42
cd /workspaces/isaac_ros-dev/src/DALBD/uma-8/scripts && ./start.sh
```

**CONTAINER 3 — drone warning**
```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=42
cd /workspaces/isaac_ros-dev/src/DALBD/uma-8/scripts && ./warning.sh
```

---

## 4. Camera

**CONTAINER 4 — receive the camera**
```bash
source /opt/ros/humble/setup.bash; source /workspaces/isaac_ros-dev/install/setup.bash
export ROS_DOMAIN_ID=0
cd /workspaces/isaac_ros-dev/tools/net_camera
python3 net_camera_node.py --ros-args -p host:=192.168.10.2 -p port:=9300 -p image_topic:=image -p camera_info_topic:=camera_info
```

**LAPTOP 2 — send the camera** (use the number found in step 2; keep `--raw`, otherwise only 15 fps)
```bash
pkill -f "[c]am_sender.py"; sleep 1
cd ~/net_camera && python3 cam_sender.py --device 4 --width 640 --height 512 --fps 30 --raw
```
You must see: `[STAT] 640x512 | 30.0 fps`.

---

## 5. Detection, tracking and fusion

**CONTAINER 5 — recorder** (optional: saves one CSV per run)
```bash
source /opt/ros/humble/setup.bash; source /workspaces/isaac_ros-dev/install/setup.bash
mkdir -p /workspaces/isaac_ros-dev/fusion_logs/runs
ros2 run dalbd_fusion run_recorder
```

**CONTAINER 6 — YOLO + tracker** (`detector_every` = number of frames between two YOLO checks)
```bash
source /opt/ros/humble/setup.bash; source /workspaces/isaac_ros-dev/install/setup.bash
export ROS_DOMAIN_ID=0
ros2 launch drone_target_tracker thermal_tracking.launch.py detector_every:=25 video_device:=/dev/null_cam 2>&1 | tee -i /workspaces/isaac_ros-dev/fusion_logs/runs/launch.log
```
Wait 20–30 s until the tracker lines appear.

**CONTAINER 7 — fusion**
```bash
source /opt/ros/humble/setup.bash; source /workspaces/isaac_ros-dev/install/setup.bash
export ROS_DOMAIN_ID=0
ros2 run dalbd_fusion state_machine --ros-args --params-file $(ros2 pkg prefix dalbd_fusion)/share/dalbd_fusion/config/fusion.yaml
```

---

## 6. See the result

### Tracker image (camera + tracker box)

**CONTAINER 8 — send the tracker image to the laptop** (start it only when CONTAINER 6 shows tracker lines)
```bash
source /opt/ros/humble/setup.bash; source /workspaces/isaac_ros-dev/install/setup.bash
export ROS_DOMAIN_ID=0
cd /workspaces/isaac_ros-dev/tools/net_camera
pkill -f "[s]tream_server_node"; sleep 1
python3 stream_server_node.py --ros-args -p topic:=/target/debug_image -p port:=9400 -p quality:=80
```

**LAPTOP 3 — show the tracker image** (`q` = quit, `s` = snapshot)
```bash
cd ~/net_camera && python3 view_stream.py --host 192.168.10.1 --port 9400
```

### RViz — fusion 3D view (sound arrow, camera field of view, fused direction)

**CONTAINER 9 — RViz markers**
```bash
source /opt/ros/humble/setup.bash; source /workspaces/isaac_ros-dev/install/setup.bash
export ROS_DOMAIN_ID=42
ros2 run dalbd_fusion fusion_viz
```

**LAPTOP 4 — RViz fusion view**
```bash
scp akkurad@192.168.10.1:~/DALBD/dalbd_fusion/rviz/dalbd.rviz ~/
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=42 ROS_LOCALHOST_ONLY=0
rviz2 -d ~/dalbd.rviz
```
- grey pyramid = camera field of view
- yellow arrow = sound direction
- blue arrow = fused output from vision, green arrow = fused output from sound
- text = current state (SEARCH / VISUAL_TRACK / SOUND_TRACK)

### RViz — camera view

**LAPTOP 5 — RViz camera view**
```bash
scp akkurad@192.168.10.1:~/DALBD/rviz/drone_target_view.rviz ~/
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=0 ROS_LOCALHOST_ONLY=0
rviz2 -d ~/drone_target_view.rviz
```

### Live latency and state (optional)

**CONTAINER 10 — latency of each part, every second**
```bash
source /opt/ros/humble/setup.bash; source /workspaces/isaac_ros-dev/install/setup.bash
ros2 run dalbd_fusion latency_trace
```

**CONTAINER 11 — fusion state**
```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=0
ros2 topic echo /fused/state
```

---

## 7. Check that everything runs — HOST 2

```bash
~/DALBD/run/jetson/status.sh
```
Every topic must show data. `/image` ≈ 30 Hz.

---

## 8. Change the number of frames

1. **CONTAINER 6:** Ctrl-C, then run the launch again with the new value, e.g. `detector_every:=50`.
2. **CONTAINER 8:** Ctrl-C, then run it again (wait for the tracker lines first).
3. Nothing else changes. The recorder (CONTAINER 5) starts a new CSV by itself.

---

## 9. Results on the laptop — LAPTOP

```bash
scp -r akkurad@192.168.10.1:~/workspaces/isaac_ros-dev/fusion_logs/runs ~/dalbd_runs
python3 ~/runs_report.py ~/dalbd_runs
xdg-open ~/dalbd_runs/report/runs_report.html
```
For a fair comparison of frame values: same duration (at least 3 min), target always in view, sound running.

---

## 9b. Record and analyse WHEN each part works (YOLO, tracker, sound) + power + temperature

Same session name in every terminal (change `test1` for each new session).

**CONTAINER 5 — recorder** (start it BEFORE CONTAINER 6; it also loads the sound messages)
```bash
source /opt/ros/humble/setup.bash; source /workspaces/isaac_ros-dev/install/setup.bash
source /workspaces/isaac_ros-dev/sound_ws/install_container/setup.bash
REC=/workspaces/isaac_ros-dev/fusion_logs/rec_test1; mkdir -p $REC
ros2 run dalbd_fusion run_recorder out:=$REC
```
**CONTAINER 6 — YOLO + tracker** (one run per frame value, 3 min each, Ctrl-C between runs)
```bash
source /opt/ros/humble/setup.bash; source /workspaces/isaac_ros-dev/install/setup.bash
export ROS_DOMAIN_ID=0
REC=/workspaces/isaac_ros-dev/fusion_logs/rec_test1
ros2 launch drone_target_tracker thermal_tracking.launch.py detector_every:=25 video_device:=/dev/null_cam 2>&1 | tee -i $REC/launch.log
```
After the last run: Ctrl-C in CONTAINER 6, then in CONTAINER 5.

**LAPTOP — copy and analyse**
```bash
scp -r akkurad@192.168.10.1:~/workspaces/isaac_ros-dev/fusion_logs/rec_test1 ~/
python3 ~/activity_report.py ~/rec_test1 && xdg-open ~/rec_test1/activity/activity_report.html
python3 ~/runs_report.py ~/rec_test1 && xdg-open ~/rec_test1/report/runs_report.html
```
`activity_report.html`: per run, a timeline of every YOLO inference, every tracked image and every sound output,
tracker status and fusion state, with latency, power, GPU / CPU load and temperatures on the same time axis,
a 15 s zoom, and a table (YOLO active %, YOLO rate searching vs tracking, real time between two YOLO checks,
tracker / sound timing, power and temperature by mode, power with vs without YOLO).

## 10. Stop

Ctrl-C in every terminal (laptop too), or on **HOST 2**:
```bash
~/DALBD/run/jetson/stop_all.sh
```

---

## 11. If something goes wrong

| Problem | What to do |
|---|---|
| `Package 'drone_target_tracker' not found` | the `source ...` line was not run in that terminal: run it |
| `No such container` | the container is not running: step 1 |
| `run_dev.sh` starts building an image | Ctrl-C, and always add `-b` |
| CONTAINER 2: `Cannot get card index for Loopback` | HOST 2: `docker exec -u root isaac_ros_dev-aarch64-container bash -c "chmod a+rw /dev/snd/*"` |
| LAPTOP 1: `Cannot get card index for SPK` | UMA-8 not seen: replug it (other port or cable), `sudo dmesg -w` |
| LAPTOP 2: `cannot open camera` | the camera is used by another program: `pkill -f "[c]am_sender.py"`, check the number with `v4l2-ctl --list-devices` |
| `/image` only 15 Hz | LAPTOP 2 was started without `--raw` |
| `/image` ≈ 6–10 Hz while LAPTOP 2 says `30.0 fps` | network buffers too small for the large images: HOST 2 `~/DALBD/run/jetson/host_prep.sh` (sets them), then restart CONTAINER 4 and 6 |
| Everything slow, `tegrastats` shows CPU cores below 1984 MHz | clocks not locked: HOST 2 `~/DALBD/run/jetson/host_prep.sh` (after every boot) |
| Slow motion, two `tracker_node` in `pgrep -af "[t]racker_node"` | a leftover from an old launch: `kill` the old PID; stop the launch with ONE Ctrl-C and use `tee -i`, never plain `tee` |
| CONTAINER 4: `connected, no frames` | two camera receivers are running: `stop_all.sh`, then start one of each again |
| LAPTOP 3 window black | restart CONTAINER 8 (after CONTAINER 6 shows tracker lines) |
| CONTAINER 8: `Address already in use` | `pkill -f "[s]tream_server_node"`, then start it again |
| RViz shows nothing | LAPTOP: `sudo ufw disable`; in RViz set **Fixed Frame = camera**; CONTAINER 9 must run in domain 42 |
| RViz: `rviz2: command not found` | LAPTOP: `sudo apt install -y ros-humble-rviz2` |
| Tracker box stays on the background | `detector_every` too large: use 25–50 |
| Slow after a reboot | HOST 2: `~/DALBD/run/jetson/host_prep.sh` (MAXN + clocks) |

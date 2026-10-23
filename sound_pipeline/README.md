# DALBD — drone detection and tracking (thermal vision + acoustic) on Jetson Orin NX

DALBD finds a drone with two independent pipelines and fuses them:

- **Vision**: UV640-9 thermal camera → YOLOv8 (TensorRT, Isaac ROS) → DyHiT single-target tracker → `/target/bearing`
- **Sound**: UMA-8 microphone array → ODAS (direction) → known-source drop → MobileNetV2 (drone / no drone) → `/drone_warning`
- **Fusion**: state machine `SEARCH / VISUAL_TRACK / SOUND_TRACK`. Vision always wins; when vision loses the target,
  the sound direction is used until vision finds it again → `/fused/state`, `/fused/bearing`

```
LAPTOP 192.168.10.2                                    JETSON A 192.168.10.1 (JNX120X + Orin NX 16GB)
 thermal camera /dev/video4 ─ cam_sender.py :9300 ──Ethernet──> [container, domain 0]
                                                                  net_camera_node → /image
                                                                  YOLOv8 (bestperson.plan) → /detections_output
                                                                  DyHiT tracker → /target/bearing, /target/status
                                                                  dalbd_fusion state_machine ──┐
 UMA-8 (card SPK) ─ arecord | nc :5005 ─────────Ethernet──> [host, domain 42]                  │ joins both
                                                                  nc | aplay → Loopback card   │ domains
                                                                  ODAS → /raw/sst              │
                                                                  MobileNetV2 → /drone_warning ┘
 view_stream.py  <────────── tracker image :9400 ────────────── stream_server_node
```

| Item | Value |
|---|---|
| Board | Auvidea JNX120X carrier + Jetson Orin NX 16GB, supply 12–24 V (used: 23 V) |
| Software | JetPack 6.2.3 (L4T R36.5.0), Ubuntu 22.04, ROS 2 Humble, TensorRT 10.3 (host and container), Isaac ROS 3.2 |
| Power mode | **MAXN** (`nvpmodel` mode 0) + `jetson_clocks` |
| ROS domains | vision **0**, sound **42** (the fusion and the trace tools join both) |
| Container | `isaac_ros_dev-aarch64-container`, user `admin`, host networking (sees host topics of the same domain) |

---

## 1. Where everything is

| What | JETSON host | JETSON container |
|---|---|---|
| Isaac ROS workspace | `~/workspaces/isaac_ros-dev` | `/workspaces/isaac_ros-dev` |
| DALBD (this folder) | `~/DALBD` → link to `~/workspaces/isaac_ros-dev/src/DALBD` | `/workspaces/isaac_ros-dev/src/DALBD` |
| Run scripts + settings | `~/DALBD/run/` (`run.conf` = the only file to edit) | same, under `/workspaces/...` |
| Fusion package | `~/DALBD/dalbd_fusion` | built into `/workspaces/isaac_ros-dev/install` |
| Vision repo (YOLO + tracker) | `~/workspaces/isaac_ros-dev/src/yolo-pipline` | `/workspaces/isaac_ros-dev/src/yolo-pipline` |
| Models (all) | `~/workspaces/isaac_ros-dev/src/model/` | `/workspaces/isaac_ros-dev/src/model/` |
| Camera bridge tools | `~/workspaces/isaac_ros-dev/tools/net_camera` | `/workspaces/isaac_ros-dev/tools/net_camera` |
| Logs (fusion, trace, sweep, tegrastats) | `~/workspaces/isaac_ros-dev/fusion_logs` | `/workspaces/isaac_ros-dev/fusion_logs` |
| Sound ROS workspace (`odas_ros`, `audio_utils`) | `~/ros2_ws` → link to `~/workspaces/isaac_ros-dev/sound_ws` (host build: `install/`) | `/workspaces/isaac_ros-dev/sound_ws` (container build: `install_container/`) |
| ONNX Runtime 1.16.3 (for `audio_utils`) | `sound_ws/onnxruntime-linux-aarch64-1.16.3` (+ `/usr/local/lib`) | copied to `/usr/local/lib`, `/usr/local/include/onnxruntime` |
| Sound logs | `~/dataset/odas_logs` | `/home/akkurad/dataset/odas_logs` (link) |
| Host paths in the container | — | `/home/akkurad/workspaces`, `/home/akkurad/DALBD`, `/home/akkurad/dataset` are links, so one config works in both |

| LAPTOP | |
|---|---|
| `~/DALBD/run/` | copy of the run scripts (`laptop/`) + the same `run.conf` |
| `~/net_camera/` | `cam_sender.py`, `view_stream.py` |

### DALBD folder

```
DALBD/
├── README.md                this file
├── ARCHITECTURE.md, TOPICS.md
├── run/                     start / check / stop everything (see section 3)
│   ├── run.conf             IPs, ports, domains, detector_every, camera mode, where sound runs
│   ├── jetson/              host_prep, start_all, start_sound, start_vision, status, stop_all, common.sh
│   └── laptop/              start_laptop, stop_laptop
├── dalbd_fusion/            ROS 2 package: fusion state machine + tools (README inside)
├── uma-8/                   sound pipeline: scripts/ (start.sh, warning.sh, classifier.sh), config/drone_classifier.yaml
├── nodes/                   drone_classifier_node.py, mnv2_core.py, sss_target_classifier_node.py
├── tools/                   make_sss_cfg.py, sss_classify.py, classify_wav.py, mnv2_inspect.py, enable_nulled_sss.py
├── warning/                 drone warning
├── rviz/                    drone_target_view.rviz (vision camera view)
├── docs/                    pictures and reports
└── yolo_pipline_local_changes.diff   backup of the local (uncommitted) changes of the vision repo
```

### Models (`src/model/`)

| File | Used by |
|---|---|
| `bestperson.onnx` + `bestperson.plan` | vision (default of `thermal_tracking.launch.py`) |
| `mobilenetv2_drone.onnx` + `.engine` | sound classifier (`uma-8/config/drone_classifier.yaml`) |
| `best_drone.onnx` | drone YOLO model, not used yet |
| `best.pt` | training weights (source of the YOLO ONNX) |
| DyHiT checkpoint | `yolo-pipline/drone_target_tracker/tracker_lib/trackers/dyhit/models/DyHiT_ep0060.pth.tar` |

Rebuild a TensorRT engine (same TensorRT on host and container, so an engine works in both):
```bash
# JETSON host
/usr/src/tensorrt/bin/trtexec --onnx=$HOME/workspaces/isaac_ros-dev/src/model/mobilenetv2_drone.onnx \
  --saveEngine=$HOME/workspaces/isaac_ros-dev/src/model/mobilenetv2_drone.engine --fp16
/usr/src/tensorrt/bin/trtexec --loadEngine=$HOME/workspaces/isaac_ros-dev/src/model/mobilenetv2_drone.engine --iterations=10 | grep -E "PASSED|FAILED"
```
Use host paths (`$HOME/workspaces/...`) on the host and `/workspaces/...` in the container.

---

## 2. One-time setup

Container extras for the sound part (already installed and saved in the image): `tmux iproute2 procps psmisc alsa-utils netcat-openbsd
libasound2-dev libfftw3-dev libconfig-dev libpulse-dev`, pip (system-wide) `pycuda libconf onnxruntime`, ONNX Runtime 1.16.3
in `/usr/local`, links `/home/akkurad/{workspaces,DALBD,dataset}`. `start_all.sh` adds `admin` to the `audio` group at each
start. Rebuild the sound workspace in the container:
```bash
cd /workspaces/isaac_ros-dev/sound_ws && source /opt/ros/humble/setup.bash
colcon build --base-paths src --build-base build_container --install-base install_container --cmake-args -DCMAKE_BUILD_TYPE=Release
```
(on the host: `cd ~/ros2_ws && colcon build --base-paths src`)

### 2.1 Power mode MAXN (JETSON host)
```bash
sudo nvpmodel -q                 # current mode
sudo nvpmodel -m 0               # MAXN (may ask to reboot)
sudo jetson_clocks               # lock clocks at max - after EVERY reboot (host_prep.sh does it)
```
Modes on Orin NX 16GB: 0 = MAXN, 1 = 10W, 2 = 15W, 3 = 25W. MAXN needs a supply that delivers the peak power
(bench supply at 23 V was fine).

### 2.2 Start the Isaac ROS container (JETSON host)
```bash
cd ~/workspaces/isaac_ros-dev/src/yolo-pipline/isaac_ros_common
./scripts/run_dev.sh -d ~/workspaces/isaac_ros-dev -b
```
- **Always `-b`** (skip image build). Without it `run_dev.sh` tries to rebuild the image from the internet and,
  if it succeeds, replaces the saved image.
- The container is started with `--rm`: it disappears when its first shell exits. Keep that shell open.
- More container terminals (JETSON host): `docker exec -it -u admin isaac_ros_dev-aarch64-container bash`
- After installing packages with `apt` in the container, save them (JETSON host):
  `docker commit isaac_ros_dev-aarch64-container isaac_ros_dev-aarch64`

### 2.3 Build the fusion package (JETSON container)
```bash
cd /workspaces/isaac_ros-dev
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select dalbd_fusion
source install/setup.bash
ros2 pkg executables dalbd_fusion        # detector_sweep fake_sound fusion_viz latency_trace state_machine state_report
python3 -m pytest -q src/DALBD/dalbd_fusion/test      # 24 passed
```
Vision packages (already built): `colcon build --symlink-install --packages-select isaac_ros_yolov8 drone_target_tracker`

### 2.4 Laptop files (LAPTOP)
```bash
mkdir -p ~/DALBD
scp -r akkurad@192.168.10.1:~/DALBD/run ~/DALBD/
scp -r akkurad@192.168.10.1:~/workspaces/isaac_ros-dev/tools/net_camera ~/
sudo apt install -y tmux v4l-utils netcat-openbsd alsa-utils
chmod +x ~/DALBD/run/laptop/*.sh
```
Network: laptop `192.168.10.2`, Jetson A `192.168.10.1` on the same Ethernet link. Check: `ping -c 3 192.168.10.1`.

---

## 3. Run the whole pipeline (scripts)

Order does not matter: receivers and senders retry until they meet.

**JETSON host — T1** (container must already run, section 2.2)
```bash
~/DALBD/run/jetson/start_all.sh            # or: start_all.sh 25   (detector_every = 25 for this run)
```
It runs `host_prep.sh` (MAXN check, `jetson_clocks`, Loopback card, PulseAudio off, tegrastats logger), then the
sound part (host, tmux session `sound`) and the vision part (container, tmux session `vision`).

**LAPTOP — T1**
```bash
~/DALBD/run/laptop/start_laptop.sh
```
Finds the 640x512 thermal camera, starts the camera sender, the UMA-8 audio sender and (after 20 s) the tracker
image viewer (tmux session `laptop`).

**JETSON host — T2** (check, about 40 s)
```bash
~/DALBD/run/jetson/status.sh
```
Expected: `/image` ≈ 30 Hz, `/detections_output` and `/target/bearing` with data, `/raw/sst` and `/drone_warning`
with data, `/fused/state` with data, board line `VDD_IN … GR3D_FREQ … tj@…`.

### Watch the running parts
| Part | Command |
|---|---|
| sound (JETSON host) | `tmux attach -t sound` (windows: rx, odas, warning) |
| vision (JETSON host) | `docker exec -it -u admin isaac_ros_dev-aarch64-container tmux attach -t vision` (windows: cam, track, fusion, trace, stream) |
| laptop (LAPTOP) | `tmux attach -t laptop` (windows: cam, audio, view) |

tmux keys: `Ctrl-b n` next window, `Ctrl-b p` previous, `Ctrl-b w` list, `Ctrl-b d` leave (programs keep running).

### Stop
```bash
~/DALBD/run/jetson/stop_all.sh             # JETSON host   (--all also stops the tegrastats logger)
~/DALBD/run/laptop/stop_laptop.sh          # LAPTOP
```

### Settings (`run/run.conf`, same file on laptop and Jetson)
| Key | Default | Meaning |
|---|---|---|
| `JETSON_IP` / `LAPTOP_IP` | 192.168.10.1 / .2 | Ethernet addresses |
| `DETECTOR_EVERY` | 10 | YOLO checks the tracker every N tracker images (only while tracking) |
| `CAM_DEVICE` | auto | auto = first 640x512 camera, or a number (4 = `/dev/video4`) |
| `CAM_MODE` | mjpg | `raw` = uncompressed + skip duplicate frames |
| `STREAM_TOOLS` | 1 | also start `latency_trace` and the tracker image stream (port 9400) |
| `AUDIO_PORT`, `UMA8_CARD` | 5005, SPK | audio link and UMA-8 ALSA card name |
| `SOUND_DOMAIN` / `VISION_DOMAIN` | 42 / 0 | ROS domains |
| `SOUND_WHERE` | container | where the sound part runs: `container` or `host` (both builds exist; switch back any time) |

---

## 4. Run terminal by terminal (everything on the Jetson in the container)

Use **either** section 3 (scripts, tmux) **or** this section, never both: two camera receivers or two audio receivers
steal each other's connection ("connected, no frames").

| Label | Prompt | Open it with |
|---|---|---|
| LAPTOP | laptop user | a terminal on the laptop |
| JETSON host | `akkurad@tegra-ubuntu` | `ssh akkurad@192.168.10.1` |
| JETSON container | `admin@tegra-ubuntu` | from a JETSON host terminal: `docker exec -it -u admin isaac_ros_dev-aarch64-container bash` |

### Step 1 — prepare (JETSON host — H1, once per boot / container start)
```bash
docker ps --format '{{.Names}}' | grep -q isaac_ros_dev-aarch64-container || \
  (cd ~/workspaces/isaac_ros-dev/src/yolo-pipline/isaac_ros_common && ./scripts/run_dev.sh -d ~/workspaces/isaac_ros-dev -b)
```
(if it opened a container shell, keep it open and use another host terminal for the rest)
```bash
~/DALBD/run/jetson/stop_all.sh                      # no old copies left (scripts or manual)
~/DALBD/run/jetson/host_prep.sh                     # MAXN check, clocks, Loopback, PulseAudio off, power log
docker exec -u root isaac_ros_dev-aarch64-container usermod -aG audio admin     # sound cards for 'admin'
```
Optional, once per container start — every new container terminal then loads ROS + the workspace by itself:
```bash
docker exec -u admin isaac_ros_dev-aarch64-container bash -c "grep -q isaac_ros-dev/install ~/.bashrc || echo 'source /opt/ros/humble/setup.bash; source /workspaces/isaac_ros-dev/install/setup.bash' >> ~/.bashrc"
```

### Step 2 — sound (JETSON container, domain 42)
**C1 — audio receiver (laptop UMA-8 → Loopback card)**
```bash
while true; do nc -l 5005 | aplay -D hw:Loopback,0 -t raw -f S32_LE -c 8 -r 48000 -B 100000 -F 10000; echo "[rx] waiting again"; sleep 1; done
```
**C2 — sound pipeline (ODAS → known-source drop → MobileNetV2 → direction)**
```bash
source /opt/ros/humble/setup.bash; export ROS_DOMAIN_ID=42
cd /workspaces/isaac_ros-dev/src/DALBD/uma-8/scripts && ./start.sh
```
**C3 — drone warning (`/drone_warning`)**
```bash
source /opt/ros/humble/setup.bash; export ROS_DOMAIN_ID=42
cd /workspaces/isaac_ros-dev/src/DALBD/uma-8/scripts && ./warning.sh
```
**LAPTOP — L1 — UMA-8 → Jetson**
```bash
arecord -D plughw:CARD=SPK,DEV=0 -t raw -f S32_LE -c 8 -r 48000 --buffer-time=40000 --period-time=8000 | nc -N 192.168.10.1 5005
```

### Step 3 — vision + fusion (JETSON container, domain 0)
First line of every terminal below:
```bash
source /opt/ros/humble/setup.bash; source /workspaces/isaac_ros-dev/install/setup.bash; export ROS_DOMAIN_ID=0
```
**C4 — camera receiver (laptop → `/image`)**
```bash
cd /workspaces/isaac_ros-dev/tools/net_camera
python3 net_camera_node.py --ros-args -p host:=192.168.10.2 -p port:=9300 -p image_topic:=image -p camera_info_topic:=camera_info
```
**LAPTOP — L2 — thermal camera → Jetson** (camera number: `v4l2-ctl --list-devices`)
```bash
cd ~/net_camera && python3 cam_sender.py --device 4 --width 640 --height 512
```
**C5 — YOLO + DyHiT tracker** (number of frames = `detector_every`; to change it: Ctrl-C here and rerun with another value)
```bash
ros2 launch drone_target_tracker thermal_tracking.launch.py detector_every:=10 video_device:=/dev/null_cam
```
**C6 — fusion state machine**
```bash
ros2 run dalbd_fusion state_machine --ros-args --params-file $(ros2 pkg prefix dalbd_fusion)/share/dalbd_fusion/config/fusion.yaml
```
**C7 — tracker image → laptop**
```bash
cd /workspaces/isaac_ros-dev/tools/net_camera
python3 stream_server_node.py --ros-args -p topic:=/target/debug_image -p port:=9400 -p quality:=80
```
**LAPTOP — L3 — tracker image window**
```bash
cd ~/net_camera && python3 view_stream.py --host 192.168.10.1 --port 9400
```

### Step 4 — watch (optional)
**C8 — live latency per stage** (domain 0 line first): `ros2 run dalbd_fusion latency_trace`
**C9 — fused state** (domain 0 line first): `ros2 topic echo /fused/state`
**C10 — RViz markers**
```bash
source /opt/ros/humble/setup.bash; source /workspaces/isaac_ros-dev/install/setup.bash; export ROS_DOMAIN_ID=42
ros2 run dalbd_fusion fusion_viz
```
**LAPTOP — L4 — RViz**
```bash
source /opt/ros/humble/setup.bash; export ROS_DOMAIN_ID=42 ROS_LOCALHOST_ONLY=0
rviz2 -d ~/dalbd.rviz
```

### Step 5 — check (JETSON host)
```bash
~/DALBD/run/jetson/status.sh
```
| Topic | Expected |
|---|---|
| `/image` | ≈ 30 Hz (C4 + L2) |
| `/detections_output`, `/target/bearing` | data (C5) |
| `/raw/sst` | data (C1 + C2 + L1) |
| `/drone_warning` | data (C3) |
| `/fused/state` | data (C6) |

### Step 6 — stop
Ctrl-C in every terminal (laptop too), or JETSON host: `~/DALBD/run/jetson/stop_all.sh` (stops all Jetson terminals at once).

---

## 5. Measurement and analysis tools (`dalbd_fusion`, JETSON container)

### Live latency trace
```bash
ros2 run dalbd_fusion latency_trace
```
One line per second (YOLO / tracker / sound rate and latency, fusion state) + EVENT lines on every change.
Press ENTER when you act (step in view, start the drone sound…): the delay of each stage's first reaction is printed.
Ctrl-C → `fusion_logs/trace_events_<date>.csv`, `trace_stats_<date>.csv`. Plot a saved run:
```bash
ros2 run dalbd_fusion latency_trace --plot /workspaces/isaac_ros-dev/fusion_logs/trace_events_<date>.csv
```
Also publishes `/trace/*` (Float32) for `rqt_plot` / PlotJuggler.

### Consumption / performance sweep over `detector_every`
Stop only the tracker window first (JETSON host):
```bash
docker exec -u admin isaac_ros_dev-aarch64-container tmux kill-window -t vision:track
```
**JETSON container**
```bash
ros2 run dalbd_fusion detector_sweep --values 25,50,75,100 --duration 60 --ask-supply \
  --tegrastats-log /workspaces/isaac_ros-dev/fusion_logs/tegrastats.log
```
- `--values` list (`25,50,100,500`) or range (`25:500:25`); without it, it asks.
- For each N: starts the launch, waits for TensorRT, asks you to stand in view and press ENTER, records, stops.
- `--ask-supply`: type the bench supply V and A after each case → whole-system power and energy.
- Output folder `fusion_logs/sweep_<date>/`: `case<i>_N<N>.csv`, `consumption.csv` (power per rail, energy,
  temperatures, fan, CPU/RAM per process, % vs case 1), `summary.csv`, `sweep_report.png`, `sweep_report.html`, ranking.
- Redo the report: `ros2 run dalbd_fusion detector_sweep --report /workspaces/isaac_ros-dev/fusion_logs/sweep_<date>`

### Automatic CSV for every vision run (`run_recorder`)
Leave it running in its own container terminal; each time you start the tracking launch by hand (any `detector_every`),
it opens a new CSV, and closes it when you stop the launch (Ctrl-C).

**JETSON container — R**
```bash
ros2 run dalbd_fusion run_recorder
```
**JETSON container — C5** (as usual; `| tee …` adds the tracker counters to the CSV)
```bash
ros2 launch drone_target_tracker thermal_tracking.launch.py detector_every:=10 video_device:=/dev/null_cam 2>&1 | tee /workspaces/isaac_ros-dev/fusion_logs/runs/launch.log
```
Files in `fusion_logs/runs/`:
| File | Content |
|---|---|
| `run_<date>_<time>_N<N>.csv` | one row per second: power per rail, GPU load + frequency, CPU load + frequency, RAM, every temperature, fan, CPU/RAM of each part (YOLO, tracker, camera, fusion, sound rx / ODAS / classifier / warning), YOLO / tracker / sound rate + latency (mean, max), tracker status, sound level + probability, fusion state, tracker counters |
| `run_<…>_N<N>_events.csv` | every change: YOLO sees / loses, tracking / searching, sound DRONE / CLEAR, fusion state |
| `runs_summary.csv` | one line per run: N, duration, mean / max of every column, % tracking, % VISUAL_TRACK / SOUND_TRACK / SEARCH, energy (Wh) — compare the N values here |

### Full recording session (resources + latency of every part, one folder)
Pipeline running as in section 4, **C5 stopped** (the recorder starts YOLO + tracker itself). Same folder name in the 3 terminals:
```bash
REC=/workspaces/isaac_ros-dev/fusion_logs/rec_$(date +%F_%H%M); mkdir -p $REC      # run once, then copy the printed name
```
- **R1** `ros2 run dalbd_fusion latency_trace log_dir:=$REC` — vision, tracker, sound, fusion latency every second + reaction test
- **R2** `cd $REC && python3 /workspaces/isaac_ros-dev/tools/net_camera/latency_probe.py --ros-args -p topics:="['/image','/tensor_pub','/tensor_sub','/detections_output','/target/bearing']"` — time spent in each YOLO stage
- **R3** `ros2 run dalbd_fusion detector_sweep --values 10 --duration 300 --out $REC --tegrastats-log /workspaces/isaac_ros-dev/fusion_logs/tegrastats.log --ask-supply` — power per rail, GPU/CPU load, RAM, every temperature, fan, CPU/RAM per process (YOLO, tracker, camera, fusion, sound rx / ODAS / classifier / warning), YOLO + tracker rate and latency

Stop R1 and R2 with Ctrl-C after R3 ends. Folder content: `case1_N10.csv` (one row per second), `consumption.csv`, `summary.csv`,
`sweep_report.html/.png`, `trace_events_*.csv`, `trace_stats_*.csv`, `latency_*.csv`. Copy to the laptop:
`scp -r akkurad@192.168.10.1:~/workspaces/isaac_ros-dev/fusion_logs/rec_<name> ~/`

### Fusion state report
```bash
ros2 run dalbd_fusion state_report        # last fusion_<date>.csv → state table + timeline PNG
```

### RViz (fusion view)
**JETSON container**
```bash
ROS_DOMAIN_ID=42 ros2 run dalbd_fusion fusion_viz
```
**LAPTOP** (ROS 2 Humble installed; copy `dalbd_fusion/rviz/dalbd.rviz` once)
```bash
scp akkurad@192.168.10.1:~/DALBD/dalbd_fusion/rviz/dalbd.rviz ~/
ROS_DOMAIN_ID=42 rviz2 -d ~/dalbd.rviz
```
Grey pyramid = camera field of view, yellow arrow = sound direction, blue / green arrow = fused output
(vision / sound), text = state.

### Power and temperature
```bash
tail -f ~/workspaces/isaac_ros-dev/fusion_logs/tegrastats.log        # JETSON host (started by host_prep.sh)
```

---

## 6. Configuration files

| File | What to change there |
|---|---|
| `run/run.conf` | IPs, ports, domains, `DETECTOR_EVERY`, camera mode |
| `dalbd_fusion/config/fusion.yaml` | `lost_s`, timeouts, `sound_axes`, calibration (`sound_yaw_offset_deg`, `sound_az_sign`) — rebuild `dalbd_fusion` after editing |
| `yolo-pipline/drone_target_tracker/config/tracker.yaml` | tracker decisions: `verify_iou`, `trust_tracker_s`, `max_catch_up`… |
| `uma-8/config/drone_classifier.yaml` | MobileNetV2 `model_path`, audio window, decision thresholds |

Inspect the running tracker parameters (JETSON container): `ros2 param list /tracker_node`

---

## 7. Measured results (so far)

| | 15 W mode | MAXN |
|---|---|---|
| YOLO latency | ≈ 145 ms | ≈ 67 ms |
| Tracker latency | ≈ 56 ms | ≈ 25 ms |
| YOLO rate while searching | — | 26–29 Hz |

- Camera: 30 fps of real new frames over USB to the laptop. The thermal camera pauses ≈ 0.5 s at each shutter
  calibration (FFC); the tracker keeps the target through it.
- `detector_every` counts tracker images, not seconds, and only applies while tracking (searching = YOLO on every image).
- Energy per `detector_every` value: run the sweep (section 5) — `consumption.csv` / `sweep_report.html`.

---

## 8. Second Jetson (Jetson B)

Jetson B (`192.168.10.3`, same Ethernet switch) reads Jetson A's topics in domain 0 and keeps its GSM modem
(`wwan0`) as the internet route: ROS 2 traffic is limited to the Ethernet interface (Fast DDS profile), multicast is
routed to Ethernet, and the firewall blocks ROS ports on `wwan0`. It can share the thermal image as an RTSP stream
(H.264 hardware encoder + MediaMTX, password-protected, only TCP 8554 open on `wwan0`): see the Jetson B kit (`jetsonB_stream.zip`, its `README.md`).

---

## 9. Troubleshooting

| Symptom | Fix |
|---|---|
| `docker: command not found` / `jetson_clocks: not found` | you are in the container — run it on the JETSON host |
| `/workspaces/...: No such file` on the host | host path is `~/workspaces/...` |
| `No such container: isaac_ros_dev-aarch64-container` | container was removed on exit — start it again (section 2.2, with `-b`) |
| `run_dev.sh` starts building / `lookup nvcr.io … timeout` | Ctrl-C, add `-b` |
| `aplay: … Device or resource busy` (Loopback) | `fuser -v /dev/snd/*`; old `aplay` → `pkill aplay`; PulseAudio → `systemctl --user stop pulseaudio.socket pulseaudio.service; systemctl --user mask pulseaudio.socket pulseaudio.service` |
| `arecord: … No such file` on the laptop | card name: `arecord -l` (UMA-8 = `SPK`), write it without `<…>` |
| `/image` no data | laptop `cam_sender.py` running? `ping 192.168.10.2`; camera number (`v4l2-ctl --list-devices`) |
| C4 `connected, no frames` while the laptop sends | a second camera receiver took the connection (scripts + manual at the same time): `stop_all.sh`, start one of each again |
| `ss` / `tmux` / `aplay`: command not found in the container | `docker exec -u root isaac_ros_dev-aarch64-container apt-get install -y iproute2 procps psmisc tmux alsa-utils netcat-openbsd`, then `docker commit …` |
| `aplay: no soundcards found` in the container | `docker exec -u root isaac_ros_dev-aarch64-container usermod -aG audio admin`, open a new container terminal |
| `Package 'drone_target_tracker' not found` | source line missing: `source /workspaces/isaac_ros-dev/install/setup.bash` |
| Topics not seen between host and container | same `ROS_DOMAIN_ID` on both (vision 0, sound 42) |
| Jetson has no internet (apt fails) | LAPTOP: `sudo sysctl -w net.ipv4.ip_forward=1; sudo iptables -t nat -A POSTROUTING -o wlp108s0 -j MASQUERADE` — JETSON host: `sudo ip route replace default via 192.168.10.2` and a DNS (`8.8.8.8`) in `/etc/resolv.conf` |
| Packages gone after the container restarts | reinstall, then `docker commit isaac_ros_dev-aarch64-container isaac_ros_dev-aarch64` |
| Slow detection after reboot | `sudo nvpmodel -q` (must be MAXN), `sudo jetson_clocks` |
| USB camera / UMA-8 on the carrier J41 port: `error -71`, full-speed only | signal quality of the hand-made cable; use the Ethernet setup (laptop) or a proper USB 2.0 high-speed cable (90 Ω twisted, shielded) |

---

## 10. Work log (what was done)

- Vision pipeline (YOLOv8 TensorRT + DyHiT) running in the Isaac ROS container from the thermal camera streamed by
  the laptop; sound pipeline (UMA-8 → ODAS → known-source drop → MobileNetV2 → warning) on the host.
- Fusion state machine `dalbd_fusion` (SEARCH / VISUAL_TRACK / SOUND_TRACK), 24 unit tests, RViz view (`fusion_viz`).
- Latency analysis: `latency_trace` (live, per stage, reaction test) and `latency_probe.py`; MAXN halved the YOLO
  and tracker latency compared with 15 W.
- `detector_every` study and the YOLO / tracker state machine diagram (`docs`).
- Consumption sweep tool `detector_sweep` (power per rail, whole-system supply power, energy, temperatures, fan,
  per-process CPU/RAM, ranking).
- Camera sender options: `--raw`, `--skip-dup`, FFC freeze logging.
- J41 USB tests (direct camera / UMA-8 on the carrier): wiring diagram, D+/D− swap found, high-speed failure from
  cable signal quality.
- Second Jetson over Ethernet with GSM kept as default route; RTSP stream-server kit for Jetson B.
- One-command run scripts (`run/`), everything renamed to DALBD, DALBD moved into the workspace (link `~/DALBD`),
  single model folder, board cleaned (home ≈ 2.8 GB → 1.6 GB).
- Sound pipeline moved into the container (separate container build of `odas_ros` + `audio_utils`, host build kept).

### Open items
- Detection quality: compare `CAM_MODE=mjpg` vs `raw`.
- Run the consumption sweep (`detector_sweep --ask-supply`) and pick `DETECTOR_EVERY`.
- Calibrate camera ↔ UMA-8 (`sound_yaw_offset_deg`, `sound_az_sign` in `fusion.yaml`).
- J41: proper cable (Auvidea adapter + 1:1 micro-coax) for direct USB.
- Jetson B: check the `/image` rate over the link.

# DracoViLoc Isaac ROS YOLO

Isaac ROS YOLO runs inside NVIDIA's Isaac ROS development container.
DracoViLoc/RViz runs on the host. The two processes communicate through ROS 2
DDS; the host launch must not source or include the container workspace.

The authoritative trained checkpoint is stored under:

```text
~/DracoViLoc/models/yolo/
```

The Isaac workspace `models/` directory is only a deployment/runtime copy.
When a new YOLO version is selected, update DracoViLoc's model directory first.

## Source repository and fresh workspace

The source repository is `/home/dhianeifar/IsaacROS` (or its remote clone).
The runtime/build workspace is `$HOME/workspaces/isaac_ros-dev`. Do not edit
generated files under `build/` or `install/` as source code.

For a fresh installation:

```bash
sudo apt update
sudo apt install -y \
  git git-lfs curl rsync \
  build-essential cmake \
  python3-rosdep python3-colcon-common-extensions

git lfs install
export ISAAC_ROS_WS=$HOME/workspaces/isaac_ros-dev
mkdir -p "$ISAAC_ROS_WS/src"

cd "$ISAAC_ROS_WS/src"
git clone --recurse-submodules https://github.com/DhiaNeifar/IsaacROS.git temp_repo
git -C temp_repo submodule update --init --recursive
rsync -a --exclude .git --exclude .gitmodules temp_repo/ ./
rm -rf temp_repo
```

The NVIDIA repositories and local packages must be directly below workspace
`src`; do not leave an additional `src/IsaacROS` directory. Always build
inside the container because host and container paths differ.

## Project-owned packages

- `isaac_ros_yolo_bringup`: preprocessing, eight-block low-memory TensorRT
  inference, YOLO decoding, and annotated visualization.
- `isaac_ros_yolo_direction`: selects the highest-confidence box and publishes
  `/yolo/directions` and `/yolo/target_marker`.
- `yolo_video_publisher`: publishes a V4L2 camera or prerecorded video.

## 1. Deploy the latest checkpoint

On the host:

```bash
cp ~/DracoViLoc/models/yolo/drone_yolo11n_20260825_best.pt \
  ~/workspaces/isaac_ros-dev/models/
```

To deploy updated project packages as well:

```bash
cd ~/DracoViLoc
./scripts/deploy_isaac_ros.sh ~/workspaces/isaac_ros-dev
```

This command overwrites the corresponding package directories in the Isaac
workspace. Before using it on an already validated installation, review the
diff and make sure any live workspace calibration changes have first been
copied back into `isaac_ros/packages/`.

The deployment overlay can contain an older generic
`drone_yolo11n_best` model. The explicitly versioned checkpoint copied from
`~/DracoViLoc/models/yolo` is the one used below.

## 2. Enter the container

```bash
cd ~/workspaces/isaac_ros-dev
./src/isaac_ros_common/scripts/run_dev.sh -b
```

Confirm the acoustic camera is visible inside it:

```bash
v4l2-ctl --list-devices
```

The validated camera device is `/dev/video0`.

## 3. Install container dependencies once

```bash
sudo apt update
sudo apt install -y \
  v4l-utils \
  ros-humble-image-tools \
  ros-humble-magic-enum
```

Use a persistent/committed container image if these packages should survive a
container recreation. Do not install generic PyPI PyTorch: it may download a
second CUDA 13/cuDNN stack, conflict with JetPack, and fill the filesystem.

## 4. YOLO export environment

The validated interpreter is:

```text
/workspaces/isaac_ros-dev/models/yolo/bin/python3
```

Validated key versions:

```text
numpy        1.24.4
pillow       10.4.0
onnx         1.16.1
ml-dtypes    0.3.2
ultralytics  8.4.123
```

Verify the existing Jetson-compatible PyTorch rather than reinstalling it:

```bash
/workspaces/isaac_ros-dev/models/yolo/bin/python3 -c \
  "import torch, ultralytics, onnx; print(torch.__version__, ultralytics.__version__, onnx.__version__)"
```

## 5. Export PT to ONNX

```bash
cd /workspaces/isaac_ros-dev

models/yolo/bin/python3 scripts/export_yolo.py \
  --model models/drone_yolo11n_20260825_best.pt \
  --imgsz 640
```

This creates:

```text
/workspaces/isaac_ros-dev/models/drone_yolo11n_20260825_best.onnx
```

The export uses static batch 1, `640x640`, and no embedded NMS. For one drone
class, the expected raw output is `[1,5,8400]`: four box values plus one class
score.

Optional verification:

```bash
models/yolo/bin/python3 scripts/verify_yolo_onnx.py \
  --model models/drone_yolo11n_20260825_best.onnx \
  --num-classes 1 \
  --imgsz 640
```

## 6. Generate the TensorRT plan

Stop other GPU-heavy processes first:

```bash
cd /workspaces/isaac_ros-dev

/usr/src/tensorrt/bin/trtexec \
  --onnx=/workspaces/isaac_ros-dev/models/drone_yolo11n_20260825_best.onnx \
  --saveEngine=/workspaces/isaac_ros-dev/models/drone_yolo11n_20260825_best.plan \
  --fp16 \
  --skipInference
```

TensorRT may remain at `Local timing cache in use` for several minutes while
profiling tactics. If `tegrastats` shows changing CPU/GPU activity, wait.

```bash
ls -lh \
  /workspaces/isaac_ros-dev/models/drone_yolo11n_20260825_best.onnx \
  /workspaces/isaac_ros-dev/models/drone_yolo11n_20260825_best.plan
```

Generate the plan before starting the integrated pipeline. Do not let the
launch rebuild it while AST, GRE, Gazebo, and RViz are using memory.

## 7. Build the overlay

```bash
cd /workspaces/isaac_ros-dev
source /opt/ros/humble/setup.bash

colcon build --symlink-install \
  --packages-up-to \
    isaac_ros_yolo_bringup \
    isaac_ros_yolo_direction \
    yolo_video_publisher \
  --cmake-args -DBUILD_TESTING=OFF

source install/setup.bash
```

## 8. Run the live camera

```bash
cd /workspaces/isaac_ros-dev
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 launch isaac_ros_yolo_direction yolo_camera.launch.py \
  camera:=/dev/video0 \
  model_path:=/workspaces/isaac_ros-dev/models/drone_yolo11n_20260825_best.onnx \
  engine_path:=/workspaces/isaac_ros-dev/models/drone_yolo11n_20260825_best.plan \
  width:=640 \
  height:=480 \
  camera_fps:=30 \
  publish_rate:=15.0 \
  direction_frame:=table_mic_link \
  use_viewer:=true
```

- `use_viewer:=true` opens the annotated image window.
- `use_viewer:=false` keeps inference and ROS output without that window.
- `direction_frame:=table_mic_link` is required for the current
  hardware-in-the-loop simulation because the physical camera remains fixed.

The publisher selects only the highest-confidence box. Its center becomes a
direction `(x,y,1)`; there is no depth estimate. The mounted-camera convention
currently inverts image X and Y. `/yolo/target_marker` uses the latest TF so
wall time from the camera does not conflict with Gazebo `/clock`.

## 9. Verify

```bash
ros2 topic hz /image
ros2 topic hz /detections_output
ros2 topic echo /yolo/directions --once
ros2 topic echo /yolo/target_marker --once
```

Outputs:

- `/detections_output`: YOLO boxes and confidence.
- `/yolov8_processed_image`: annotated image.
- `/yolo/directions`: one ray for the best box.
- `/yolo/target_marker`: blue RViz arrow.

## 10. Integrated host launch

Leave YOLO running in the container. Open a separate, clean host terminal and
run the GRE + AST + ODAS + RViz command in the root README. Do not source the
Isaac install on the host.

The current validated host mode is `tracking_mode:=direct_yolo`. EKF and the
placeholder YOLO-to-EKF adapter are disabled.

## Troubleshooting

### `v4l2-ctl` not found

```bash
sudo apt install -y v4l-utils
```

### Isaac package not found on the host

Expected: Isaac packages are container-only. Start YOLO inside the container;
do not launch it from `arm_audio_demo.launch.py`.

### `NvMapMemAlloc... error 12`

Stop duplicate ROS launches and engine builders, confirm the `.plan` already
exists, and retry the runtime alone. The launch uses `num_blocks=8` to reduce
NITROS memory.

### Engine generation looks frozen

Check `tegrastats`. If utilization changes, TensorRT is still profiling and
should be allowed to finish.

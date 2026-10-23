# drone_target_tracker

A ROS 2 package that follows **one target** with a single-object tracker (**DyHiT**, **HiT-Base** or **SUTrack**) on
every camera image, checked and corrected by the YOLO detections of `isaac_ros_yolov8`. It publishes the target's box
and its bearing (the angle to it) for the guidance layer.

It adds one node to the existing graph. The tracker runs on every image; the detector only on every 5th image
(`detector_every`), which tracker_node forwards to it, and on every image while the target is being searched for:

```
v4l2_camera ──/image──> tracker_node ──/image_for_detector (every 5th)──> image encoder -> TensorRT -> yolov8_decoder
                             ^                                                                           │
                             └──────────────────────── /detections_output ───────────────────────────────┘
                        tracker_node ──/target, /target/bearing──> guidance
```

(Full diagrams: `../ARCHITECTURE.md`.)

## What the node does

| state | what happens |
|---|---|
| SEARCH | no target. The first time, the most confident car of a detection message becomes the target (`initial_target`). |
| TRACK | the tracker runs on **every image** and its box is published. Every `detector_every` images (5) the image goes to the detector, and its detections **check** the tracker's box on that same image: **confirm** (a detection is on the box), **correct** (the tracker drifted, a detection is near: restart the tracker on it), **switch** (the tracker jumped to another car: corrected back), **unverified** (no detection near: keep the tracker's box). |
| lost | the target has not been confirmed or corrected for `lost_after_s` (0.5 s). Status becomes `searching`, no box is published, but the tracker keeps running unseen. |
| re-acquire | within `trust_tracker_s` (2 s) of the last confirmation, only a detection near the tracker's box brings the target back (the detector had missed it, the tracker probably had not): other cars are ignored. Later, the detection nearest to the last known position, within a radius that grows with time (`search_growth`). |

Detections arrive a little after their image. The node keeps the last images in a buffer and compares each detection
message with the tracker's box **on the same image**. If it has to restart the tracker, it restarts it on that image
and re-runs it over the few newer images to catch up (`max_catch_up`).

## Topics

| | topic | type | notes |
|---|---|---|---|
| in | `/image` | `sensor_msgs/Image` | rgb8, bgr8, mono8, mono16, yuv422_yuy2 (YUYV) or yuv422 (UYVY) |
| in | `/detections_output` | `vision_msgs/Detection2DArray` | class id `"0"` = car; only `target_classes` are used |
| in | `/camera_info` | `sensor_msgs/CameraInfo` | focal length and centre, for the bearing |
| out | `/target` | `vision_msgs/Detection2D` | box centre and size in pixels, image timestamp; score = SUTrack's confidence (0 for HiT / DyHiT) |
| out | `/target/bearing` | `geometry_msgs/Vector3Stamped` | x = azimuth, y = elevation, radians, right / up positive |
| out | `/target/status` | `std_msgs/String` | `tracking` or `searching`, on every image |
| out | `/target/debug_image` | `sensor_msgs/Image` | the image with the tracker's box (red) and the detections (cyan); drawn only while someone views it |
| out | `/image_for_detector` | `sensor_msgs/Image` | the images the detector should process (see "How often the detector runs") |
| out | `/camera_info_for_detector` | `sensor_msgs/CameraInfo` | their CameraInfo, same timestamps (the encoder needs both) |

All parameters, with their meaning, are in [config/tracker.yaml](config/tracker.yaml).

## Folder

```
drone_target_tracker/
  drone_target_tracker/   the node's code (installed by colcon)
    tracker_node.py         ROS 2 node: messages in and out
    core.py                 the loop without ROS: image buffer, timestamp matching, catch-up, tracker loading
    logic.py                the decisions (confirm / correct / switch / lost / re-acquire), pure Python
    image_conv.py           sensor_msgs/Image <-> BGR array, all encodings above
  config/tracker.yaml     parameters
  launch/                 tracker.launch.py (the node only), thermal_tracking.launch.py (camera + detector + node)
  docker/Dockerfile.tracker   Isaac ROS image layer with the trackers' Python packages
  tracker_lib/            the trackers: code (common/, trackers/), the authors' repos (external/), config.json;
                          not installed, read from the source folder (tracker_lib parameter); COLCON_IGNORE inside
  test/                   tests that need neither ROS nor a GPU
```

## Setup (once, on the Jetson)

**1. The container layer.** The trackers need PyTorch with torchvision and a few Python packages inside the Isaac ROS
container. They are added as an extra image layer, so they stay installed when the container is rebuilt. On the
Jetson (outside the container), create or edit `~/.isaac_ros_common-config`:

```bash
CONFIG_IMAGE_KEY="ros2_humble.tracker"
CONFIG_DOCKER_SEARCH_DIRS=($HOME/workspaces/isaac_ros-dev/src/drone_target_tracker/docker)
```

If the file already has a `CONFIG_IMAGE_KEY` (e.g. `ros2_humble.realsense`), add `.tracker` at its end instead. The
next `./src/isaac_ros_common/scripts/run_dev.sh` builds the new layer (several minutes, once) and starts the
container. Check inside it:

```bash
python3 -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())"
```

It must end with `True`. The layer replaces the image's PyTorch 2.5 (which has no torchvision) with a matching torch +
torchvision pair from the Jetson AI Lab index for JetPack 6 / CUDA 12.6.

**2. Build** (inside the container):

```bash
cd /workspaces/isaac_ros-dev
colcon build --symlink-install --packages-select drone_target_tracker
source install/setup.bash
```

**3. The trackers' model files** (inside the container, ~2.6 GB, once; they are never committed to git):

```bash
python3 src/drone_target_tracker/tracker_lib/download_models.py
```

## Run

**Recommended: everything in one command** (camera + detector + tracker; the detector runs every 5th image and its
decoder gets `num_classes: 4`):

```bash
ros2 launch drone_target_tracker thermal_tracking.launch.py                         # DyHiT, detector every 5
ros2 launch drone_target_tracker thermal_tracking.launch.py tracker:=sutrack detector_every:=10
```

**Or only the node**, next to the camera and the detector started as usual (the `doc` commands or the
`yolov8_tensor_rt.launch.py` command). That detector still reads `/image` and runs on every image, so no GPU is saved,
but the node uses only the detections of the images it forwards, so it behaves the same. In another terminal of the
container:

```bash
source install/setup.bash
ros2 launch drone_target_tracker tracker.launch.py                    # DyHiT (default)
ros2 launch drone_target_tracker tracker.launch.py tracker:=sutrack   # or hit_base
```

The node needs 10-60 s to load the tracker (longer for SUTrack the first time: CLIP). Then:

```bash
ros2 topic echo /target/status            # tracking / searching
ros2 topic echo /target --once            # the box
ros2 topic hz /target                     # should follow the camera's frame rate
ros2 run rqt_image_view rqt_image_view /target/debug_image
```

Every 5 s the node prints a summary line: images, tracker time per image (`trk_ms_mean`, `trk_ms_p95`), detection
messages, confirmed / corrected / switches / lost / reacquired, restarts.

**Without the camera:** start the node, then play a recording of `/image`, `/camera_info` and
`/detections_output` (the node matches messages by their own timestamps, so a replay works as live):

```bash
ros2 launch drone_target_tracker tracker.launch.py
ros2 bag play bags/thermal_car_01          # in another terminal
```

## How often the detector runs

`detector_every` in [config/tracker.yaml](config/tracker.yaml) (default **5**), or `detector_every:=N` on a launch
command:

- **while tracking**, every N-th image is forwarded to the detector (`/image_for_detector` + `/camera_info_for_detector`);
  with SUTrack, also an image whose score drops below `lost_score` (`check_on_low_score`);
- **while searching** (no target yet, or lost), every image is forwarded, to find the target quickly;
- detections of images that were not forwarded are ignored, so the node behaves the same whichever image topic the
  detector reads; the GPU is only saved when the detector reads the forwarded images (`thermal_tracking.launch.py`);
- `0` = the node does not drive the detector: it uses every detection message it receives.

At 30 fps, N = 5 is 6 checks per second; lost after `lost_after_s` 0.5 s = about 3 checks without the target. In the
replay below the detector ran on 208 of 900 images (77% fewer) with the same accuracy as on every image. Measure on
the Jetson whether N = 5 is the right trade-off: the tracker and the detector share the GPU.

## Tests

```bash
cd src/drone_target_tracker && python3 -m pytest test -q       # 15 tests, no ROS and no GPU needed
```

What has been checked so far, on a Windows laptop (RTX 3050 Ti), **without ROS**:
- the 15 tests: decisions, timestamp matching, catch-up, detections arriving before their image, which images go to
  the detector, all image encodings including YUYV;
- `core.py` with the three real trackers on a 900-frame drone video of a small car (UAV123 car16), with simulated
  detections arriving 2 frames late, 20% missed, a 1.5 s detector blackout and a second car nearby: 93-95% of the
  frames within 20 px of the true car (median error 1.2-1.4 px); the target is lost during the blackout and
  re-acquired on the right car when the detector sees it again; tracker time 11 ms (DyHiT), 18 ms (HiT-Base),
  24 ms (SUTrack) per image on that laptop. With the detector on every 5th image only (208 of 900 runs): 94-96%
  within 20 px, same re-acquisition.

**Not run yet:** the ROS node itself (`tracker_node.py`), the launch files and the container layer. They need ROS 2
and the Jetson. Report anything that breaks, with the full error message.

## Things to know

- **Decoder `num_classes`:** `yolov8_decoder_node` defaults to 80 classes and the workspace's
  `yolov8_tensor_rt.launch.py` does not set it; the model has 4. `thermal_tracking.launch.py` sets 4. With the
  workspace's own launch, add `'num_classes': 4` to the decoder's parameters there. The node ignores detections whose
  class id is not 0-3 either way.
- **Which car:** the first target is the most confident car of the first detection message with one. With several
  cars in view, that may not be the one you want (an operator choice could be added later).
- **Thermal images:** the three trackers were trained on colour video; on thermal images they may track less well.
  The detector's checks catch and correct this.
- **mono16** images are stretched to 8 bits per image (min..max); the detector's encoder may do it differently.
- **Licences:** this package MIT; SUTrack and HiT code and weights MIT (`tracker_lib/external/*/LICENSE`); the YOLO
  model is an Ultralytics fine-tune, AGPL-3.0.

# DALBD/run - start / check / stop the whole DALBD pipeline with a few commands

Edit only `run.conf`. On the Jetson the DALBD repo lives in `~/workspaces/isaac_ros-dev/src/DALBD`
(container: `/workspaces/isaac_ros-dev/src/DALBD`, host link: `~/DALBD`), so this folder is `DALBD/run`.
On the laptop: `~/DALBD/run`.

| Where | Command | Does |
|---|---|---|
| JETSON host | `~/DALBD/run/jetson/start_all.sh [N]` | host prep (clocks, Loopback, PulseAudio off, tegrastats log) + sound + vision (N = detector_every) |
| LAPTOP | `~/DALBD/run/laptop/start_laptop.sh` | camera sender + UMA-8 audio sender (+ tracker view window after 20 s) |
| JETSON host | `~/DALBD/run/jetson/status.sh` | rate of every key topic + power/temperature |
| JETSON host | `~/DALBD/run/jetson/stop_all.sh` (`--all` also stops tegrastats) | stops everything |
| LAPTOP | `~/DALBD/run/laptop/stop_laptop.sh` | stops the laptop side |

Order does not matter: the audio receiver and sender retry until they meet; the camera receiver retries too.

Watch the running parts (tmux, one window per program):
- sound (host): `tmux attach -t sound`
- vision (container): `docker exec -it -u admin isaac_ros_dev-aarch64-container tmux attach -t vision`
- laptop: `tmux attach -t laptop`

tmux keys: `Ctrl-b n` next window, `Ctrl-b p` previous, `Ctrl-b w` list, `Ctrl-b d` leave (programs keep running).

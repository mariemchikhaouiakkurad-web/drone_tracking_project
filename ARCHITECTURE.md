# Architecture

## Audio path
| Stage | Where | What |
|---|---|---|
| Capture | LAPTOP | `arecord` UMA-8, 8 ch, 48 kHz, S32_LE -> `nc` over Ethernet |
| Receive | JETSON | `nc -l 5005 \| aplay` into `Loopback DEV=0`; captured back on `Loopback DEV=1` |
| ODAS input | JETSON `uma-8/scripts/start.sh` | `arecord -D plughw:CARD=Loopback,DEV=1` -> `/tmp/odas_raw.fifo` |
| ODAS | `odas_server_node` (ns `/raw`) + one `odas_core_node`, `uma-8/config/uma8.cfg` | SSL -> SST (Kalman tracks) -> SSS (4 separated channels, 16 kHz) |
| Known-point drop | `nodes/direction_gate_node.py` | silences the known source's channel in `/gate/sss`, removes its track from `/gate/sst` |
| Classification | `nodes/sss_target_classifier_node.py` + `nodes/mnv2_core.py` | MobileNetV2, 1 s windows, per track |
| Localisation | `nodes/target_direction_node.py` | x y z of MobileNet-confirmed targets, RViz markers |
| Warning | `warning/drone_warning_node.py` | CLEAR / DRONE with hold time |

## Known point (direction_gate_node)
1. **Calibration** (`cal_s` 10 s, known source alone): the most active track is the known source.
   Its id, its mean 3D direction (folded to the upper hemisphere) and its cone
   (95th percentile of its points + 5°, clamped 15–30°) are saved to `uma-8/config/known_point.json`.
2. **Run**, every SST frame: a track is KNOWN if it has the known id, or if its direction is inside
   the cone (its new id is then remembered as known for its whole life). Everything else is TARGET.
3. **Pairing** SST track <-> SSS channel uses the classifier's own rule: live channels
   (rms > -100 dBFS) are matched in order with the listed tracks; when the counts differ the
   previous pairing is kept (`pairing_mismatch` in the logs).
4. KNOWN channels are zeroed in `/gate/sss`, KNOWN tracks are removed from `/gate/sst`.
   MobileNet's output is not changed: only its input.

## Run order (`uma-8/scripts/start.sh`)
Loopback stream check -> ODAS server (+ core) -> arecord into the FIFO -> `/raw/sst` check ->
direction gate (calibration or `load`) -> target_direction -> classifier -> watchdog
(every 20 s; restarts a dead node; restarts the chain after 3 failed rate checks, keeping the saved point).

## Files
| Path | Role |
|---|---|
| `uma-8/scripts/start.sh` | one command: start / `load` / `stop` + watchdog |
| `uma-8/scripts/warning.sh` | start the warning node |
| `uma-8/scripts/sss_rec.sh` | record / listen to MobileNet's input before and after the drop |
| `uma-8/config/uma8.cfg` | the ODAS config (geometry, tracking, 16 kHz separation) |
| `uma-8/config/drone_classifier.yaml` | MobileNet parameters (model path, window, threshold) |
| `uma-8/config/known_point.json` | saved known point (generated, not in git) |
| `nodes/` | gate, classifier, MobileNet core, localisation |
| `warning/` | warning node |
| `tools/sss_recorder.py`, `tools/sss_analyze.py` | recording + analysis |
| `tools/mnv2_inspect.py` | print the model's input shape and the yaml lines to match it |
| `rviz/drone_target_view.rviz` | RViz omnisphere view |
| `check.sh` | live check |
| `docs/known_source_point_drop.pdf` | why and how the point drop works |

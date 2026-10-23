# Sound pipeline — drop the known source, classify and localise the target

A UMA-8 microphone array hears two sources: a **known source** (our own sound, e.g. a phone playing a
video whose sound keeps changing) and a **target**. The known source is removed **as one point**
(its ODAS track id + its direction cone), so MobileNet only receives the target's sound and only the
target is localised (x y z) and shown in RViz.

No null, energy or frequency calibration: the only calibration is 10 s of the known source alone,
which gives its direction. Details: [`docs/known_source_point_drop.pdf`](docs/known_source_point_drop.pdf),
[`ARCHITECTURE.md`](ARCHITECTURE.md), [`TOPICS.md`](TOPICS.md).

```
UMA-8 (USB) -> LAPTOP 192.168.10.2 (arecord) -> Ethernet -> JETSON 192.168.10.1 Loopback card
-> ODAS (uma-8/config/uma8.cfg)                 /raw/sst tracks, /raw/sss 4 separated channels (16 kHz)
-> nodes/direction_gate_node.py                 known point = track id + direction cone (10 s calibration)
     /gate/sss  known channel silenced  -> nodes/sss_target_classifier_node.py (MobileNetV2) -> /classifier/tracks
     /gate/sst  known track removed     -> nodes/target_direction_node.py -> /target/direction, /target/markers
-> warning/drone_warning_node.py                /drone_warning  CLEAR | DRONE id x y z
```

## Requirements
| Machine | Needs |
|---|---|
| LAPTOP | UMA-8 v2 (RAW firmware, 8 ch 48 kHz) on USB, `alsa-utils`, `netcat`; ROS 2 Humble + RViz2 to view |
| JETSON | ROS 2 Humble, `~/ros2_ws` with `odas_ros` (+ msgs), `snd-aloop` (Loopback card), Python 3 + numpy, the MobileNet engine set in `uma-8/config/drone_classifier.yaml` (`model_path`) |
| Link | Ethernet, laptop `192.168.10.2` <-> Jetson `192.168.10.1` |

Get the code (JETSON): `git clone -b pipeline_sound https://github.com/mariemchikhaouiakkurad-web/DALBD.git ~/known_source_drop`

## Run
| T | Machine | Command |
|---|---|---|
| T0 | LAPTOP | `ssh -t akkurad@192.168.10.1 "sudo date -s '$(date -u '+%Y-%m-%d %H:%M:%S') UTC'"` (same clock) |
| T1 | JETSON | `sudo modprobe snd-aloop; nc -l 5005 \| aplay -D hw:CARD=Loopback,DEV=0 -c 8 -r 48000 -f S32_LE -t raw -B 1000000 -F 50000 -R 500000` |
| T2 | LAPTOP | `arecord -D plughw:CARD=SPK,DEV=0 -c 8 -r 48000 -f S32_LE -t raw \| nc -N 192.168.10.1 5005` |
| T3 | JETSON | `cd ~/known_source_drop && uma-8/scripts/start.sh` — **play ONLY the known source during the 10 s calibration**, then keep it on |
| T4 | JETSON | `tail -F ~/dataset/odas_logs/classifier.log` (MobileNet per track) |
| T5 | JETSON | `tail -F ~/dataset/odas_logs/target_direction.log` (DRONE id x y z) |
| T6 | JETSON | `cd ~/known_source_drop && uma-8/scripts/warning.sh` (CLEAR / DRONE) |
| T7 | LAPTOP | `scp akkurad@192.168.10.1:known_source_drop/rviz/drone_target_view.rviz ~/ && ROS_DOMAIN_ID=42 ROS_LOCALHOST_ONLY=0 rviz2 -d ~/drone_target_view.rviz` |
| T8 | JETSON | `cd ~/known_source_drop && ./check.sh` (everything up, one publisher per topic) |

`uma-8/scripts/start.sh load` reuses the saved known point, `uma-8/scripts/start.sh stop` stops all,
`CAL_S=15 uma-8/scripts/start.sh` calibrates longer. Recalibrate whenever the known source moves.

## Expected
| Situation | MobileNet (T4) | warning (T6) |
|---|---|---|
| known source alone | `no active track` / `no drone` | `[CLEAR] no drone` |
| nothing / background | `no drone` | `[CLEAR] no drone` |
| target drone sound, >= 60° from the known source, >= -65 dBFS | `DRONE` | `[DRONE] id N p .. x y z` |

RViz: translucent upper dome, red cone = known point (dropped), green = DRONE target, yellow = other target.

## Record what MobileNet hears (before / after the drop)
| T | Machine | Command |
|---|---|---|
| R1 | LAPTOP | `nc -l 6000 \| aplay -t raw -f S16_LE -r 16000 -c 2` (left = before, right = after) |
| R2 | JETSON | `cd ~/known_source_drop && uma-8/scripts/sss_rec.sh listen both` (Ctrl+C saves) |
| R3 | JETSON | `python3 tools/sss_analyze.py` (report + spectrogram PNG of the last take) |

Files in `~/dataset/sss_rec/`: `sss_<date>_before_4ch.wav` (MobileNet's input, 4 channels unmixed),
`_after_4ch.wav`, `_after_mix.wav`, `_tracks.csv`.

## Self-tests (no ROS needed)
```bash
python3 nodes/direction_gate_node.py --selftest
python3 nodes/target_direction_node.py --selftest
python3 warning/drone_warning_node.py --selftest
python3 tools/sss_analyze.py --selftest
```

## Limits
- Reflections of the known source arrive from other directions and are not part of the point.
- A target inside the known cone (15–30°) is dropped too.
- The planar UMA-8 cannot tell up from down: directions are folded to the upper hemisphere.

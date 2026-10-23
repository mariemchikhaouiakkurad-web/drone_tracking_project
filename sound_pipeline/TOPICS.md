# Topics

| Topic | Type | Publisher | Subscribers |
|---|---|---|---|
| `/raw/sst` | `odas_ros_msgs/OdasSstArrayStamped` | odas_server_node (ns `/raw`) | direction_gate_node, sss_recorder |
| `/raw/ssl` | `odas_ros_msgs/OdasSslArrayStamped` | odas_server_node | — |
| `/raw/sss` | `audio_utils_msgs/AudioFrame` (4 ch, 16 kHz) | odas_server_node | direction_gate_node, sss_recorder |
| `/gate/sst` | `odas_ros_msgs/OdasSstArrayStamped` (known tracks removed) | direction_gate_node | sss_target_classifier_node, target_direction_node |
| `/gate/sss` | `audio_utils_msgs/AudioFrame` (known channels silenced) | direction_gate_node | sss_target_classifier_node |
| `/gate/known` | `std_msgs/String` JSON `{azimuth_deg, elevation_deg, cone_deg, id, phase}` | direction_gate_node | target_direction_node, sss_recorder |
| `/gate/status` | `std_msgs/String` JSON per channel `{slot, id, az, el, act, role}` | direction_gate_node | sss_recorder |
| `/classifier/tracks` | `std_msgs/String` JSON per track `{id, prob, is_drone, rms_dbfs}` | sss_target_classifier_node | target_direction_node |
| `/fusion/drone_targets` | `std_msgs/String` JSON | sss_target_classifier_node | — |
| `/target/direction` | `std_msgs/String` JSON `{drone, targets: [{id, drone_prob, x, y, z, azimuth_deg, elevation_deg}], known_source_dropped}` | target_direction_node | drone_warning_node |
| `/target/markers` | `visualization_msgs/MarkerArray` (frame `odas`, 25 Hz) | target_direction_node | RViz (Reliable) |
| `/drone_warning` | `std_msgs/String` JSON `{level, level_name, id, prob, x, y, z, azimuth_deg, elevation_deg}` | drone_warning_node | — |

Roles in `/gate/status`: `CAL` (calibration, all channels blocked), `KNOWN` (dropped), `TARGET` (passed to MobileNet), `EMPTY`.

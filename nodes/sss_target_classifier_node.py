#!/usr/bin/env python3
"""sss_target_classifier_node.py - drone / no-drone for each TARGET of the known-source drop (dual path).

The nulled ODAS path (known source removed) separates each tracked source into its own channel (/nulled/sss).
This node classifies every active channel with MobileNetV2, keeps a per-track decision (N of M windows),
and labels the targets of the fusion node:

  /nulled/sss  (audio_utils_msgs/AudioFrame, one channel per ODAS slot) ─┐
  /nulled/sst  (odas_ros_msgs/OdasSstArrayStamped, active tracks)  ──────┼─> [this node]
  /fusion/targets (std_msgs/String JSON, from fusion_node.py)  ──────────┘
     -> /fusion/targets_classified  every target + drone_prob, is_drone, class ("drone"/"other"/"unknown")
     -> /fusion/drone_targets       only the targets classified as drone
     -> /classifier/tracks          JSON per nulled track: id, slot, az, el, prob, is_drone
     -> /classifier/markers         RViz text "DRONE 0.93" at each track (frame odas)
  + console line per decision and ~/dataset/sss_targets_<date_time>.csv

Slot <-> track: odas_server_node drops empty slots (id 0) from /sst but /sss keeps all slots, so the active
tracks of /sst are paired, in order, with the SSS channels that carry audio.

Target <-> track: by id (fusion target id == nulled ODAS id); if ids differ and there is one target and one
classified track, they are paired. Blind-zone targets (no separated audio) stay "unknown".

Run: bash uma-8/scripts/sss_targets.sh     (needs tools/enable_nulled_sss.py applied to the nulled cfg)
"""
import csv
import json
import math
import os
import sys
import threading
import time
from datetime import datetime

import numpy as np
import rclpy
import yaml
from rclpy.node import Node
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from mnv2_core import ClassifierConfig, load_model  # noqa: E402
from sss_classify import SourceClassifier  # noqa: E402

from audio_utils_msgs.msg import AudioFrame  # noqa: E402
from odas_ros_msgs.msg import OdasSstArrayStamped  # noqa: E402

DTYPES = {"signed_16": (np.int16, 32768.0), "signed_32": (np.int32, 2147483648.0), "signed_8": (np.int8, 128.0)}


class SssTargetClassifier(Node):
    def __init__(self):
        super().__init__("sss_target_classifier")
        params = self.declare_parameter("params", os.path.join(ROOT, "uma-8", "config", "drone_classifier.yaml")).value
        model_path = self.declare_parameter("model_path", "").value
        self.sss_topic = self.declare_parameter("sss_topic", "/nulled/sss").value
        self.sst_topic = self.declare_parameter("sst_topic", "/nulled/sst").value
        self.min_activity = float(self.declare_parameter("min_activity", 0.1).value)
        self.silent_dbfs = float(self.declare_parameter("silent_dbfs", -100.0).value)
        self.max_age_s = float(self.declare_parameter("max_age_s", 1.5).value)
        self.frame_id = self.declare_parameter("frame", "odas").value
        log_dir = os.path.expanduser(self.declare_parameter("log_dir", "~/dataset").value)

        d = yaml.safe_load(open(params))
        d = d.get("drone_classifier", d).get("ros__parameters", d)
        if d.get("resize") == [0]:
            d["resize"] = []
        if model_path:
            d["model_path"] = model_path
        self.cfg = ClassifierConfig.from_dict(d)
        self.model = load_model(self.cfg)

        os.makedirs(log_dir, exist_ok=True)
        self.csv_path = os.path.join(log_dir, f"sss_targets_{datetime.now():%Y%m%d_%H%M%S}.csv")
        self.csv_f = open(self.csv_path, "w", newline="")
        self.csv = csv.writer(self.csv_f)
        self.csv.writerow(["t_s", "slot", "track_id", "az_deg", "el_deg", "activity", "rms_dbfs", "prob",
                           "is_drone", "ms"])

        self.clf = None                     # created on the first AudioFrame (sample rate / channels known)
        self.sources = []                   # latest /nulled/sst active tracks (dicts, slot order)
        self.slot_tracks = None             # per-slot track dict used by the classifier
        self.lock = threading.Lock()
        self.n_frames = 0
        self.mismatch = 0
        self.warned_ids = False

        self.pub_cls = self.create_publisher(String, "/fusion/targets_classified", 10)
        self.pub_drone = self.create_publisher(String, "/fusion/drone_targets", 10)
        self.pub_tracks = self.create_publisher(String, "/classifier/tracks", 10)
        self.pub_mk = self.create_publisher(MarkerArray, "/classifier/markers", 10)
        self.create_subscription(OdasSstArrayStamped, self.sst_topic, self.on_sst, 10)
        self.create_subscription(AudioFrame, self.sss_topic, self.on_sss, 50)
        self.create_subscription(String, "/fusion/targets", self.on_targets, 10)
        self.create_timer(0.5, self.publish_tracks)
        self.create_timer(5.0, self.summary)
        self.get_logger().info(f"model {self.cfg.model_path} ({self.model.name}) | {self.sss_topic} + {self.sst_topic} "
                               f"-> /fusion/targets_classified | csv {self.csv_path}")

    # ---------------------------------------------------------------- inputs
    def on_sst(self, msg):
        src = [{"id": int(s.id), "x": float(s.x), "y": float(s.y), "z": float(s.z), "activity": float(s.activity)}
               for s in msg.sources]
        with self.lock:
            self.sources = src

    def on_sss(self, msg):
        dt, scale = DTYPES.get(msg.format, (None, None))
        if dt is None:
            self.get_logger().error(f"unsupported SSS format {msg.format}")
            return
        n_ch = int(msg.channel_count)
        x = np.frombuffer(bytes(msg.data), dt)
        n = x.size // n_ch
        if n == 0:
            return
        x = x[: n * n_ch].reshape(n, n_ch).astype(np.float32) / scale
        if self.clf is None:
            self.clf = SourceClassifier(self.cfg, self.model, int(msg.sampling_frequency), n_ch,
                                        self.min_activity, -200.0, self.csv)
            self.get_logger().info(f"SSS stream: {n_ch} slots, {msg.sampling_frequency} Hz, {msg.format}, "
                                   f"{msg.frame_sample_count} samples/frame")

        # pair active /sst tracks (slot order, empty slots dropped) with the SSS channels that carry audio
        rms = 20 * np.log10(np.sqrt(np.mean(x ** 2, axis=0)) + 1e-12)
        live = [k for k in range(n_ch) if rms[k] > self.silent_dbfs]
        with self.lock:
            src = list(self.sources)
        if len(live) == len(src):
            slots = [{"id": 0, "activity": 0.0} for _ in range(n_ch)]
            for k, s in zip(live, src):
                slots[k] = s
            self.slot_tracks = slots
        else:
            self.mismatch += 1                # track birth/death in progress: keep the previous pairing
            if self.slot_tracks is None:
                self.slot_tracks = [{"id": 0, "activity": 0.0} for _ in range(n_ch)]
        self.clf.set_tracks(self.slot_tracks)
        self.clf.push(x)
        self.n_frames += 1

    # ---------------------------------------------------------------- outputs
    def fresh_results(self):
        if self.clf is None:
            return {}
        now = time.time()
        active = {s["id"] for s in (self.slot_tracks or []) if s.get("id")}
        return {tid: r for tid, r in self.clf.results.items() if tid in active and now - r["t"] <= self.max_age_s}

    def on_targets(self, msg):
        try:
            targets = json.loads(msg.data)
        except ValueError:
            return
        if not isinstance(targets, list):
            return
        res = self.fresh_results()
        out = []
        for tg in targets:
            tg = dict(tg)
            r = None
            if tg.get("source", "nulled") != "blindzone":
                r = res.get(int(tg.get("id", -1))) if str(tg.get("id", "")).lstrip("-").isdigit() else None
                if r is None and len(targets) == 1 and len(res) == 1:
                    r = next(iter(res.values()))
                    if not self.warned_ids:
                        self.get_logger().warn("fusion target id != nulled ODAS id: paired the single target "
                                               "with the single classified track")
                        self.warned_ids = True
            tg["drone_prob"] = r["prob"] if r else None
            tg["is_drone"] = bool(r["is_drone"]) if r else False
            tg["class"] = ("drone" if r["is_drone"] else "other") if r else "unknown"
            tg["sss_slot"] = r["slot"] if r else None
            out.append(tg)
        self.pub_cls.publish(String(data=json.dumps(out)))
        self.pub_drone.publish(String(data=json.dumps([t for t in out if t["is_drone"]])))

    def publish_tracks(self):
        res = self.fresh_results()
        self.pub_tracks.publish(String(data=json.dumps(
            [{k: r[k] for k in ("id", "slot", "az", "el", "prob", "is_drone", "rms_dbfs")} for r in res.values()])))
        ma = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        ma.markers.append(clear)
        for i, r in enumerate(res.values()):
            n = math.sqrt(r["x"] ** 2 + r["y"] ** 2 + r["z"] ** 2) or 1.0
            m = Marker()
            m.header.frame_id = self.frame_id
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns, m.id, m.type, m.action = "drone_class", i + 1, Marker.TEXT_VIEW_FACING, Marker.ADD
            m.pose.position.x, m.pose.position.y, m.pose.position.z = (0.7 * r["x"] / n, 0.7 * r["y"] / n,
                                                                       0.7 * r["z"] / n + 0.05)
            m.pose.orientation.w = 1.0
            m.scale.z = 0.07
            m.color.a = 1.0
            m.color.r, m.color.g, m.color.b = (0.1, 0.9, 0.1) if r["is_drone"] else (0.75, 0.75, 0.75)
            m.text = f"{'DRONE' if r['is_drone'] else 'no drone'} {r['prob']:.2f} (id {r['id']})"
            ma.markers.append(m)
        self.pub_mk.publish(ma)

    def summary(self):
        res = self.fresh_results()
        lab = ", ".join(f"id {t}: {'DRONE' if r['is_drone'] else 'no drone'} {r['prob']:.2f}" for t, r in res.items())
        self.get_logger().info(f"frames={self.n_frames} pairing_mismatch={self.mismatch} | {lab or 'no active track'}")
        self.csv_f.flush()

    def destroy_node(self):
        try:
            self.csv_f.close()
        except Exception:
            pass
        if hasattr(self.model, "close"):
            self.model.close()
        super().destroy_node()


def main():
    rclpy.init()
    node = SssTargetClassifier()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()

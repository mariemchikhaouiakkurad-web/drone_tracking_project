#!/usr/bin/env python3
"""state_machine_node.py - ROS 2 fusion node of DALBD (State_machine_drone_pipeline).

Vision always wins. While the tracker follows the target, the output is the vision bearing and the sound is ignored.
When vision loses the target, the sound direction ([DRONE] label) is used so that vision can find it again.

The two pipelines live in different ROS domains (vision 0 in the Isaac ROS container, sound 42 on the host).
This node joins BOTH domains (two rclpy contexts), so neither pipeline has to change.

Inputs   vision domain  /target/status    std_msgs/String               "tracking" | "searching"
                        /target/bearing   geometry_msgs/Vector3Stamped  x = az, y = el (rad, camera frame)
         sound domain   /drone_warning    std_msgs/String JSON          {level_name: CLEAR | DRONE, x, y, z,
                                                                        azimuth_deg, elevation_deg, ...}
                        /target/direction (optional, JSON {drone, targets: [...]}: best drone_prob target)
Outputs  both domains   /fused/bearing    geometry_msgs/Vector3Stamped  x = az, y = el (rad, camera frame),
                                                                        z = valid (0 none, 1 vision, 2 sound)
                        /fused/state      std_msgs/String               JSON {state, valid, az_deg, el_deg, vision,
                                                                              sound, target_lost, ...}
         vision domain  /fused/search_cue geometry_msgs/Vector3Stamped  sound direction while SOUND_TRACK (rad)
Log      <log_dir>/fusion_<date>.csv (one row per step)

Domains: parameters vision_domain / sound_domain (-p vision_domain:=0) or env FUSION_VISION_DOMAIN /
FUSION_SOUND_DOMAIN. Defaults 0 and 42.
"""
import csv
import json
import math
import os
import sys
import threading
import time

from dalbd_fusion.sound_parsing import (array_to_camera, extract_azel, parse_axes, parse_warning,
                                             warning_xyz, xyz_to_camera)
from dalbd_fusion.state_machine import SOUND, VALID_NAME, StateMachine


def _arg(name, default):
    for a in sys.argv:
        if a.startswith(name + ":="):
            return a.split(":=", 1)[1]
    return default


def main(args=None):
    import rclpy
    from geometry_msgs.msg import Vector3Stamped
    from rclpy.context import Context
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    from rosidl_runtime_py.utilities import get_message
    from std_msgs.msg import String

    vdom = int(_arg("vision_domain", os.environ.get("FUSION_VISION_DOMAIN", "0")))
    sdom = int(_arg("sound_domain", os.environ.get("FUSION_SOUND_DOMAIN", "42")))

    ctx_v, ctx_s = Context(), Context()
    rclpy.init(args=sys.argv, context=ctx_v, domain_id=vdom)
    dual = sdom != vdom
    if dual:
        rclpy.init(args=sys.argv, context=ctx_s, domain_id=sdom)
    else:
        ctx_s = ctx_v

    qos_out = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=10, reliability=ReliabilityPolicy.RELIABLE)
    qos_in = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

    # ------------------------------------------------------------------ vision side (owns the chart)
    class FusionNode(Node):
        def __init__(self):
            super().__init__("state_machine", context=ctx_v)
            p = self.declare_parameter
            p("vision_domain", vdom)
            p("sound_domain", sdom)
            self.cfg = {
                "lost_s": float(p("lost_s", 0.5).value),
                "sound_timeout_s": float(p("sound_timeout_s", 2.0).value),
                "search_timeout_s": float(p("search_timeout_s", 5.0).value),
                "vision_timeout_s": float(p("vision_timeout_s", 0.6).value),
                "require_drone_label": bool(p("require_drone_label", True).value),
                "sound_requires_loss": bool(p("sound_requires_loss", True).value),
                "use_vision_status": bool(p("use_vision_status", True).value),
                "rate_hz": float(p("rate_hz", 20.0).value),
                "sound_az_sign": float(p("sound_az_sign", -1.0).value),
                "sound_yaw_offset_deg": float(p("sound_yaw_offset_deg", 0.0).value),
                "sound_pitch_offset_deg": float(p("sound_pitch_offset_deg", 0.0).value),
                "sound_axes": p("sound_axes", "").value,
                "vision_status_topic": p("vision_status_topic", "/target/status").value,
                "vision_bearing_topic": p("vision_bearing_topic", "/target/bearing").value,
                "sound_warning_topic": p("sound_warning_topic", "/drone_warning").value,
                "sound_direction_topic": p("sound_direction_topic", "").value,
                "log_dir": os.path.expanduser(p("log_dir", "/workspaces/isaac_ros-dev/fusion_logs").value),
            }
            c = self.cfg
            self.axes = parse_axes(c["sound_axes"])       # None -> old flat-array az/el + sign/offset method
            self.sm = StateMachine(c["lost_s"], c["sound_timeout_s"], c["search_timeout_s"], c["vision_timeout_s"],
                                   c["require_drone_label"], c["sound_requires_loss"],
                                   c["use_vision_status"])
            self.lock = threading.Lock()
            self.peer = None                      # sound-side node (publishes the outputs in domain 42)
            self.create_subscription(String, c["vision_status_topic"], self.on_status, qos_in)
            self.create_subscription(Vector3Stamped, c["vision_bearing_topic"], self.on_bearing, qos_in)
            self.pub_bearing = self.create_publisher(Vector3Stamped, "/fused/bearing", qos_out)
            self.pub_state = self.create_publisher(String, "/fused/state", qos_out)
            self.pub_cue = self.create_publisher(Vector3Stamped, "/fused/search_cue", qos_out)
            self.create_timer(1.0 / c["rate_hz"], self.tick)
            self.create_timer(5.0, self.report)
            os.makedirs(c["log_dir"], exist_ok=True)
            self.csv_path = os.path.join(c["log_dir"], time.strftime("fusion_%Y%m%d_%H%M%S.csv"))
            self.csv_f = open(self.csv_path, "w", newline="")
            self.csv_w = csv.writer(self.csv_f)
            self.csv_w.writerow(["t", "state", "valid", "az_deg", "el_deg", "vision", "sound", "target_lost",
                                 "vision_az", "vision_el", "sound_az", "sound_el"])
            self.n_valid = {k: 0 for k in VALID_NAME}
            self.n_status_msgs = 0
            self.last_status = "-"
            self.n_sound_msgs = 0
            self.n_vision_msgs = 0
            self.get_logger().info(
                f"vision domain {vdom} <-> sound domain {sdom} | lost_s {c['lost_s']} vision_timeout "
                f"{c['vision_timeout_s']} use_status {c['use_vision_status']} | sound_timeout "
                f"{c['sound_timeout_s']} search_timeout {c['search_timeout_s']} | sound needs loss "
                f"{c['sound_requires_loss']} | axes '{c['sound_axes'] or 'flat'}' az sign {c['sound_az_sign']} "
                f"yaw {c['sound_yaw_offset_deg']} "
                f"pitch {c['sound_pitch_offset_deg']} deg | log {self.csv_path}")

        @staticmethod
        def now():
            return time.monotonic()

        # ---- vision inputs
        def on_status(self, msg):
            self.n_status_msgs += 1
            self.last_status = msg.data.strip()[:20]
            with self.lock:
                self.sm.on_vision_status(self.now(), msg.data.strip().lower().startswith("tracking"))

        def on_bearing(self, msg):
            if math.isnan(msg.vector.x) or math.isnan(msg.vector.y):
                return
            with self.lock:
                self.sm.on_vision_bearing(self.now(), math.degrees(msg.vector.x), math.degrees(msg.vector.y))
                self.n_vision_msgs += 1

        # ---- sound inputs (called from the sound-side thread)
        def on_sound_array(self, az_s, el_s):
            c = self.cfg
            az, el = array_to_camera(az_s, el_s, c["sound_az_sign"], c["sound_yaw_offset_deg"],
                                     c["sound_pitch_offset_deg"])
            with self.lock:
                self.sm.on_sound(self.now(), az, el)
                self.n_sound_msgs += 1

        def on_sound_xyz(self, xyz):
            """Vertical / any mounting: x y z -> camera az el with the axis remap, then the fine offsets."""
            c = self.cfg
            az, el = xyz_to_camera(xyz, self.axes)
            az = (az + c["sound_yaw_offset_deg"] + 180.0) % 360.0 - 180.0
            el = el + c["sound_pitch_offset_deg"]
            with self.lock:
                self.sm.on_sound(self.now(), az, el)
                self.n_sound_msgs += 1

        def on_drone(self, is_drone):
            with self.lock:
                self.sm.on_drone_label(self.now(), is_drone)

        # ---- one chart step
        def tick(self):
            t = self.now()
            with self.lock:
                n0 = len(self.sm.events)
                az, el, valid = self.sm.step(t)
                events = self.sm.events[n0:]
                snap = self.sm.snapshot(t)
                sm = self.sm
                row = [round(time.time(), 3), sm.state, valid, snap["az_deg"], snap["el_deg"], snap["vision"],
                       snap["sound"], sm.target_lost, _r(sm.vis_az), _r(sm.vis_el), _r(sm.snd_az), _r(sm.snd_el)]
            for (_, fr, to, why) in events:
                self.get_logger().info(f"[{fr} -> {to}] {why}")
            self.n_valid[valid] += 1
            self.csv_w.writerow(row)

            stamp = self.get_clock().now().to_msg()
            b = Vector3Stamped()
            b.header.stamp, b.header.frame_id = stamp, "camera"
            b.vector.x = math.radians(az) if not math.isnan(az) else float("nan")
            b.vector.y = math.radians(el) if not math.isnan(el) else float("nan")
            b.vector.z = float(valid)
            s = String()
            s.data = json.dumps(snap)
            self.pub_bearing.publish(b)
            self.pub_state.publish(s)
            if self.peer is not None:
                self.peer.pub_bearing.publish(b)
                self.peer.pub_state.publish(s)
            if snap["state"] == SOUND and not math.isnan(az):
                cue = Vector3Stamped()
                cue.header.stamp, cue.header.frame_id = stamp, "camera"
                cue.vector.x, cue.vector.y = math.radians(az), math.radians(el)
                self.pub_cue.publish(cue)

        def report(self):
            tot = max(sum(self.n_valid.values()), 1)
            share = " ".join(f"{VALID_NAME[k]} {100 * v / tot:.0f}%" for k, v in self.n_valid.items() if v)
            with self.lock:
                st, tl = self.sm.state, self.sm.target_lost
            self.get_logger().info(f"[STAT] state {st} | target_lost {tl} | output {share} | last 5 s: bearings "
                                   f"{self.n_vision_msgs} ({self.n_vision_msgs / 5.0:.1f} Hz), status msgs "
                                   f"{self.n_status_msgs} (last '{self.last_status}'), sound msgs {self.n_sound_msgs}")
            self.n_vision_msgs = self.n_sound_msgs = self.n_status_msgs = 0
            self.csv_f.flush()

    # ------------------------------------------------------------------ sound side (domain 42)
    class SoundSide(Node):
        def __init__(self, fusion):
            super().__init__("state_machine_sound", context=ctx_s)
            self.fusion = fusion
            self.pub_bearing = self.create_publisher(Vector3Stamped, "/fused/bearing", qos_out)
            self.pub_state = self.create_publisher(String, "/fused/state", qos_out)
            self.create_subscription(String, fusion.cfg["sound_warning_topic"], self.on_warning, qos_in)
            self.dir_sub = None
            self.waiting_logged = False
            self.create_timer(1.0, self.try_direction_topic)

        def on_warning(self, msg):
            is_drone, azel = parse_warning(msg.data)
            self.fusion.on_drone(is_drone)
            if not is_drone or self.dir_sub:
                return
            if self.fusion.axes is not None:
                xyz = warning_xyz(msg.data)
                if xyz is not None:
                    self.fusion.on_sound_xyz(xyz)      # vertical mounting: proper axis remap
            elif azel is not None:
                self.fusion.on_sound_array(*azel)      # flat mounting: az/el + sign/offset

        def try_direction_topic(self):
            if self.dir_sub is not None:
                return
            topic = self.fusion.cfg["sound_direction_topic"]
            if not topic:
                if not self.waiting_logged:
                    self.get_logger().info(f"sound label + direction from {self.fusion.cfg['sound_warning_topic']} "
                                           f"(domain {sdom})")
                    self.waiting_logged = True
                self.dir_sub = False
                return
            types = dict(self.get_topic_names_and_types())
            if topic not in types:
                if not self.waiting_logged:
                    self.get_logger().info(f"{topic} not published in domain {sdom} yet: using the direction "
                                           f"inside {self.fusion.cfg['sound_warning_topic']}")
                    self.waiting_logged = True
                return
            tname = types[topic][0]
            try:
                mtype = get_message(tname)
            except (AttributeError, ModuleNotFoundError, ValueError) as e:
                self.get_logger().warn(f"{topic}: type {tname} unknown here ({e}); keeping /drone_warning x y z")
                self.dir_sub = False
                return

            def cb(msg):
                r = extract_azel(msg)
                if r is not None:
                    self.fusion.on_sound_array(*r)
            self.dir_sub = self.create_subscription(mtype, topic, cb, qos_in)
            self.get_logger().info(f"sound direction from {topic} [{tname}] (domain {sdom})")

    fusion = FusionNode()
    sound = SoundSide(fusion)
    if dual:
        fusion.peer = sound

    ex_v = SingleThreadedExecutor(context=ctx_v)
    ex_v.add_node(fusion)
    ex_s = th_s = None
    if dual:
        ex_s = SingleThreadedExecutor(context=ctx_s)
        ex_s.add_node(sound)
        th_s = threading.Thread(target=ex_s.spin, daemon=True)
        th_s.start()
    else:
        ex_v.add_node(sound)
    try:
        ex_v.spin()
    except KeyboardInterrupt:
        pass
    finally:
        fusion.csv_f.close()
        print(f"\n[DONE] {len(fusion.sm.events)} transitions | log {fusion.csv_path}", flush=True)
        for (_, fr, to, why) in fusion.sm.events[-20:]:
            print(f"  {str(fr):>12} -> {to:<12} {why}", flush=True)
        # stop the sound-domain thread cleanly before the contexts go away (avoids "terminate called ...")
        for ex in (ex_s, ex_v):
            try:
                if ex is not None:
                    ex.shutdown(timeout_sec=1.0)
            except Exception:
                pass
        if th_s is not None:
            th_s.join(timeout=2.0)
        for n in (sound, fusion):
            try:
                n.destroy_node()
            except Exception:
                pass
        for ctx in {ctx_v, ctx_s}:
            try:
                if ctx.ok():
                    rclpy.shutdown(context=ctx)
            except Exception:
                pass


def _r(x):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(x, 3)


if __name__ == "__main__":
    main()

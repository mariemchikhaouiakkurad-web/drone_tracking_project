#!/usr/bin/env python3
"""fake_sound_node.py - plays a scripted "drone heard at azimuth X" sequence on the SOUND side (domain 42), in the
same JSON format as DALBD's drone_warning_node on /drone_warning:
  {"level": 0, "level_name": "CLEAR", "stamp": .., "input_ok": true}
  {"level": 1, "level_name": "DRONE", "stamp": .., "input_ok": true, "id": 1, "prob": 0.93, "x": .., "y": .., "z": ..,
   "azimuth_deg": .., "elevation_deg": ..}
Used to test state_machine_node without a drone, a scooter or the UMA-8.

The schedule is given in CAMERA angles (what the fusion node should output), converted to the array frame with the same
convention as the fusion node defaults (sound_az_sign = -1, offsets 0):  az_array = -az_camera.

  schedule  "0-20:10/5, 20-30:off, 30-50:-25/0"   = from 0 to 20 s a drone at az 10 deg, el 5 deg (camera frame),
                                                   20-30 s nothing, 30-50 s a drone at az -25 deg, el 0
  noise_deg 2.0     random noise on the direction (the UMA-8 is not exact)
  sync_file /tmp/scenario_start   wait until video_test_node.py starts playing, then t = 0 (same scenario clock)
            ""                    start immediately

Run (inside the container, sound domain):
  ROS_DOMAIN_ID=42 ros2 run dalbd_fusion fake_sound --ros-args -p schedule:="0-20:10/5,20-30:off,30-60:10/5"
"""
import json
import math
import os
import random
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


def parse_schedule(text):
    out = []
    for part in text.replace(" ", "").split(","):
        if not part:
            continue
        span, what = part.split(":", 1)
        a, b = (float(x) for x in span.split("-", 1))
        if what.lower() == "off":
            out.append((a, b, None))
        else:
            az, el = (what.split("/") + ["0"])[:2]
            out.append((a, b, (float(az), float(el))))
    return out


def clear():
    return json.dumps({"level": 0, "level_name": "CLEAR", "stamp": time.time(), "input_ok": True})


class FakeSound(Node):
    def __init__(self):
        super().__init__("fake_sound")
        p = self.declare_parameter
        self.sched = parse_schedule(p("schedule", "0-20:10/5,20-30:off,30-60:10/5").value)
        self.noise = float(p("noise_deg", 2.0).value)
        self.sync = p("sync_file", "/tmp/scenario_start").value
        self.pub = self.create_publisher(String, "/drone_warning", 10)
        self.t_launch = time.time()
        self.t0 = None
        self.phase = "wait"
        self.create_timer(0.1, self.tick)
        self.get_logger().info(f"domain {os.environ.get('ROS_DOMAIN_ID', '0')} | schedule {self.sched} | waiting for "
                               f"{self.sync} (start video_test_node.py) ...")

    def tick(self):
        if self.t0 is None and not self.sync:
            self.t0 = time.time()                      # sync_file "" -> start now
            self.get_logger().info("scenario started now (no sync file): t = 0")
        if self.t0 is None:
            try:
                ts = float(open(self.sync).read().strip())
                if ts >= self.t_launch - 5.0:
                    self.t0 = ts
                    self.get_logger().info("scenario started: t = 0")
            except (OSError, ValueError):
                pass
            if self.t0 is None:
                return
        t = time.time() - self.t0
        cur = next((s for s in self.sched if s[0] <= t < s[1]), "end")
        if cur == "end":
            if self.phase != "end":
                self.phase = "end"
                self.get_logger().info(f"[PHASE] t = {t:5.1f} s  schedule finished: [CLEAR]")
            self.pub.publish(String(data=clear()))
            return
        a, b, d = cur
        name = "off" if d is None else f"drone az {d[0]:+.0f} el {d[1]:+.0f}"
        if name != self.phase:
            self.phase = name
            self.get_logger().info(f"[PHASE] t = {t:5.1f} s  sound: {name}")
        if d is None:
            self.pub.publish(String(data=clear()))
            return
        az_cam = d[0] + random.gauss(0, self.noise)
        el = d[1] + random.gauss(0, self.noise / 2)
        az_arr = math.radians(-az_cam)                       # fusion default sound_az_sign = -1
        x = math.cos(math.radians(el)) * math.cos(az_arr)
        y = math.cos(math.radians(el)) * math.sin(az_arr)
        z = math.sin(math.radians(el))
        self.pub.publish(String(data=json.dumps({
            "level": 1, "level_name": "DRONE", "stamp": time.time(), "input_ok": True, "id": 1, "prob": 0.93,
            "x": round(x, 4), "y": round(y, 4), "z": round(z, 4),
            "azimuth_deg": round(math.degrees(az_arr), 1), "elevation_deg": round(el, 1)})))


def main(args=None):
    rclpy.init(args=args)
    n = FakeSound()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        n.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

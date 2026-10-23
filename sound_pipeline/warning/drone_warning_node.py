#!/usr/bin/env python3
"""ROS 2 node: drone warning at the end of the sound pipeline.

Input
  /target/direction  std_msgs/String JSON from target_direction_node
                     {"drone": bool, "targets": [{id, drone_prob, x, y, z, azimuth_deg, elevation_deg, rms_dbfs}],
                      "known_source_dropped": {...}}
                     Only tracks that passed the known-point drop AND that MobileNet confirmed are in "targets".
Output
  /drone_warning     std_msgs/String JSON {level, level_name, id, prob, x, y, z, azimuth_deg, elevation_deg, stamp}
                     level 0 CLEAR | 1 DRONE   (published at 10 Hz)
  console            [DRONE] id 37  p 0.91  x +0.62 y -0.63 z +0.46  az -45.5 el +27.4   (every 1 s while DRONE)
                     [CLEAR] no drone                                                   (once, after hold_s)
                     [WARN] no /target/direction for 3 s                                (input lost)

    python3 warning/drone_warning_node.py [--ros-args -p hold_s:=1.5 -p min_prob:=0.5]
    python3 warning/drone_warning_node.py --selftest
"""
import json
import sys
import time

CLEAR, DRONE = 0, 1
NAMES = {CLEAR: "CLEAR", DRONE: "DRONE"}


class WarningCore:
    """CLEAR / DRONE with a hold time, so one missing frame does not flip the warning."""

    def __init__(self, hold_s=1.5, min_prob=0.5, stale_s=3.0):
        self.hold_s, self.min_prob, self.stale_s = hold_s, min_prob, stale_s
        self.best = None          # last confirmed drone target
        self.t_drone = -1e9       # last time a drone was seen
        self.t_msg = None         # last /target/direction message
        self.level = CLEAR

    def on_direction(self, obj, now):
        self.t_msg = now
        tg = [t for t in (obj.get("targets") or []) if float(t.get("drone_prob") or 0.0) >= self.min_prob]
        if obj.get("drone") and tg:
            self.best = max(tg, key=lambda t: float(t.get("drone_prob") or 0.0))
            self.t_drone = now

    def evaluate(self, now):
        """-> (state dict, changed bool)"""
        lvl = DRONE if now - self.t_drone <= self.hold_s else CLEAR
        changed = lvl != self.level
        self.level = lvl
        s = {"level": lvl, "level_name": NAMES[lvl], "stamp": time.time(),
             "input_ok": self.t_msg is not None and now - self.t_msg <= self.stale_s}
        if lvl == DRONE and self.best:
            b = self.best
            s.update({"id": b.get("id"), "prob": b.get("drone_prob"), "x": b.get("x"), "y": b.get("y"), "z": b.get("z"),
                      "azimuth_deg": b.get("azimuth_deg"), "elevation_deg": b.get("elevation_deg")})
        return s, changed


def fmt(s):
    if s["level"] == CLEAR:
        return "[CLEAR] no drone"
    f = lambda k: float(s.get(k) or 0.0)  # noqa: E731
    return (f"[DRONE] id {s.get('id')}  p {f('prob'):.2f}  x {f('x'):+.2f} y {f('y'):+.2f} z {f('z'):+.2f}  "
            f"az {f('azimuth_deg'):+6.1f} el {f('elevation_deg'):+5.1f}")


def run_node():
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String

    class DroneWarningNode(Node):
        def __init__(self):
            super().__init__("drone_warning_node")
            p = lambda k, v: self.declare_parameter(k, v).value  # noqa: E731
            self.core = WarningCore(hold_s=float(p("hold_s", 1.5)), min_prob=float(p("min_prob", 0.5)))
            self.pub = self.create_publisher(String, "/drone_warning", 10)
            self.create_subscription(String, p("direction_topic", "/target/direction"), self.on_dir, 10)
            self.create_timer(0.1, self.tick)
            self.t_print = 0.0
            self.warned = False
            self.t0 = time.monotonic()
            print("drone_warning_node: /target/direction -> /drone_warning", flush=True)

        def on_dir(self, msg):
            try:
                self.core.on_direction(json.loads(msg.data), time.monotonic())
            except (ValueError, AttributeError):
                pass

        def tick(self):
            now = time.monotonic()
            s, changed = self.core.evaluate(now)
            out = String()
            out.data = json.dumps(s, default=float)
            self.pub.publish(out)
            stamp = time.strftime("%H:%M:%S")
            if changed or (s["level"] == DRONE and now - self.t_print >= 1.0):
                print(f"{stamp} {fmt(s)}", flush=True)
                self.t_print = now
            if not s["input_ok"] and now - self.t0 > 3.0 and not self.warned:
                print(f"{stamp} [WARN] no /target/direction for 3 s (is target_direction_node running?)", flush=True)
                self.warned = True
            elif s["input_ok"]:
                self.warned = False

    rclpy.init()
    node = DroneWarningNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
            rclpy.shutdown()
        except Exception:  # noqa: BLE001
            pass


def selftest():
    c = WarningCore(hold_s=1.5)
    s, _ = c.evaluate(0.0)
    assert s["level"] == CLEAR and not s["input_ok"]
    c.on_direction({"drone": False, "targets": []}, 0.1)
    s, _ = c.evaluate(0.1)
    assert s["level"] == CLEAR and s["input_ok"]
    print("known source only / nothing            -> CLEAR ok")
    c.on_direction({"drone": True, "targets": [{"id": 37, "drone_prob": 0.91, "x": 0.62, "y": -0.63, "z": 0.46,
                                                "azimuth_deg": -45.5, "elevation_deg": 27.4}]}, 1.0)
    s, ch = c.evaluate(1.0)
    assert s["level"] == DRONE and s["id"] == 37 and ch
    print(fmt(s), "          -> DRONE ok")
    c.on_direction({"drone": False, "targets": []}, 1.5)
    s, ch = c.evaluate(2.0)
    assert s["level"] == DRONE and not ch
    print("one empty frame (hold 1.5 s)           -> still DRONE ok")
    s, ch = c.evaluate(3.0)
    assert s["level"] == CLEAR and ch
    print("no drone for 2 s                       -> CLEAR ok")
    c.on_direction({"drone": True, "targets": [{"id": 5, "drone_prob": 0.3}]}, 3.1)
    s, _ = c.evaluate(3.1)
    assert s["level"] == CLEAR
    print("drone_prob 0.3 < min_prob              -> CLEAR ok")
    print("selftest PASSED")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        run_node()

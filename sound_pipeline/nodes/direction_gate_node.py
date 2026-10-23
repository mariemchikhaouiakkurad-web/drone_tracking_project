#!/usr/bin/env python3
"""Known source dropped as ONE POINT: its direction + its ODAS track id. No null, no energy,
no frequency calibration -> works whatever the known source plays (a video that changes sound).

Works on the RAW ODAS path (no spatial null):
  in   /raw/sst  (tracks)   /raw/sss  (separated audio, slot i <-> sources[i])
  out  /gate/sst            same tracks, known-source tracks removed      -> target_direction_node
       /gate/sss            same audio, known-source channels silenced    -> MobileNet input
       /gate/known          std_msgs/String JSON {azimuth_deg, elevation_deg, cone_deg, id, phase}
       /gate/status         std_msgs/String JSON per slot {slot, id, az, el, act, role}

Calibration = only the DIRECTION of the known source (cal_s, default 10 s, known source alone):
  the track with the most activity is the known source -> its id, its direction (3D mean of
  its points), its cone (95 % of its points + margin, 10..30 deg). Saved to known_point.json.
Then, per frame, a slot is blocked (KNOWN) when
  its track id is the known id, or
  its direction is inside the cone (that new id is then remembered as known for its whole life).
Everything else passes to MobileNet untouched (TARGET).

    python3 nodes/direction_gate_node.py [--ros-args -p cal_s:=10.0 -p load:=false]
    python3 nodes/direction_gate_node.py --selftest
"""
import json
import math
import os
import sys
import time
from array import array

import numpy as np


def unit(x, y, z, fold=True):
    if fold:
        z = abs(z)                    # flat array: above / below the array plane are mirror images
    n = math.sqrt(x * x + y * y + z * z)
    return None if n < 1e-6 else (x / n, y / n, z / n)


def az_el(v):
    return math.degrees(math.atan2(v[1], v[0])), math.degrees(math.asin(max(-1.0, min(1.0, v[2]))))


def angle(u, v):
    return math.degrees(math.acos(max(-1.0, min(1.0, u[0] * v[0] + u[1] * v[1] + u[2] * v[2]))))


class GateCore:
    """No ROS. phase WAIT -> CAL (cal_s) -> RUN."""

    def __init__(self, cal_s=10.0, cal_min_act=0.3, min_act=0.1, cone_min=10.0, cone_max=30.0,
                 margin=5.0, fold=True):
        self.cal_s, self.cal_min_act, self.min_act = cal_s, cal_min_act, min_act
        self.cone_min, self.cone_max, self.margin, self.fold = cone_min, cone_max, margin, fold
        self.phase, self.t0 = "WAIT", None
        self.samples = []              # (id, unit vector, activity) during calibration
        self.known = None              # {"vec", "az", "el", "cone", "id"}
        self.known_ids = set()
        self.slots = []                # per slot (id, vec, act)
        self.roles = []

    # -- calibration
    def load(self, d):
        v = (math.cos(math.radians(d["elevation_deg"])) * math.cos(math.radians(d["azimuth_deg"])),
             math.cos(math.radians(d["elevation_deg"])) * math.sin(math.radians(d["azimuth_deg"])),
             math.sin(math.radians(d["elevation_deg"])))
        self.known = {"vec": v, "az": d["azimuth_deg"], "el": d["elevation_deg"], "cone": d["cone_deg"],
                      "id": d.get("id")}
        self.phase = "RUN"

    def _finish(self):
        if len(self.samples) < 20:
            return False                                  # not enough: keep calibrating
        w = {}
        for tid, _v, a in self.samples:
            w[tid] = w.get(tid, 0.0) + a
        kid = max(w, key=w.get)
        pts = [(v, a) for tid, v, a in self.samples if tid == kid]
        V = np.array([v for v, _ in pts])
        A = np.array([a for _, a in pts])
        m = (V * A[:, None]).sum(0)
        m = m / (np.linalg.norm(m) or 1.0)
        ang = np.degrees(np.arccos(np.clip(V @ m, -1, 1)))
        p95 = float(np.percentile(ang, 95))
        cone = float(min(self.cone_max, max(self.cone_min, p95 + self.margin)))
        az, el = az_el(m)
        self.known = {"vec": tuple(m), "az": az, "el": el, "cone": cone, "id": kid,
                      "spread_p95": p95, "frames": len(pts), "share": w[kid] / sum(w.values())}
        self.known_ids = {kid}
        self.phase = "RUN"
        return True

    def to_dict(self):
        k = self.known
        return None if k is None else {"azimuth_deg": round(k["az"], 1), "elevation_deg": round(k["el"], 1),
                                       "cone_deg": round(k["cone"], 1), "id": k["id"],
                                       "spread_p95_deg": round(k.get("spread_p95", 0.0), 1),
                                       "share": round(k.get("share", 1.0), 2), "stamp": time.time()}

    # -- per SST frame
    def on_sst(self, sources, now):
        slots = []
        for s in sources:
            g = (lambda k: s[k]) if isinstance(s, dict) else (lambda k: getattr(s, k))
            try:
                tid, act = int(g("id")), float(g("activity"))
                v = unit(float(g("x")), float(g("y")), float(g("z")), self.fold)
            except (KeyError, AttributeError, TypeError, ValueError):
                tid, act, v = 0, 0.0, None
            slots.append((tid, v, act))
        self.slots = slots
        if self.phase == "WAIT":
            self.phase, self.t0 = "CAL", now
        if self.phase == "CAL":
            for tid, v, act in slots:
                if tid and v is not None and act >= self.cal_min_act:
                    self.samples.append((tid, v, act))
            if now - self.t0 >= self.cal_s:
                self._finish()
        self.roles = [self.role(i) for i in range(len(slots))]
        return self.roles

    def role(self, i):
        tid, v, act = self.slots[i]
        if tid == 0 or v is None or act < self.min_act:
            return "EMPTY"
        if self.phase != "RUN":
            return "CAL"                                   # nothing reaches MobileNet while calibrating
        if tid in self.known_ids:
            return "KNOWN"
        if angle(v, self.known["vec"]) <= self.known["cone"]:
            self.known_ids.add(tid)                        # same point, new id: known for its whole life
            return "KNOWN"
        return "TARGET"

    def blocked(self, i):
        return i < len(self.roles) and self.roles[i] in ("CAL", "KNOWN")

    def status(self):
        out = []
        for i, (tid, v, act) in enumerate(self.slots):
            az, el = az_el(v) if v is not None else (None, None)
            out.append({"slot": i, "id": tid, "az": None if az is None else round(az, 1),
                        "el": None if el is None else round(el, 1), "act": round(act, 2),
                        "role": self.roles[i] if i < len(self.roles) else "EMPTY"})
        return out


def live_channels(raw, ch, fmt, silent_dbfs=-100.0):
    """Channels carrying audio (rms > silent_dbfs), as the classifier decides them."""
    fmt = str(fmt).lower()
    dt, sc = ("<f4", 1.0) if "float" in fmt else (("<i4", 2.0 ** 31) if "32" in fmt else ("<i2", 2.0 ** 15))
    a = np.frombuffer(raw, dt)
    if ch <= 0 or a.size % ch or a.size == 0:
        return []
    x = a.reshape(-1, ch).astype(np.float32) / sc
    rms = 20 * np.log10(np.sqrt(np.mean(x ** 2, axis=0)) + 1e-12)
    return [k for k in range(ch) if rms[k] > silent_dbfs]


def silence(raw, ch, fmt, idx):
    fmt = str(fmt).lower()
    dt = "<f4" if "float" in fmt else ("<i4" if "32" in fmt else "<i2")
    a = np.frombuffer(raw, dt)
    if ch <= 0 or a.size % ch:
        return raw
    b = a.reshape(-1, ch).copy()
    b[:, idx] = 0
    return b.tobytes()


def run_node():
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
    Sst = Audio = None
    for mod in ("odas_ros_msgs.msg", "odas_ros.msg"):
        try:
            Sst = __import__(mod, fromlist=["OdasSstArrayStamped"]).OdasSstArrayStamped
            break
        except (ImportError, AttributeError):
            pass
    try:
        from audio_utils_msgs.msg import AudioFrame as Audio
    except ImportError:
        pass
    if Sst is None or Audio is None:
        sys.exit("odas_ros_msgs / audio_utils_msgs not found: source ~/ros2_ws/install/setup.bash first")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    class GateNode(Node):
        def __init__(self):
            super().__init__("direction_gate")

            def p(name, default):
                if not self.has_parameter(name):
                    self.declare_parameter(name, default)
                return self.get_parameter(name).value

            self.core = GateCore(cal_s=float(p("cal_s", 10.0)), cal_min_act=float(p("cal_min_activity", 0.3)),
                                 min_act=float(p("min_activity", 0.1)), cone_min=float(p("cone_min_deg", 15.0)),
                                 cone_max=float(p("cone_max_deg", 30.0)), margin=float(p("margin_deg", 5.0)),
                                 fold=bool(p("upper_only", True)))
            self.file = os.path.expanduser(p("point_file", os.path.join(root, "uma-8", "config", "known_point.json")))
            if bool(p("load", False)) and os.path.exists(self.file):
                self.core.load(json.load(open(self.file)))
                print(f"[gate] known point loaded: {self.core.to_dict()}", flush=True)
            else:
                print(f"[gate] CALIBRATION {self.core.cal_s:.0f} s: play ONLY the known source "
                      f"(its direction + track id are learned, nothing else)", flush=True)
            self.saved = self.core.phase == "RUN"
            self.pub_sst = self.create_publisher(Sst, "/gate/sst", 10)
            self.pub_sss = self.create_publisher(Audio, "/gate/sss", 50)
            self.pub_known = self.create_publisher(String, "/gate/known", 10)
            self.pub_status = self.create_publisher(String, "/gate/status", 10)
            self.create_subscription(Sst, p("in_sst", "/raw/sst"), self.on_sst, 10)
            self.create_subscription(Audio, p("in_sss", "/raw/sss"), self.on_sss, 50)
            self.create_timer(0.5, self.tick)
            self.last_status = 0.0
            self.last_line = None
            self.silent_dbfs = float(p("silent_dbfs", -100.0))   # same threshold as the classifier
            self.src_n, self.src_blocked = 0, []
            self.ch_map, self.ch_block, self.mismatch = {}, [], 0

        def on_sst(self, msg):
            now = time.monotonic()
            self.core.on_sst(msg.sources, now)
            self.src_n = len(msg.sources)
            self.src_blocked = [self.core.blocked(i) for i in range(len(msg.sources))]
            if not self.saved and self.core.phase == "RUN":
                d = self.core.to_dict()
                try:
                    with open(self.file, "w") as f:
                        json.dump(d, f, indent=1)
                except OSError:
                    pass
                self.saved = True
                print(f"[gate] KNOWN POINT  id {d['id']}  az {d['azimuth_deg']:+.1f}  el {d['elevation_deg']:+.1f}  "
                      f"cone {d['cone_deg']:.1f} deg  (spread p95 {d['spread_p95_deg']:.1f}, "
                      f"{100 * d['share']:.0f} % of the activity) -> {self.file}", flush=True)
                if d["share"] < 0.6:
                    print("[gate] WARN: another sound was active during calibration (known source < 60 % of "
                          "the activity) -> recalibrate with ONLY the known source", flush=True)
            # blocked tracks are REMOVED (not just activity 0): the classifier pairs the k-th listed
            # track with the k-th channel that carries audio, and blocked channels are silenced
            keep = [s for i, s in enumerate(msg.sources) if not self.src_blocked[i]]
            msg.sources = keep
            self.pub_sst.publish(msg)
            line = " | ".join(f"id {s['id']} az {s['az']:+.0f} el {s['el']:+.0f} {s['role']}"
                              for s in self.core.status() if s["role"] != "EMPTY") or "no active track"
            key = tuple((s["id"], s["role"]) for s in self.core.status())
            if key != self.last_line:
                self.last_line = key
                print(f"[{time.strftime('%H:%M:%S')}] [gate] {self.core.phase}: {line}", flush=True)

        def on_sss(self, msg):
            ch = int(getattr(msg, "channel_count", 0) or 0)
            fmt = getattr(msg, "format", "signed_16")
            raw = bytes(msg.data)
            live = live_channels(raw, ch, fmt, self.silent_dbfs)
            if len(live) == self.src_n:
                # same pairing as the classifier: k-th listed track <-> k-th live channel
                self.ch_map = {live[k]: k for k in range(len(live))}
                self.ch_block = [live[k] for k in range(len(live)) if self.src_blocked[k]]
            else:
                self.mismatch += 1           # track birth/death in progress: keep the previous mapping
            if self.core.phase != "RUN":
                self.ch_block = list(live)   # calibrating: nothing reaches MobileNet
            idx = [c for c in self.ch_block if c < ch]
            if idx:
                msg.data = array("B", silence(raw, ch, fmt, idx))
            self.pub_sss.publish(msg)
            now = time.monotonic()
            if now - self.last_status >= 0.1:
                self.last_status = now
                st_src = self.core.status()
                slots = []
                for c in range(ch):
                    k = self.ch_map.get(c)
                    if k is not None and k < len(st_src):
                        s = dict(st_src[k]); s["slot"] = c
                    else:
                        s = {"slot": c, "id": 0, "az": None, "el": None, "act": 0.0, "role": "EMPTY"}
                    slots.append(s)
                st = String()
                st.data = json.dumps({"phase": self.core.phase, "slots": slots, "pairing_mismatch": self.mismatch})
                self.pub_status.publish(st)

        def tick(self):
            d = self.core.to_dict() or {}
            d["phase"] = self.core.phase
            m = String()
            m.data = json.dumps(d)
            self.pub_known.publish(m)

    rclpy.init()
    node = GateNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception:  # noqa: BLE001
        if rclpy.ok():
            raise
    finally:
        try:
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:  # noqa: BLE001
            pass


def _src(i, az, el, act=0.9):
    a, e = math.radians(az), math.radians(el)
    return {"id": i, "x": math.cos(e) * math.cos(a), "y": math.cos(e) * math.sin(a), "z": math.sin(e), "activity": act}


def selftest():
    rng = np.random.default_rng(0)
    g = GateCore(cal_s=10.0)
    # calibration: known source id 4 at az -3 el +11 (jitter 3 deg) + a weak other track for a moment
    t = 0.0
    for k in range(300):
        srcs = [_src(4, -3 + rng.normal(0, 3), 11 + rng.normal(0, 3))]
        if k % 10 == 0:
            srcs.append(_src(7, 120, 20, 0.4))
        roles = g.on_sst(srcs, t)
        if t < 10:
            assert all(r in ("CAL", "EMPTY") for r in roles), roles      # nothing passes while calibrating
        t += 0.034
    assert g.phase == "RUN" and g.known["id"] == 4, g.known
    assert abs(g.known["az"] + 3) < 2 and abs(g.known["el"] - 11) < 2 and 10 <= g.known["cone"] <= 20, g.known
    d = g.to_dict()
    print("calibrated:", d)
    # run: known id 4 drifts far (-60, 58) -> still KNOWN by id; new id 9 inside the cone -> KNOWN;
    # target id 12 at az 80 -> TARGET; mirror below the plane (el -11) of the known point -> KNOWN
    roles = g.on_sst([_src(4, -60, 58), _src(9, 2, 15), _src(12, 80, 10), _src(15, -3, -11)], t)
    assert roles == ["KNOWN", "KNOWN", "TARGET", "KNOWN"], roles
    assert g.blocked(0) and g.blocked(1) and not g.blocked(2)
    roles = g.on_sst([_src(9, 100, 10)], t + 1)                          # id 9 moved away: still KNOWN
    assert roles == ["KNOWN"], roles
    # audio: blocked channels silenced, others untouched
    a = (rng.normal(size=(384, 4)) * 3000).astype("<i2")
    b = np.frombuffer(silence(a.tobytes(), 4, "signed_16", [0, 1]), "<i2").reshape(-1, 4)
    assert not b[:, :2].any() and (b[:, 2:] == a[:, 2:]).all()
    # pairing like the classifier: tracks listed [t1, t3] live in channels 1 and 3
    z4 = np.zeros((384, 4), "<i2"); z4[:, 1] = 2000; z4[:, 3] = 1500
    assert live_channels(z4.tobytes(), 4, "signed_16") == [1, 3]
    # load from file
    h = GateCore()
    h.load(d)
    assert h.phase == "RUN" and h.on_sst([_src(30, -3, 11), _src(31, 90, 0)], 0.0) == ["KNOWN", "TARGET"]
    # not enough data -> keeps calibrating
    z = GateCore(cal_s=1.0)
    z.on_sst([], 0.0); z.on_sst([], 2.0)
    assert z.phase == "CAL"
    print("selftest PASSED")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        run_node()

#!/usr/bin/env python3
"""latency_trace.py - live trace of how FAST each stage of DALBD detects (vision, tracker, sound, fusion).

Joins BOTH domains (vision 0, sound 42) like the fusion node. Run it in the Isaac ROS container.

Every second, one line:
  14:02:11 | YOLO 29.7Hz det 1 lat 41ms | TRACK 29.9Hz tracking lat 48ms | SOUND 10.0Hz DRONE p0.62 lat 95ms | FUSION VISUAL_TRACK
  lat = processing latency on the Jetson = now - header.stamp (stamp = frame arrival in net_camera_node,
        or the "stamp" field of /drone_warning).
Every change as soon as it happens (EVENT lines): YOLO sees / loses, tracker status, sound DRONE / CLEAR,
  fusion state change.
REACTION TEST: press ENTER at the exact moment you act (step into the camera view, start the drone sound,
  step out of view...). Each following first event is printed with its delay from that ENTER:
      +0.31 s  YOLO detection     +0.45 s  TRACK tracking     +1.62 s  SOUND DRONE    +1.70 s  FUSION SOUND_TRACK
  At Ctrl-C: summary of all reaction times + CSV files (+ PNG timeline if matplotlib is installed).

Also publishes (domain 0) std_msgs/Float32 for rqt_plot / PlotJuggler:
  /trace/yolo_hz /trace/yolo_lat_ms /trace/track_lat_ms /trace/sound_lat_ms /trace/sound_prob /trace/fusion_valid

Run:  ros2 run dalbd_fusion latency_trace          (env FUSION_VISION_DOMAIN / FUSION_SOUND_DOMAIN, default 0 / 42)
Plot a saved run:  ros2 run dalbd_fusion latency_trace --plot <log_dir>/trace_events_<date>.csv
Options:  det_topic:=/detections_output  log_dir:=/workspaces/isaac_ros-dev/fusion_logs
"""
import collections
import csv
import json
import os
import sys
import threading
import time


def _arg(name, default):
    for a in sys.argv:
        if a.startswith(name + ":="):
            return a.split(":=", 1)[1]
    return default


def hhmmss(t):
    return time.strftime("%H:%M:%S", time.localtime(t)) + f".{int((t % 1) * 1000):03d}"


class Trace:
    """Pure bookkeeping (no ROS): rates, latencies, edges, ENTER marks."""
    KINDS = ("YOLO detection", "YOLO lost", "TRACK tracking", "TRACK searching", "SOUND DRONE", "SOUND CLEAR",
             "FUSION VISUAL_TRACK", "FUSION SOUND_TRACK", "FUSION SEARCH")

    def __init__(self, ev_writer=None, out=print):
        self.lock = threading.Lock()
        self.win = collections.defaultdict(collections.deque)   # name -> deque[(t, lat_ms)]
        self.last = {}                                           # name -> last value
        self.ev_w, self.out = ev_writer, out
        self.mark_t, self.mark_n, self.mark_seen = None, 0, set()
        self.reactions = []                                      # (mark_n, kind, delay_s)
        self.det_seen, self.det_last_t = False, 0.0
        self.track, self.sound, self.fusion = None, None, None

    # -------------------------------------------------------------- samples
    def sample(self, name, t, lat_ms=None, value=None):
        with self.lock:
            self.win[name].append((t, lat_ms))
            while self.win[name] and t - self.win[name][0][0] > 1.0:
                self.win[name].popleft()
            if value is not None:
                self.last[name] = value

    def stats(self, name, now):
        with self.lock:
            d = [x for x in self.win.get(name, ()) if now - x[0] <= 1.0]
        lats = [x[1] for x in d if x[1] is not None]
        return len(d), (sum(lats) / len(lats) if lats else None), (max(lats) if lats else None)

    # -------------------------------------------------------------- edges
    def event(self, t, kind, info=""):
        line = f"{hhmmss(t)}  EVENT  {kind:<20} {info}"
        with self.lock:
            if self.mark_t is not None and kind not in self.mark_seen and t >= self.mark_t:
                self.mark_seen.add(kind)
                dly = t - self.mark_t
                self.reactions.append((self.mark_n, kind, dly))
                line += f"   <-- +{dly:.2f} s after MARK {self.mark_n}"
        self.out(line)
        if self.ev_w:
            self.ev_w.writerow([round(t, 3), kind, info, self.mark_n if self.mark_t else "",
                                round(t - self.mark_t, 3) if self.mark_t and t >= self.mark_t else ""])

    def mark(self, t):
        with self.lock:
            self.mark_n += 1
            self.mark_t, self.mark_seen = t, set()
        self.out(f"{hhmmss(t)}  ===== MARK {self.mark_n} (ENTER) ===== delays below are measured from here")
        if self.ev_w:
            self.ev_w.writerow([round(t, 3), "MARK", self.mark_n, self.mark_n, 0.0])

    def on_detections(self, t, n, lat_ms):
        self.sample("yolo", t, lat_ms, n)
        if n > 0:
            self.det_last_t = t
            if not self.det_seen:
                self.det_seen = True
                self.event(t, "YOLO detection", f"{n} box(es), lat {lat_ms:.0f} ms" if lat_ms is not None else "")

    def check_det_lost(self, now, hold_s=None):
        if hold_s is None:                       # 3 YOLO frame periods (at 3 Hz a single empty gap is 0.33 s)
            n = len([x for x in self.win.get("yolo", ()) if now - x[0] <= 1.0])
            hold_s = max(0.3, 3.0 / n) if n else 1.0
        if self.det_seen and now - self.det_last_t > hold_s:
            self.det_seen = False
            self.event(self.det_last_t, "YOLO lost", f"last box (confirmed after {hold_s:.1f} s without box)")

    def on_status(self, t, s):
        self.sample("status", t, None, s)
        s = s.strip().lower()
        if s != self.track and s in ("tracking", "searching"):
            self.track = s
            self.event(t, f"TRACK {s}")

    def on_warning(self, t, d):
        lvl = str(d.get("level_name", "?")).upper()
        lat = None
        st = d.get("stamp")
        try:
            age = t - float(st)
            lat = age * 1000.0 if 0.0 <= age < 30.0 else None
        except (TypeError, ValueError):
            pass
        self.sample("sound", t, lat, lvl)
        if lvl == "DRONE":
            self.last["sound_prob"] = float(d.get("prob") or 0.0)
        if lvl in ("DRONE", "CLEAR") and lvl != self.sound:
            self.sound = lvl
            info = f"p {float(d.get('prob') or 0):.2f} id {d.get('id')}" if lvl == "DRONE" else ""
            self.event(t, f"SOUND {lvl}", info)

    def on_fused(self, t, d):
        st = d.get("state")
        self.sample("fusion", t, None, st)
        self.last["fusion_valid"] = d.get("valid")
        if st and st != self.fusion:
            self.fusion = st
            self.event(t, f"FUSION {st}", f"source {d.get('source')}")

    # -------------------------------------------------------------- outputs
    def line(self, now):
        def part(label, name, extra=""):
            n, mean, mx = self.stats(name, now)
            lat = f" lat {mean:.0f}ms (max {mx:.0f})" if mean is not None else ""
            return f"{label} {n:4.1f}Hz{extra}{lat}"
        y = part("YOLO", "yolo", f" det {self.last.get('yolo', '-')}")
        tr = part("TRACK", "bearing", f" {self.track or '-'}")
        so = part("SOUND", "sound", f" {self.sound or '-'}" + (f" p{self.last['sound_prob']:.2f}"
                                                                if self.sound == "DRONE" and "sound_prob" in self.last else ""))
        return f"{time.strftime('%H:%M:%S')} | {y} | {tr} | {so} | FUSION {self.fusion or '-'}"

    def summary(self):
        if not self.reactions:
            return "no ENTER mark was given: no reaction times (press ENTER at the moment you act)"
        rows = ["", "REACTION TIMES (seconds after each ENTER mark)"]
        for k in self.KINDS:
            v = [(m, d) for (m, kk, d) in self.reactions if kk == k]
            if v:
                ds = [d for _, d in v]
                rows.append(f"  {k:<20} mean {sum(ds) / len(ds):5.2f}  min {min(ds):5.2f}  max {max(ds):5.2f}   "
                            + "  ".join(f"M{m}:{d:.2f}" for m, d in v))
        return "\n".join(rows)


def plot(ev_csv, png=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ev = list(csv.reader(open(ev_csv)))[1:]
    st_csv = ev_csv.replace("trace_events_", "trace_stats_")
    stats = list(csv.DictReader(open(st_csv))) if os.path.exists(st_csv) else []
    if not ev and not stats:
        print("empty log")
        return
    t0 = min([float(r[0]) for r in ev] + [float(r["t"]) for r in stats])
    lanes = {"YOLO": 4, "TRACK": 3, "SOUND": 2, "FUSION": 1}
    on = {"YOLO detection": ("YOLO", True), "YOLO lost": ("YOLO", False), "TRACK tracking": ("TRACK", True),
          "TRACK searching": ("TRACK", False), "SOUND DRONE": ("SOUND", True), "SOUND CLEAR": ("SOUND", False)}
    fcol = {"FUSION VISUAL_TRACK": "tab:blue", "FUSION SOUND_TRACK": "tab:green", "FUSION SEARCH": "lightgrey"}
    t_end = max([float(r[0]) for r in ev] + [float(r["t"]) for r in stats]) - t0
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True, gridspec_kw={"height_ratios": [1.2, 1]})
    open_t = {}
    fus = None
    for r in ev:
        t, k = float(r[0]) - t0, r[1]
        if k == "MARK":
            a1.axvline(t, color="red", lw=1.5)
            a1.text(t, 4.6, f"MARK {r[2]}", color="red", fontsize=8)
            continue
        if k in on:
            lane, is_on = on[k]
            if is_on:
                open_t[lane] = t
            elif lane in open_t:
                t1 = open_t.pop(lane)
                a1.broken_barh([(t1, t - t1)], (lanes[lane] - 0.35, 0.7), color="tab:orange")
        if k in fcol:
            if fus:
                a1.broken_barh([(fus[0], t - fus[0])], (0.65, 0.7), color=fcol[fus[1]])
            fus = (t, k)
        if r[4]:
            a1.text(t, lanes.get(k.split()[0], 1) + 0.38, f"+{float(r[4]):.2f}s", fontsize=7)
    for lane, t in open_t.items():
        a1.broken_barh([(t, t_end - t)], (lanes[lane] - 0.35, 0.7), color="tab:orange")
    if fus:
        a1.broken_barh([(fus[0], t_end - fus[0])], (0.65, 0.7), color=fcol[fus[1]])
    a1.set_yticks(list(lanes.values()), [f"{k}" for k in lanes])
    a1.set_ylim(0.4, 4.9)
    a1.set_title("DALBD detection trace (orange = detected; fusion: blue vision, green sound, grey search; "
                 "red = ENTER mark)")
    if stats:
        ts = [float(r["t"]) - t0 for r in stats]
        for key, lab in (("yolo_lat_ms", "YOLO latency"), ("track_lat_ms", "tracker latency"),
                         ("sound_lat_ms", "sound latency")):
            ys = [float(r[key]) if r[key] else float("nan") for r in stats]
            a2.plot(ts, ys, label=lab)
        a2.set_ylabel("ms (now - stamp)")
        a2.legend(loc="upper right")
        a2.grid(alpha=0.3)
    a2.set_xlabel("time (s)")
    png = png or ev_csv.replace(".csv", ".png")
    fig.tight_layout()
    fig.savefig(png, dpi=110)
    print("saved", png)


def main(args=None):
    if "--plot" in sys.argv:
        f = sys.argv[sys.argv.index("--plot") + 1]
        plot(f)
        return

    import rclpy
    from rclpy.context import Context
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    from rosidl_runtime_py.utilities import get_message
    from std_msgs.msg import Float32

    vdom = int(_arg("vision_domain", os.environ.get("FUSION_VISION_DOMAIN", "0")))
    sdom = int(_arg("sound_domain", os.environ.get("FUSION_SOUND_DOMAIN", "42")))
    det_topic = _arg("det_topic", "/detections_output")
    log_dir = _arg("log_dir", "/workspaces/isaac_ros-dev/fusion_logs")
    os.makedirs(log_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    ev_path = os.path.join(log_dir, f"trace_events_{stamp}.csv")
    st_path = os.path.join(log_dir, f"trace_stats_{stamp}.csv")
    ev_f, st_f = open(ev_path, "w", newline=""), open(st_path, "w", newline="")
    ev_w, st_w = csv.writer(ev_f), csv.writer(st_f)
    ev_w.writerow(["t", "kind", "info", "mark", "delay_s"])
    st_w.writerow(["t", "yolo_hz", "yolo_lat_ms", "bearing_hz", "track_lat_ms", "sound_hz", "sound_lat_ms",
                   "det", "track", "sound", "fusion"])
    tr = Trace(ev_w, out=lambda s: print(s, flush=True))

    ctx_v, ctx_s = Context(), Context()
    rclpy.init(args=sys.argv, context=ctx_v, domain_id=vdom)
    dual = sdom != vdom
    if dual:
        rclpy.init(args=sys.argv, context=ctx_s, domain_id=sdom)
    else:
        ctx_s = ctx_v
    qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=20, reliability=ReliabilityPolicy.BEST_EFFORT)

    def lat_ms(msg, now):
        h = getattr(msg, "header", None)
        if h is None or (h.stamp.sec == 0 and h.stamp.nanosec == 0):
            return None
        age = now - (h.stamp.sec + h.stamp.nanosec * 1e-9)
        return age * 1000.0 if -1.0 < age < 30.0 else None

    class Auto(Node):
        """subscribes to topics once they appear (type discovered at run time)."""
        def __init__(self, name, ctx, wanted):
            super().__init__(name, context=ctx)
            self.wanted, self.subs = dict(wanted), {}
            self.create_timer(1.0, self.discover)
            self.discover()

        def discover(self):
            types = dict(self.get_topic_names_and_types())
            for topic, cb in self.wanted.items():
                if topic in self.subs or topic not in types:
                    continue
                try:
                    mt = get_message(types[topic][0])
                except (AttributeError, ModuleNotFoundError, ValueError) as e:
                    self.get_logger().warn(f"{topic}: cannot load {types[topic][0]} ({e})")
                    continue
                self.subs[topic] = self.create_subscription(mt, topic, cb, qos)
                print(f"  subscribed {topic} [{types[topic][0]}] (domain {vdom if self.context is ctx_v else sdom})",
                      flush=True)

    def on_det(msg):
        now = time.time()
        tr.on_detections(now, len(getattr(msg, "detections", []) or []), lat_ms(msg, now))

    def on_bearing(msg):
        now = time.time()
        tr.sample("bearing", now, lat_ms(msg, now))

    def on_status(msg):
        tr.on_status(time.time(), msg.data)

    def on_fused(msg):
        try:
            tr.on_fused(time.time(), json.loads(msg.data))
        except ValueError:
            pass

    def on_warning(msg):
        try:
            d = json.loads(msg.data)
        except ValueError:
            return
        if isinstance(d, dict):
            tr.on_warning(time.time(), d)

    vnode = Auto("latency_trace", ctx_v, {det_topic: on_det, "/target/bearing": on_bearing,
                                          "/target/status": on_status, "/fused/state": on_fused})
    snode = Auto("latency_trace_sound", ctx_s, {"/drone_warning": on_warning}) if dual else vnode
    if not dual:
        vnode.wanted["/drone_warning"] = on_warning
    pubs = {k: vnode.create_publisher(Float32, f"/trace/{k}", 10) for k in
            ("yolo_hz", "yolo_lat_ms", "track_lat_ms", "sound_lat_ms", "sound_prob", "fusion_valid")}

    def tick():
        now = time.time()
        yn, ylat, _ = tr.stats("yolo", now)
        bn, blat, _ = tr.stats("bearing", now)
        sn, slat, _ = tr.stats("sound", now)
        vals = {"yolo_hz": yn, "yolo_lat_ms": ylat, "track_lat_ms": blat, "sound_lat_ms": slat,
                "sound_prob": tr.last.get("sound_prob") if tr.sound == "DRONE" else 0.0,
                "fusion_valid": tr.last.get("fusion_valid")}
        for k, v in vals.items():
            if v is not None:
                pubs[k].publish(Float32(data=float(v)))
        st_w.writerow([round(now, 3), yn, "" if ylat is None else round(ylat, 1), bn,
                       "" if blat is None else round(blat, 1), sn, "" if slat is None else round(slat, 1),
                       tr.last.get("yolo", ""), tr.track or "", tr.sound or "", tr.fusion or ""])

    def report():
        print(tr.line(time.time()), flush=True)
        ev_f.flush()
        st_f.flush()

    vnode.create_timer(0.1, lambda: tr.check_det_lost(time.time()))
    vnode.create_timer(0.2, tick)
    vnode.create_timer(1.0, report)

    ex_v = SingleThreadedExecutor(context=ctx_v)
    ex_v.add_node(vnode)
    th = None
    if dual:
        ex_s = SingleThreadedExecutor(context=ctx_s)
        ex_s.add_node(snode)
        th = threading.Thread(target=ex_s.spin, daemon=True)
        th.start()

    def keys():
        for _ in sys.stdin:
            tr.mark(time.time())
    threading.Thread(target=keys, daemon=True).start()

    print(f"latency_trace: vision domain {vdom}, sound domain {sdom}, detections {det_topic}\n"
          f"  press ENTER at the moment you act (step into view / start the sound / step out) -> reaction times\n"
          f"  logs {ev_path}\n       {st_path}", flush=True)
    try:
        ex_v.spin()
    except KeyboardInterrupt:
        pass
    finally:
        print(tr.summary(), flush=True)
        ev_f.close()
        st_f.close()
        if dual:
            ex_s.shutdown()
            if th:
                th.join(timeout=1.0)
            snode.destroy_node()
        ex_v.shutdown()
        vnode.destroy_node()
        for c in (ctx_v, ctx_s) if dual else (ctx_v,):
            if c.ok():
                rclpy.shutdown(context=c)
        print(f"\nlogs: {ev_path}\n      {st_path}\n"
              f"timeline PNG: ros2 run dalbd_fusion latency_trace --plot {ev_path}", flush=True)


if __name__ == "__main__":
    main()

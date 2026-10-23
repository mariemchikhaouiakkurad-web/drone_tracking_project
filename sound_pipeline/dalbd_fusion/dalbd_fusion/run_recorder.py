#!/usr/bin/env python3
"""run_recorder.py - records ONE CSV per vision run, automatically, every time you start (or restart with another
detector_every) the tracking launch by hand:

    ros2 launch drone_target_tracker thermal_tracking.launch.py detector_every:=10 video_device:=/dev/null_cam

Leave this recorder running in its own container terminal. It watches the processes: when the launch starts it opens
    <out>/run_<date>_<time>_N<N>.csv          one row per second during the run
    <out>/run_<date>_<time>_N<N>_events.csv   every change (YOLO sees / loses, tracker status, sound, fusion state)
    <out>/run_<date>_<time>_N<N>_activity.csv one row per message = exactly WHEN each part worked:
                                              yolo (each inference), tracker (each image), sound_odas (/raw/sst),
                                              sound_classifier, sound_warning, tracker_status / fusion_state changes
                                              (busy_ms = time from the data stamp to the output = processing window)
and when the launch stops (Ctrl-C) it closes them and adds one line to
    <out>/runs_summary.csv                    one line per run: N, duration, mean / max of every column
N is read from the launch command line (detector_every:=N). Nothing in the pipeline is started or changed.

Columns (per second):
  system    P_<rail>_W (VDD_IN = whole module, VDD_CPU_GPU_CV, VDD_SOC), gpu_load_pct, gpu_freq_MHz, cpu_total_pct,
            cpu_freq_MHz, ram_used_MB, T_<zone>_C (cpu, gpu, soc, tj, ...), fan_pct
  per part  cpu_<part>_pct (100 = one core) and ram_<part>_MB for: yolo_chain, tracker, net_camera, fusion, tools,
            sound_rx, sound_odas, sound_classifier, sound_warning
  latency   yolo_hz, yolo_lat_ms / _max_ms, yolo_boxes | track_hz, track_lat_ms / _max_ms, track_status |
            sound_hz, sound_lat_ms / _max_ms, sound_level, sound_prob | fusion_state, fusion_valid
            (lat = processing latency on the Jetson: now - header.stamp, stamp = frame arrival in net_camera_node)
  tracker   trk_ms_mean, trk_ms_p95, checks, confirmed, corrected, lost, reacquired, restarts - only if the launch
            output is written to <out>/launch.log:
                ros2 launch ... detector_every:=10 video_device:=/dev/null_cam 2>&1 | tee <out>/launch.log

Run (JETSON container, any domain - it joins vision 0 and sound 42 itself):
  ros2 run dalbd_fusion run_recorder
Options (name:=value): out:=/workspaces/isaac_ros-dev/fusion_logs/runs
                       tegrastats_log:=/workspaces/isaac_ros-dev/fusion_logs/tegrastats.log (written by host_prep.sh)
                       vision_domain:=0  sound_domain:=42  print_every:=5
                       warmup_s:=10  (first seconds of each run left out of runs_summary.csv; the per-second CSV keeps them)
"""
import csv
import glob
import json
import os
import re
import sys
import threading
import time

from dalbd_fusion.detector_sweep import TRACK_RE, SysSampler
from dalbd_fusion.latency_trace import Trace

LAUNCH_RX = re.compile(r"thermal_tracking\.launch")
N_RX = re.compile(r"detector_every:=(\d+)")
TRACK_KEYS = ("mode", "images", "tracked", "trk_ms_mean", "trk_ms_p95", "checks", "confirmed", "corrected",
              "lost", "reacquired", "restarts")


def _arg(name, default):
    for a in sys.argv:
        if a.startswith(name + ":="):
            return a.split(":=", 1)[1]
    return default


def find_launch():
    """(pid, cmdline) of the running 'ros2 launch ... thermal_tracking.launch.py', or (None, '')."""
    me = os.getpid()
    for pid in os.listdir("/proc"):
        if not pid.isdigit() or int(pid) == me:
            continue
        try:
            cmd = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="ignore").strip()
        except OSError:
            continue
        if LAUNCH_RX.search(cmd) and " launch " in f" {cmd} " and "run_recorder" not in cmd:
            return int(pid), cmd
    return None, ""


def freq_mhz(patterns):
    for p in patterns:
        for f in glob.glob(p):
            try:
                v = int(open(f).read().strip())
            except (OSError, ValueError):
                continue
            return round(v / 1e6, 0) if v > 1e5 * 100 else round(v / 1e3, 0)   # Hz (devfreq) or kHz (cpufreq)
    return None


GPU_FREQ = ("/sys/devices/platform/bus@0/17000000.gpu/devfreq/17000000.gpu/cur_freq",
            "/sys/class/devfreq/17000000.gpu/cur_freq", "/sys/class/devfreq/*gpu*/cur_freq")
CPU_FREQ = ("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq",)


class TrackerLog:
    """follows <out>/launch.log (if the user tees the launch output there) and keeps the last counters."""
    def __init__(self, path):
        self.path, self.pos, self.ino, self.last = path, 0, None, {}

    def reset(self):
        self.last = {}
        try:                                    # start from the current end: only lines of this run count
            st = os.stat(self.path)
            self.ino, self.pos = st.st_ino, st.st_size
        except OSError:
            self.ino, self.pos = None, 0

    def poll(self):
        try:
            st = os.stat(self.path)
        except OSError:
            return self.last
        if st.st_ino != self.ino or st.st_size < self.pos:      # new file (tee restarted)
            self.ino, self.pos = st.st_ino, 0
        try:
            with open(self.path, "r", errors="ignore") as f:
                f.seek(self.pos)
                data = f.read()
                self.pos = f.tell()
        except OSError:
            return self.last
        for line in data.splitlines():
            m = TRACK_RE.search(line)
            if m:
                self.last = dict(zip(TRACK_KEYS, m.groups()))
        return self.last


class Session:
    def __init__(self, out, pid, cmd):
        m = N_RX.search(cmd)
        self.n = m.group(1) if m else "default"
        self.pid, self.cmd, self.t0 = pid, cmd, time.time()
        base = os.path.join(out, time.strftime(f"run_%Y%m%d_%H%M%S_N{self.n}"))
        self.path, self.ev_path = base + ".csv", base + "_events.csv"
        self.f = open(self.path, "w", newline="")
        self.ev_f = open(self.ev_path, "w", newline="")
        self.ev_w = csv.writer(self.ev_f)
        self.ev_w.writerow(["t", "kind", "info", "mark", "delay_s"])
        self.w, self.cols, self.rows = None, None, []
        self.act_path = base + "_activity.csv"           # one row per message: exactly when each part worked
        self.act_f = open(self.act_path, "w", newline="")
        self.act_w = csv.writer(self.act_f)
        self.act_w.writerow(["t", "t_s", "part", "start_s", "busy_ms", "info"])
        self.act_lock = threading.Lock()

    def act(self, part, t, lat_ms=None, info=""):
        """part done at time t; it worked on data stamped (t - lat): busy window [start_s, t_s]."""
        ts = t - self.t0
        st = ts - lat_ms / 1000.0 if lat_ms is not None else ""
        with self.act_lock:
            self.act_w.writerow([round(t, 4), round(ts, 4), part, "" if st == "" else round(st, 4),
                                 "" if lat_ms is None else round(lat_ms, 1), info])

    def write(self, row):
        if self.w is None:
            self.cols = list(row)
            self.w = csv.DictWriter(self.f, fieldnames=self.cols, extrasaction="ignore")
            self.w.writeheader()
        self.w.writerow(row)
        self.rows.append(row)
        self.f.flush()
        self.ev_f.flush()
        with self.act_lock:
            self.act_f.flush()

    def close(self):
        self.f.close()
        self.ev_f.close()
        with self.act_lock:
            self.act_f.close()


def summarize(s, warmup_s=10.0):
    t1 = time.time()
    out = {"run_file": os.path.basename(s.path), "detector_every": s.n,
           "start": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(s.t0)),
           "end": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t1)),
           "duration_s": round(t1 - s.t0, 1), "seconds_recorded": len(s.rows)}
    rows = [r for r in s.rows if (fnum(r.get('t_s')) or 0.0) >= warmup_s] or s.rows   # skip start-up (TensorRT load, first search)
    out["warmup_skipped_s"] = warmup_s if len(rows) < len(s.rows) else 0
    if not rows:
        return out

    def share(col, val):
        v = [r.get(col) for r in rows if r.get(col) not in (None, "")]
        return round(100.0 * sum(1 for x in v if str(x) == val) / len(v), 1) if v else ""
    out["tracking_pct"] = share("track_status", "tracking")
    out["visual_track_pct"] = share("fusion_state", "VISUAL_TRACK")
    out["sound_track_pct"] = share("fusion_state", "SOUND_TRACK")
    out["search_pct"] = share("fusion_state", "SEARCH")
    out["sound_drone_pct"] = share("sound_level", "DRONE")
    for c in s.cols or []:
        if c in ("time", "t_s", "detector_every"):
            continue
        v = []
        for r in rows:
            try:
                v.append(float(r[c]))
            except (KeyError, TypeError, ValueError):
                pass
        if not v:
            continue
        if c in ("lost", "reacquired", "restarts", "checks", "confirmed", "corrected", "images", "tracked"):
            out[f"{c}_total"] = v[-1]                       # counters: value at the end of the run
            continue
        out[f"mean_{c}"] = round(sum(v) / len(v), 2)
        if re.match(r"(P_|T_|cpu_|ram_|gpu_|fan|.*lat|trk_ms)", c):
            out[f"max_{c}"] = round(max(v), 2)
    def med(col, sel=None):
        v = sorted(float(r[col]) for r in rows if r.get(col) not in (None, "") and (sel is None or sel(r)))
        return round(v[len(v) // 2], 2) if v else ""
    for c in ("yolo_lat_ms", "track_lat_ms", "sound_lat_ms"):
        out[f"p50_{c}"] = med(c)                          # median: not pulled up by the rare 1-2 s spikes
    trk = lambda r: r.get("track_status") == "tracking"   # noqa: E731
    out["seconds_tracking"] = sum(1 for r in rows if trk(r))
    for c in ("P_VDD_IN_W", "P_VDD_CPU_GPU_CV_W", "gpu_load_pct", "cpu_total_pct", "yolo_hz", "track_hz"):
        v = [float(r[c]) for r in rows if trk(r) and r.get(c) not in (None, "")]
        if v:
            out[f"tracking_mean_{c}"] = round(sum(v) / len(v), 2)   # effect of N: only while tracking
    e = [float(r[k]) for r in rows for k in ("P_VDD_IN_W",) if r.get(k) not in (None, "")]
    if e:
        out["energy_VDD_IN_Wh"] = round(sum(e) / 3600.0, 4)    # one sample per second
    return out


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def append_summary(path, line):
    old = []
    if os.path.exists(path):
        with open(path, newline="") as f:
            old = list(csv.DictReader(f))
    cols = []
    for r in old + [line]:
        for k in r:
            if k not in cols:
                cols.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in old + [line]:
            w.writerow(r)


def main(args=None):
    import rclpy
    from rclpy.context import Context
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    from rosidl_runtime_py.utilities import get_message

    out = _arg("out", "/workspaces/isaac_ros-dev/fusion_logs/runs")
    tlog = _arg("tegrastats_log", "/workspaces/isaac_ros-dev/fusion_logs/tegrastats.log")
    vdom = int(_arg("vision_domain", os.environ.get("FUSION_VISION_DOMAIN", "0")))
    sdom = int(_arg("sound_domain", os.environ.get("FUSION_SOUND_DOMAIN", "42")))
    pevery = int(_arg("print_every", "5"))
    warmup = float(_arg("warmup_s", "10"))
    os.makedirs(out, exist_ok=True)
    summary_path = os.path.join(out, "runs_summary.csv")
    trk = TrackerLog(os.path.join(out, "launch.log"))
    sysm = SysSampler(tlog)
    sysm.sample()                               # prime: CPU % needs a previous sample (first run had no CPU columns)
    state = {"s": None, "tick": 0}
    lock = threading.Lock()

    class EvProxy:                              # Trace writes events into the current run's file (if any)
        def writerow(self, r):
            s = state["s"]
            if s:
                s.ev_w.writerow(r)
    tr = Trace(EvProxy(), out=lambda s: None)   # events go to the CSV only (screen stays readable)

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
        def __init__(self, name, ctx, wanted):
            super().__init__(name, context=ctx)
            self.wanted, self.subs = dict(wanted), {}
            self.create_timer(1.0, self.discover)

        def discover(self):
            types = dict(self.get_topic_names_and_types())
            for topic, cb in self.wanted.items():
                if topic in self.subs or topic not in types:
                    continue
                try:
                    self.subs[topic] = self.create_subscription(get_message(types[topic][0]), topic, cb, qos)
                except (AttributeError, ModuleNotFoundError, ValueError):
                    pass

    def act(part, t, lat=None, info=""):
        s = state["s"]
        if s:
            try:
                s.act(part, t, lat, info)
            except ValueError:                  # file closed between the check and the write
                pass

    def on_det(m):
        now = time.time()
        n, lat = len(getattr(m, "detections", []) or []), lat_ms(m, now)
        tr.on_detections(now, n, lat)
        act("yolo", now, lat, f"boxes={n}")              # one message = one YOLO inference

    def on_bearing(m):
        now = time.time()
        lat = lat_ms(m, now)
        tr.sample("bearing", now, lat)
        act("tracker", now, lat)                          # one message = one tracked image

    last_status = {"s": None}

    def on_status(m):
        tr.on_status(time.time(), m.data)
        st = m.data.strip().lower()
        if st != last_status["s"]:
            last_status["s"] = st
            act("tracker_status", time.time(), None, st)

    def on_sst(m):                                        # ODAS sound source tracking output
        now = time.time()
        act("sound_odas", now, lat_ms(m, now))

    def on_cls(m):                                        # MobileNetV2 drone probability
        act("sound_classifier", time.time(), None, f"prob={getattr(m, 'data', '')}")

    last_fused = {"s": None}

    def on_fused(m):
        try:
            d = json.loads(m.data)
        except ValueError:
            return
        tr.on_fused(time.time(), d)
        if d.get("state") != last_fused["s"]:
            last_fused["s"] = d.get("state")
            act("fusion_state", time.time(), None, d.get("state"))

    def on_warning(m):
        try:
            d = json.loads(m.data)
        except ValueError:
            return
        if isinstance(d, dict):
            now = time.time()
            tr.on_warning(now, d)
            lat = None
            try:
                age = now - float(d.get("stamp"))
                lat = age * 1000.0 if 0.0 <= age < 30.0 else None
            except (TypeError, ValueError):
                pass
            act("sound_warning", now, lat, f"{d.get('level_name', '')} p={d.get('prob', '')}")

    vnode = Auto("run_recorder", ctx_v, {"/detections_output": on_det, "/target/bearing": on_bearing,
                                         "/target/status": on_status, "/fused/state": on_fused})
    sound_topics = {"/drone_warning": on_warning, "/raw/sst": on_sst, "/classifier/drone_prob": on_cls}
    snode = Auto("run_recorder_sound", ctx_s, sound_topics) if dual else vnode
    if not dual:
        vnode.wanted.update(sound_topics)

    def close_session(reason):
        s = state["s"]
        if not s:
            return
        state["s"] = None
        line = summarize(s, warmup)
        s.close()
        append_summary(summary_path, line)
        print(f"\n[STOP] run N={s.n} ended ({reason}) after {line['duration_s']} s -> {s.path}\n"
              f"       summary line added to {summary_path}\n"
              f"       mean: VDD_IN {line.get('mean_P_VDD_IN_W', '-')} W | GPU {line.get('mean_gpu_load_pct', '-')} % | "
              f"CPU {line.get('mean_cpu_total_pct', '-')} % | tj max {line.get('max_T_tj_C', '-')} C | "
              f"YOLO {line.get('mean_yolo_hz', '-')} Hz {line.get('mean_yolo_lat_ms', '-')} ms | "
              f"TRACK {line.get('mean_track_hz', '-')} Hz {line.get('mean_track_lat_ms', '-')} ms | "
              f"tracking {line.get('tracking_pct', '-')} %\n[WAIT] waiting for the next launch ...", flush=True)

    def tick():
        with lock:
            now = time.time()
            sysrow = sysm.sample()                      # always sampled: CPU % needs the previous sample
            tr.check_det_lost(now)
            pid, cmd = find_launch()
            s = state["s"]
            if s and pid != s.pid:
                close_session("launch stopped" if pid is None else "launch restarted")
                s = None
            if pid and not s:
                s = state["s"] = Session(out, pid, cmd)
                trk.reset()
                print(f"\n[REC ] launch found (pid {pid}) detector_every={s.n} -> {s.path}", flush=True)
            if not s:
                return
            row = {"time": time.strftime("%H:%M:%S"), "t_s": round(now - s.t0, 1), "detector_every": s.n}
            row.update(sysrow)
            row["gpu_freq_MHz"] = freq_mhz(GPU_FREQ)
            row["cpu_freq_MHz"] = freq_mhz(CPU_FREQ)
            for key, name in (("yolo", "yolo"), ("track", "bearing"), ("sound", "sound")):
                n, mean, mx = tr.stats(name, now)
                row[f"{key}_hz"] = n
                row[f"{key}_lat_ms"] = "" if mean is None else round(mean, 1)
                row[f"{key}_lat_max_ms"] = "" if mx is None else round(mx, 1)
            row["yolo_boxes"] = tr.last.get("yolo", "")
            row["track_status"] = tr.track or ""
            row["sound_level"] = tr.sound or ""
            row["sound_prob"] = tr.last.get("sound_prob", "") if tr.sound == "DRONE" else ""
            row["fusion_state"] = tr.fusion or ""
            row["fusion_valid"] = tr.last.get("fusion_valid", "")
            t = trk.poll()
            for k in TRACK_KEYS[1:]:
                row[k] = t.get(k, "")
            s.write(row)
            state["tick"] += 1
            if state["tick"] % pevery == 0:
                tj = next((row[k] for k in ("T_tj_C", "T_cpu_C") if k in row), "-")
                print(f"{row['time']} N={s.n} t={row['t_s']:.0f}s | VDD_IN {row.get('P_VDD_IN_W', '-')} W | "
                      f"GPU {row.get('gpu_load_pct', '-')}% @{row['gpu_freq_MHz']} MHz | CPU {row.get('cpu_total_pct', '-')}% | "
                      f"tj {tj} C | YOLO {row['yolo_hz']}Hz {row['yolo_lat_ms']}ms | "
                      f"TRACK {row['track_hz']}Hz {row['track_lat_ms']}ms {row['track_status']} | "
                      f"SOUND {row['sound_hz']}Hz {row['sound_lat_ms']}ms {row['sound_level']} | {row['fusion_state']}",
                      flush=True)

    vnode.create_timer(1.0, tick)
    ex_v = SingleThreadedExecutor(context=ctx_v)
    ex_v.add_node(vnode)
    th = None
    if dual:
        ex_s = SingleThreadedExecutor(context=ctx_s)
        ex_s.add_node(snode)
        th = threading.Thread(target=ex_s.spin, daemon=True)
        th.start()
    print(f"run_recorder: {sysm.describe()}\n  out {out}  (vision domain {vdom}, sound domain {sdom})\n"
          f"  tracker counters: tee the launch output to {trk.path}\n"
          f"[WAIT] waiting for 'ros2 launch drone_target_tracker thermal_tracking.launch.py ...'", flush=True)
    try:
        ex_v.spin()
    except KeyboardInterrupt:
        pass
    finally:
        with lock:
            close_session("recorder stopped")
        sysm.close()
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


if __name__ == "__main__":
    main()

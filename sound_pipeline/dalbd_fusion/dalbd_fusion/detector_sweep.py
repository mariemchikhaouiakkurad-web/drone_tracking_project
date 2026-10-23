#!/usr/bin/env python3
"""detector_sweep.py - benchmark the YOLO pipeline for several detector_every values (N) and pick the best one.

For each N it (1) starts the tracking launch with detector_every:=N, (2) waits for TensorRT, (3) asks you to stand in
view and press ENTER (or starts by itself with --no-prompt), (4) records for --duration seconds, (5) stops the launch.
Nothing else changes between runs: same camera stream, same fusion, same sound.

Recorded every second (JETSON sysfs/proc, works inside the privileged Isaac ROS container):
  power      VDD_IN (whole module), VDD_CPU_GPU_CV, VDD_SOC (INA3221 rails, W)   - falls back to tegrastats
  temperature every thermal zone (cpu, gpu, soc, tj, board...) mean / max / rise per case, fan speed %
  GPU        load %, temperature                                                   CPU: total %, RAM used MB
  processes  CPU % and RAM (MB) of: YOLO chain (component_container: encoder + TensorRT + decoder), tracker_node,
             net_camera_node, fusion state_machine, latency/stream tools, sound (rx, ODAS, classifier, warning)
Recorded from ROS (domain 0):
  /detections_output  YOLO rate, latency (now - header.stamp), share of images with a box
  /target/bearing     tracker rate, latency          /target/status  tracking share, switches
  /fused/state        share of VISUAL_TRACK / SOUND_TRACK / SEARCH
  tracker log         checks, confirmed, lost, reacquired, trk_ms_mean (from the launch output)

Outputs in <out>/ :  case<i>_N<N>.csv (per-second samples of each case), consumption.csv (one clear row per case:
                     power, energy, load, vs case 1), launch_N<N>.log, summary.csv, sweep_report.png,
                     sweep_report.html (table + charts, open it in a browser), ranking printed at the end.

Run (JETSON container, with C2 STOPPED; C1 net_camera, C3 fusion, sound keep running):
  ros2 run dalbd_fusion detector_sweep --values 2,5,10,25 --duration 60
  ros2 run dalbd_fusion detector_sweep --values 25,50,75,100 --duration 60 --ask-supply \
      --tegrastats-log /workspaces/isaac_ros-dev/fusion_logs/tegrastats.log
      (whole system: type the bench supply V and A after each case; power read from the host's tegrastats log)
  ros2 run dalbd_fusion detector_sweep --report /workspaces/isaac_ros-dev/fusion_logs/sweep_<date>   (redo charts)
Options: --warmup 25  --no-prompt  --launch-args "video_device:=/dev/null_cam confidence_threshold:=0.25"
         --weights visual=0.4,lost=0.2,correction=0.2,power=0.2
"""
import argparse
import csv
import glob
import json
import math
import os
import re
import signal
import statistics as st
import subprocess
import sys
import threading
import time

OUT_ROOT = "/workspaces/isaac_ros-dev/fusion_logs"
PROC_GROUPS = {                     # name -> regex on the process command line
    "yolo_chain": r"component_container",
    "tracker": r"tracker_node",
    "net_camera": r"net_camera_node",
    "fusion": r"state_machine",
    "tools": r"latency_trace|stream_server_node|latency_probe|fusion_viz",
    "sound_rx": r"aplay -D hw:Loopback|nc -l \d+",                       # laptop audio -> Loopback card
    "sound_odas": r"odas_core_node|odas_server_node|arecord -D",           # ODAS localisation (+ its arecord)
    "sound_classifier": r"classifier|direction_gate|target_direction|null_feeder|known_source",
    "sound_warning": r"drone_warning|warning_node",
}
TRACK_RE = re.compile(r"mode=(\w+)\s+images=(\d+)\s+tracked=(\d+)\s+trk_ms_mean=([\d.]+)\s+trk_ms_p95=([\d.]+).*?"
                      r"checks=(\d+)\s+confirmed=(\d+)\s+corrected=(\d+).*?lost=(\d+)\s+reacquired=(\d+)\s+restarts=(\d+)")


# ======================================================================= system sampling (no ROS)
def find_rails():
    """INA3221 rails -> {label: (voltage_file, current_file)}"""
    rails = {}
    for lab in glob.glob("/sys/bus/i2c/drivers/ina3221/*/hwmon/hwmon*/in*_label"):
        try:
            name = open(lab).read().strip()
        except OSError:
            continue
        idx = re.search(r"in(\d+)_label", lab).group(1)
        d = os.path.dirname(lab)
        v, c = os.path.join(d, f"in{idx}_input"), os.path.join(d, f"curr{idx}_input")
        if os.path.exists(v) and os.path.exists(c) and not name.lower().startswith("sum"):
            rails[name] = (v, c)
    return rails


def read_int(path, default=None):
    try:
        return int(open(path).read().strip())
    except (OSError, ValueError):
        return default


def gpu_load_file():
    for p in ("/sys/devices/platform/bus@0/17000000.gpu/load", "/sys/devices/platform/17000000.gpu/load",
              "/sys/devices/gpu.0/load"):
        if os.path.exists(p):
            return p
    g = glob.glob("/sys/devices/platform/*/17000000.gpu/load") + glob.glob("/sys/class/devfreq/*gpu*/device/load")
    return g[0] if g else None


def thermal_zones():
    z = {}
    for t in glob.glob("/sys/class/thermal/thermal_zone*/type"):
        try:
            name = open(t).read().strip().lower()
        except OSError:
            continue
        name = re.sub(r"[^a-z0-9]+", "_", name).strip("_").replace("_therm", "")
        z[name] = os.path.join(os.path.dirname(t), "temp")
    return z


def fan_file():
    for g in ("/sys/devices/platform/pwm-fan/hwmon/hwmon*/pwm1", "/sys/class/hwmon/hwmon*/pwm1"):
        f = glob.glob(g)
        if f:
            return f[0]
    return None


def cpu_times():
    v = [int(x) for x in open("/proc/stat").readline().split()[1:]]
    idle = v[3] + (v[4] if len(v) > 4 else 0)
    return sum(v), idle


def proc_snapshot():
    """{group: (utime+stime ticks, rss_kB)} summed over matching processes."""
    out = {g: [0, 0] for g in PROC_GROUPS}
    me = os.getpid()
    for pid in os.listdir("/proc"):
        if not pid.isdigit() or int(pid) == me:
            continue
        try:
            cmd = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="ignore")
            if not cmd or "detector_sweep" in cmd:
                continue
            g = next((g for g, rx in PROC_GROUPS.items() if re.search(rx, cmd)), None)
            if g is None:
                continue
            f = open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()
            ticks = int(f[11]) + int(f[12])
            rss = 0
            for line in open(f"/proc/{pid}/status"):
                if line.startswith("VmRSS:"):
                    rss = int(line.split()[1])
            out[g][0] += ticks
            out[g][1] += rss
        except (OSError, ValueError, IndexError):
            continue
    return out


class SysSampler:
    def __init__(self, tegra_log=""):
        self.tegra_log = tegra_log if tegra_log and os.path.exists(tegra_log) else ""
        self.rails = {} if self.tegra_log else find_rails()
        self.gpu = gpu_load_file()
        self.zones = thermal_zones()
        self.fan = fan_file()
        self.hz = os.sysconf("SC_CLK_TCK")
        self.ncpu = os.cpu_count() or 1
        self.tegra = None
        self.tegra_last = ""
        if not self.rails and not self.tegra_log:           # fallback: tegrastats
            try:
                self.tegra = subprocess.Popen(["tegrastats", "--interval", "1000"], stdout=subprocess.PIPE, text=True)
                self.tegra_last = ""
                threading.Thread(target=self._tegra_reader, daemon=True).start()
            except OSError:
                self.tegra = None
        self.prev = None

    def _tegra_reader(self):
        for line in self.tegra.stdout:
            self.tegra_last = line

    def describe(self):
        src = (f"tegrastats log {self.tegra_log}" if self.tegra_log else
               (list(self.rails) or ("tegrastats" if self.tegra else "NONE")))
        return (f"power: {src} | fan: {self.fan or 'NONE'} | GPU load: "
                f"{self.gpu or 'NONE'} | temps: {list(self.zones)}")

    def sample(self):
        now = time.monotonic()
        tot, idle = cpu_times()
        procs = proc_snapshot()
        row = {}
        for name, (v, c) in self.rails.items():
            mv, ma = read_int(v), read_int(c)
            if mv is not None and ma is not None:
                row[f"P_{name}_W"] = round(mv * ma / 1e6, 3)
        if self.tegra_log:
            try:
                with open(self.tegra_log, "rb") as f:
                    f.seek(0, 2)
                    f.seek(max(0, f.tell() - 4096))
                    lines = f.read().decode(errors="ignore").strip().splitlines()
                self.tegra_last = lines[-1] if lines else ""
            except OSError:
                self.tegra_last = ""
        if self.tegra_last:
            for name, mw in re.findall(r"(VDD_\w+) (\d+)mW", self.tegra_last):
                row[f"P_{name}_W"] = round(int(mw) / 1000.0, 3)
            m = re.search(r"GR3D_FREQ (\d+)%", self.tegra_last)
            if m:                                   # tegrastats value is averaged over its interval: preferred
                row["gpu_load_pct"] = float(m.group(1))
        if self.gpu:
            g = read_int(self.gpu)
            if g is not None:                       # sysfs 'load' is an instant snapshot (often 0 or 100 at 1 Hz)
                row["gpu_load_sysfs_pct"] = round(g / 10.0, 1)
                row.setdefault("gpu_load_pct", row["gpu_load_sysfs_pct"])
        for name, f in self.zones.items():
            t = read_int(f)
            if t is not None and -40000 < t < 150000:          # skip unused zones (-256 C etc.)
                row[f"T_{name}_C"] = round(t / 1000.0, 1)
        if self.tegra_last:                                   # tegrastats temps: cpu@45.5C gpu@44C tj@47C ...
            for name, v in re.findall(r"\b([A-Za-z0-9_]+)@(-?[\d.]+)C\b", self.tegra_last):
                key = f"T_{name.lower()}_C"
                if key not in row and -40.0 < float(v) < 150.0:
                    row[key] = float(v)
        if self.fan:
            pwm = read_int(self.fan)
            if pwm is not None:
                row["fan_pct"] = round(pwm / 255.0 * 100.0, 1)
        mem = {}
        for line in open("/proc/meminfo"):
            k, v = line.split(":")
            mem[k] = int(v.split()[0])
        row["ram_used_MB"] = round((mem["MemTotal"] - mem.get("MemAvailable", mem["MemFree"])) / 1024.0, 0)
        if self.prev:
            pt, ptot, pidle, pprocs = self.prev
            dt = now - pt
            dtot = max(tot - ptot, 1)
            row["cpu_total_pct"] = round(100.0 * (1 - (idle - pidle) / dtot), 1)
            for g in PROC_GROUPS:
                d = procs[g][0] - pprocs[g][0]
                row[f"cpu_{g}_pct"] = round(max(d, 0) / self.hz / dt * 100.0, 1)   # 100 % = one core
                row[f"ram_{g}_MB"] = round(procs[g][1] / 1024.0, 0)
        self.prev = (now, tot, idle, procs)
        return row

    def close(self):
        if self.tegra:
            self.tegra.terminate()


# ======================================================================= ROS side
class RosRecorder:
    def __init__(self):
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
        from rosidl_runtime_py.utilities import get_message
        self.rclpy = rclpy
        rclpy.init()
        self.node = Node("detector_sweep")
        self.get_message = get_message
        self.qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=20, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.lock = threading.Lock()
        self.subs = {}
        self.want = {"/detections_output": self.on_det, "/target/bearing": self.on_bearing,
                     "/target/status": self.on_status, "/fused/state": self.on_fused}
        self.reset()
        self.node.create_timer(1.0, self.discover)
        self.ex = SingleThreadedExecutor()
        self.ex.add_node(self.node)
        threading.Thread(target=self.ex.spin, daemon=True).start()

    def discover(self):
        types = dict(self.node.get_topic_names_and_types())
        for t, cb in self.want.items():
            if t in self.subs or t not in types:
                continue
            try:
                self.subs[t] = self.node.create_subscription(self.get_message(types[t][0]), t, cb, self.qos)
            except (AttributeError, ModuleNotFoundError, ValueError):
                pass

    def reset(self):
        with self.lock:
            self.d = {"det": [], "bear": [], "status": [], "fused": []}
            self.sec = {"det_n": 0, "det_box": 0, "bear_n": 0}

    def _lat(self, msg):
        h = getattr(msg, "header", None)
        if h is None or (h.stamp.sec == 0 and h.stamp.nanosec == 0):
            return None
        age = time.time() - (h.stamp.sec + h.stamp.nanosec * 1e-9)
        return age * 1000.0 if -1.0 < age < 30.0 else None

    def on_det(self, m):
        n = len(getattr(m, "detections", []) or [])
        with self.lock:
            self.d["det"].append((time.time(), n, self._lat(m)))
            self.sec["det_n"] += 1
            self.sec["det_box"] += n > 0

    def on_bearing(self, m):
        with self.lock:
            self.d["bear"].append((time.time(), self._lat(m)))
            self.sec["bear_n"] += 1

    def on_status(self, m):
        with self.lock:
            self.d["status"].append((time.time(), m.data.strip().lower()))

    def on_fused(self, m):
        try:
            s = json.loads(m.data).get("state")
        except ValueError:
            return
        with self.lock:
            self.d["fused"].append((time.time(), s))

    def take_second(self):
        with self.lock:
            s, self.sec = self.sec, {"det_n": 0, "det_box": 0, "bear_n": 0}
        return s

    def ready(self):
        with self.lock:
            return len(self.d["det"]) > 5

    def close(self):
        self.ex.shutdown()
        self.node.destroy_node()
        if self.rclpy.ok():
            self.rclpy.shutdown()


# ======================================================================= one run
def run_one(n, a, ros, sysm, out, case=1):
    log_path = os.path.join(out, f"launch_N{n}.log")
    cmd = ["ros2", "launch", "drone_target_tracker", "thermal_tracking.launch.py", f"detector_every:={n}"] + a.launch_args.split()
    print(f"\n===== N = {n}: starting  {' '.join(cmd)}", flush=True)
    logf = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    track = {"lines": []}

    def reader():
        for line in proc.stdout:
            logf.write(line)
            m = TRACK_RE.search(line)
            if m:
                track["lines"].append((time.time(), m.groups()))
    threading.Thread(target=reader, daemon=True).start()

    t0 = time.time()
    ros.reset()
    while time.time() - t0 < a.warmup or not ros.ready():
        if proc.poll() is not None:
            print(f"[FAIL] launch exited (code {proc.returncode}); see {log_path}", flush=True)
            return None
        if time.time() - t0 > a.warmup + 60:
            print("[WARN] no /detections_output after warm-up; measuring anyway", flush=True)
            break
        time.sleep(0.5)
    if not a.no_prompt:
        input(f"N = {n}: stand in the camera view (same place as the other runs), then press ENTER ... ")
    print(f"N = {n}: recording {a.duration:.0f} s", flush=True)
    ros.reset()
    track["lines"] = []
    sysm.sample()
    rows, t_start = [], time.time()
    while time.time() - t_start < a.duration:
        time.sleep(1.0)
        r = {"t": round(time.time() - t_start, 1), "N": n}
        r.update(sysm.sample())
        s = ros.take_second()
        r.update({"yolo_hz": s["det_n"], "yolo_box_share": round(s["det_box"] / s["det_n"], 3) if s["det_n"] else "",
                  "track_hz": s["bear_n"]})
        rows.append(r)
        if int(r["t"]) % 10 == 0:
            print(f"  t={r['t']:4.0f}s  YOLO {r['yolo_hz']} Hz  TRACK {r['track_hz']} Hz  "
                  f"P_in {r.get('P_VDD_IN_W', '-')} W  GPU {r.get('gpu_load_pct', '-')} %", flush=True)
    t_end = time.time()
    supply = {}
    if a.ask_supply:
        print(f"Case {case} (N = {n}) done. Read the BENCH POWER SUPPLY display (values seen during the test).")
        for key, q in (("supply_V", "  supply voltage (V), e.g. 23.0 : "), ("supply_A", "  supply current (A), e.g. 0.85: ")):
            while True:
                v = input(q).strip().replace(",", ".")
                if v == "":
                    break
                try:
                    supply[key] = float(v)
                    break
                except ValueError:
                    print("  number please (or ENTER to skip)")

    # stop the launch
    try:
        os.killpg(proc.pid, signal.SIGINT)
        proc.wait(timeout=15)
    except (subprocess.TimeoutExpired, ProcessLookupError):
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    time.sleep(3)

    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("t", "N"), k))
    with open(os.path.join(out, f"case{case}_N{n}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    s = summarize(n, rows, ros, track["lines"], t_start, t_end)
    s["case"] = case
    if "supply_V" in supply and "supply_A" in supply:
        s["supply_V"], s["supply_A"] = supply["supply_V"], supply["supply_A"]
        s["P_supply_W"] = round(supply["supply_V"] * supply["supply_A"], 2)
        s["E_supply_Wh"] = round(s["P_supply_W"] * s["duration_s"] / 3600.0, 4)
        if s.get("P_VDD_IN_W") is not None:
            s["P_carrier_and_rest_W"] = round(s["P_supply_W"] - s["P_VDD_IN_W"], 2)
    return s


def med(v):
    v = [x for x in v if x is not None and x != ""]
    return round(st.median(v), 2) if v else None


def mean(v):
    v = [x for x in v if x is not None and x != ""]
    return round(st.mean(v), 2) if v else None


def summarize(n, rows, ros, track_lines, t0, t1):
    dur = max(t1 - t0, 1.0)
    with ros.lock:
        d = {k: list(v) for k, v in ros.d.items()}
    s = {"N": n, "duration_s": round(dur, 1)}
    det = [x for x in d["det"] if t0 <= x[0] <= t1]
    s["yolo_hz"] = round(len(det) / dur, 1)
    s["yolo_lat_ms_med"] = med([x[2] for x in det])
    lats = sorted(x[2] for x in det if x[2] is not None)
    s["yolo_lat_ms_p95"] = round(lats[int(0.95 * (len(lats) - 1))], 1) if lats else None
    s["yolo_box_share"] = round(sum(1 for x in det if x[1] > 0) / len(det), 3) if det else None
    bear = [x for x in d["bear"] if t0 <= x[0] <= t1]
    s["track_hz"] = round(len(bear) / dur, 1)
    s["track_lat_ms_med"] = med([x[1] for x in bear])
    stt = [x for x in d["status"] if t0 <= x[0] <= t1]
    s["tracking_share"] = round(sum(1 for x in stt if x[1] == "tracking") / len(stt), 3) if stt else None
    s["status_switches"] = sum(1 for a, b in zip(stt, stt[1:]) if a[1] != b[1])
    fu = [x for x in d["fused"] if t0 <= x[0] <= t1]
    for name in ("VISUAL_TRACK", "SOUND_TRACK", "SEARCH"):
        s[f"fusion_{name.lower()}_share"] = round(sum(1 for x in fu if x[1] == name) / len(fu), 3) if fu else None
    s["fusion_transitions"] = sum(1 for a, b in zip(fu, fu[1:]) if a[1] != b[1])
    if len(track_lines) >= 2:
        g0, g1 = track_lines[0][1], track_lines[-1][1]
        dl = lambda i: int(g1[i]) - int(g0[i])  # noqa: E731
        s["trk_ms_mean"] = float(g1[3])
        s["checks"], s["confirmed"], s["corrected"] = dl(5), dl(6), dl(7)
        s["lost"], s["reacquired"], s["restarts"] = dl(8), dl(9), dl(10)
    elif track_lines:
        s["trk_ms_mean"] = float(track_lines[-1][1][3])
    s["lost_per_min"] = round(s.get("lost", 0) / dur * 60, 2) if "lost" in s else None
    s["confirm_ratio"] = round(s["confirmed"] / s["checks"], 3) if s.get("checks") else None
    s["correction_interval_s"] = round(n / s["track_hz"], 2) if s["track_hz"] else None
    for k in sorted({k for r in rows for k in r}):
        if k.startswith(("P_", "gpu_", "cpu_", "ram_", "T_", "fan_")):
            s[k] = mean([r.get(k) for r in rows])
        if k.startswith("T_") and k.endswith("_C"):
            vals = [r.get(k) for r in rows if r.get(k) not in (None, "")]
            if vals:
                zone = k[2:-2]
                s[f"Tmax_{zone}_C"] = round(max(vals), 1)
                s[f"Trise_{zone}_C"] = round(vals[-1] - vals[0], 1)      # heating during the case
        if k.startswith("P_") and k.endswith("_W"):
            vals = [r.get(k) for r in rows if r.get(k) not in (None, "")]
            if vals:
                rail = k[2:-2]
                s[f"P_{rail}_max_W"] = round(max(vals), 2)
                s[f"E_{rail}_Wh"] = round(sum(vals) * (dur / len(vals)) / 3600.0, 4)   # energy over the test
    return s


CONS_COLS = [
    ("case", "case"), ("N", "detector_every"), ("duration_s", "duration_s"),
    ("P_supply_W", "whole_system_power_W (bench supply)"), ("E_supply_Wh", "whole_system_energy_Wh"),
    ("P_carrier_and_rest_W", "carrier+fan+USB_power_W (supply - module)"),
    ("P_VDD_IN_W", "module_power_mean_W (VDD_IN)"), ("P_VDD_IN_max_W", "module_power_max_W"),
    ("E_VDD_IN_Wh", "module_energy_Wh"),
    ("P_VDD_CPU_GPU_CV_W", "cpu_gpu_cv_power_W"), ("P_VDD_SOC_W", "soc_power_W"),
    ("gpu_load_pct", "gpu_load_pct"), ("cpu_total_pct", "cpu_total_pct"),
    ("cpu_yolo_chain_pct", "cpu_yolo_chain_pct"), ("cpu_tracker_pct", "cpu_tracker_pct"),
    ("cpu_net_camera_pct", "cpu_net_camera_pct"), ("cpu_fusion_pct", "cpu_fusion_pct"),
    ("ram_used_MB", "ram_used_MB"), ("yolo_hz", "yolo_hz"), ("track_hz", "tracker_hz"),
    ("yolo_lat_ms_med", "yolo_latency_ms"), ("track_lat_ms_med", "tracker_latency_ms"),
    ("correction_interval_s", "yolo_check_every_s"), ("tracking_share", "tracking_share"),
    ("lost_per_min", "tracker_lost_per_min"), ("fusion_visual_track_share", "fusion_visual_share"),
]


def write_consumption(out, summary):
    """consumption.csv: one clear row per case + the change vs case 1 (%)."""
    rows = sorted(summary, key=lambda r: r.get("case", 0))
    zones = sorted({k[2:-2] for r in rows for k in r if k.startswith("T_") and k.endswith("_C")})
    tcols = []
    for z in zones:
        tcols += [(f"T_{z}_C", f"temp_{z}_mean_C"), (f"Tmax_{z}_C", f"temp_{z}_max_C"), (f"Trise_{z}_C", f"temp_{z}_rise_C")]
    tcols.append(("fan_pct", "fan_pct"))
    cols = ([(k, lab) for k, lab in CONS_COLS if any(r.get(k) is not None for r in rows)]
            + [(k, lab) for k, lab in tcols if any(r.get(k) is not None for r in rows)])
    base = rows[0] if rows else {}
    deltas = [k for k in ("P_supply_W", "P_VDD_IN_W", "E_VDD_IN_Wh", "gpu_load_pct", "cpu_total_pct")
              if any(r.get(k) is not None for r in rows)]
    path = os.path.join(out, "consumption.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([lab for _, lab in cols] + [f"{k}_vs_case1_pct" for k in deltas])
        for r in rows:
            d = []
            for k in deltas:
                b, v = base.get(k), r.get(k)
                d.append(round(100.0 * (v - b) / b, 1) if b not in (None, 0) and v is not None else "")
            w.writerow([r.get(k, "") if r.get(k) is not None else "" for k, _ in cols] + d)
    print(f"\n=== consumption per case ({path}) ===")
    hot = next((f"Tmax_{z}_C" for z in ("tj", "gpu", "cpu", "soc") if z in zones), None)
    show = [k for k in ("case", "N", "P_supply_W", "P_VDD_IN_W", "P_VDD_IN_max_W", "E_VDD_IN_Wh",
                        "gpu_load_pct", "cpu_total_pct", hot, "fan_pct", "yolo_hz", "track_hz", "lost_per_min")
            if k and any(r.get(k) is not None for r in rows)]
    print("  ".join(f"{k[:13]:>13}" for k in show))
    for r in rows:
        print("  ".join(f"{('-' if r.get(k) is None else r[k]):>13}" for k in show))
    return path


# ======================================================================= ranking + report
def parse_weights(w):
    out = {"visual": 0.4, "lost": 0.2, "correction": 0.2, "power": 0.2}
    for part in (w or "").split(","):
        if "=" in part:
            k, v = part.split("=")
            out[k.strip()] = float(v)
    return out


def rank(summary, weights):
    def norm(vals, higher_better):
        v = [x for x in vals if x is not None]
        if not v or max(v) == min(v):
            return [1.0 if x is not None else 0.0 for x in vals]
        lo, hi = min(v), max(v)
        return [((x - lo) / (hi - lo) if higher_better else (hi - x) / (hi - lo)) if x is not None else 0.0
                for x in vals]
    vis = norm([r.get("fusion_visual_track_share") for r in summary], True)
    lost = norm([r.get("lost_per_min") for r in summary], False)
    corr = norm([r.get("correction_interval_s") for r in summary], False)
    pw = norm([r.get("P_VDD_IN_W") for r in summary], False)
    for i, r in enumerate(summary):
        r["score"] = round(weights["visual"] * vis[i] + weights["lost"] * lost[i] + weights["correction"] * corr[i]
                           + weights["power"] * pw[i], 3)
    return sorted(summary, key=lambda r: -r["score"])


def report(out, weights):
    summary = list(csv.DictReader(open(os.path.join(out, "summary.csv"))))
    for r in summary:
        for k, v in r.items():
            try:
                r[k] = float(v) if v not in ("", "None") else None
            except ValueError:
                pass
    if not summary:
        print("summary.csv is empty: nothing to report")
        return
    summary.sort(key=lambda r: r["N"])
    ranked = rank([dict(r) for r in summary], weights)
    score = {r["N"]: r["score"] for r in ranked}
    for r in summary:
        r["score"] = score[r["N"]]
    cols = ["N", "score", "fusion_visual_track_share", "tracking_share", "lost_per_min", "confirm_ratio",
            "correction_interval_s", "yolo_hz", "yolo_lat_ms_med", "yolo_box_share", "track_hz", "track_lat_ms_med",
            "P_VDD_IN_W", "P_VDD_CPU_GPU_CV_W", "P_VDD_SOC_W", "gpu_load_pct", "cpu_total_pct", "ram_used_MB"]
    cols = [c for c in cols if any(r.get(c) is not None for r in summary)]
    print("\n=== detector_every sweep (best first) ===")
    print("  ".join(f"{c[:14]:>14}" for c in cols))
    for r in ranked:
        print("  ".join(f"{('-' if r.get(c) is None else (f'{r[c]:.3g}' if isinstance(r[c], float) else r[c])):>14}"
                        for c in cols))
    best = ranked[0]
    print(f"\nBEST N = {int(best['N'])}  (score {best['score']}; weights {weights})")
    png = os.path.join(out, "sweep_report.png")
    try:
        make_png(summary, png)
        print(f"charts: {png}")
    except Exception as e:  # noqa: BLE001
        print(f"(no charts: {e})")
        png = None
    html = os.path.join(out, "sweep_report.html")
    make_html(summary, ranked, cols, weights, png, html)
    print(f"report: {html}")


def make_png(summary, png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ns = [str(int(r["N"])) for r in summary]
    g = lambda k: [r.get(k) if r.get(k) is not None else 0 for r in summary]  # noqa: E731
    fig, ax = plt.subplots(3, 3, figsize=(16, 13))
    def bars(a, keys, labels, title, unit):
        w = 0.8 / len(keys)
        for i, (k, lab) in enumerate(zip(keys, labels)):
            a.bar([x + i * w for x in range(len(ns))], g(k), w, label=lab)
        a.set_xticks([x + w * (len(keys) - 1) / 2 for x in range(len(ns))], ns)
        a.set_xlabel("detector_every N")
        a.set_ylabel(unit)
        a.set_title(title)
        a.grid(axis="y", alpha=0.3)
        if len(keys) > 1:
            a.legend(fontsize=8)
    bars(ax[0][0], ["fusion_visual_track_share", "tracking_share", "yolo_box_share"],
         ["fusion VISUAL share", "tracker tracking share", "YOLO images with a box"], "Tracking quality", "share (0-1)")
    bars(ax[0][1], ["lost_per_min", "status_switches"], ["tracker lost / min", "tracking<->searching switches"],
         "Stability (lower = better)", "count")
    bars(ax[0][2], ["yolo_lat_ms_med", "track_lat_ms_med"], ["YOLO latency", "tracker latency"], "Latency (median)", "ms")
    bars(ax[1][0], ["yolo_hz", "track_hz"], ["YOLO", "tracker"], "Rates", "Hz")
    pk = [k for k in ("P_VDD_IN_W", "P_VDD_CPU_GPU_CV_W", "P_VDD_SOC_W") if any(r.get(k) for r in summary)]
    if pk:
        bars(ax[1][1], pk, [k[2:-2] for k in pk], "Power (mean)", "W")
    else:
        bars(ax[1][1], ["gpu_load_pct"], ["GPU load"], "GPU load (mean)", "%")
    ck = [f"cpu_{g}_pct" for g in PROC_GROUPS if any(r.get(f"cpu_{g}_pct") for r in summary)]
    bottom = [0.0] * len(ns)
    for k in ck:
        vals = g(k)
        ax[1][2].bar(ns, vals, bottom=bottom, label=k[4:-4])
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax[1][2].set_title("CPU per component (100 % = 1 core)")
    ax[1][2].set_xlabel("detector_every N")
    ax[1][2].set_ylabel("%")
    ax[1][2].legend(fontsize=8)
    ax[1][2].grid(axis="y", alpha=0.3)
    zones = [z for z in ("tj", "gpu", "cpu", "soc") if any(r.get(f"T_{z}_C") for r in summary)]
    if zones:
        bars(ax[2][0], [f"T_{z}_C" for z in zones], zones, "Temperature (mean per case)", "°C")
        bars(ax[2][1], [f"Tmax_{z}_C" for z in zones], zones, "Temperature (max per case)", "°C")
    else:
        ax[2][0].axis("off"); ax[2][1].axis("off")
    if any(r.get("fan_pct") for r in summary):
        bars(ax[2][2], ["fan_pct"], ["fan"], "Fan speed (mean)", "%")
    else:
        ax[2][2].axis("off")
    fig.suptitle("DALBD YOLO pipeline - detector_every sweep (MAXN)", fontsize=14)
    fig.tight_layout()
    fig.savefig(png, dpi=110)


def make_html(summary, ranked, cols, weights, png, html):
    import base64
    img = ""
    if png and os.path.exists(png):
        img = f'<img src="data:image/png;base64,{base64.b64encode(open(png, "rb").read()).decode()}" style="width:100%">'
    fmt = lambda v: "-" if v is None else (f"{v:.3g}" if isinstance(v, float) else str(v))  # noqa: E731
    head = "".join(f"<th>{c}</th>" for c in cols)
    body = "".join("<tr" + (' class="best"' if r is ranked[0] else "") + ">" +
                   "".join(f"<td>{fmt(r.get(c))}</td>" for c in cols) + "</tr>" for r in ranked)
    extra = sorted({k for r in summary for k in r if k.startswith(("cpu_", "ram_", "T_"))})
    head2 = "".join(f"<th>{c}</th>" for c in ["N"] + extra)
    body2 = "".join("<tr>" + "".join(f"<td>{fmt(r.get(c))}</td>" for c in ["N"] + extra) + "</tr>" for r in summary)
    open(html, "w").write(f"""<!doctype html><html><head><meta charset="utf-8"><title>detector_every sweep</title>
<style>body{{font-family:sans-serif;margin:20px;background:#fff;color:#222}} table{{border-collapse:collapse;margin:10px 0;font-size:13px}}
td,th{{border:1px solid #ccc;padding:4px 8px;text-align:right}} th{{background:#eef}} tr.best td{{background:#dfd;font-weight:bold}}</style></head>
<body><h2>YOLO pipeline - detector_every sweep</h2>
<p><b>Best N = {int(ranked[0]['N'])}</b> (score {ranked[0]['score']}). Score = {weights['visual']}·VISUAL share + {weights['lost']}·(few losses)
+ {weights['correction']}·(short correction interval) + {weights['power']}·(low power), each normalised 0-1 across the runs.</p>
<h3>Ranking (best first)</h3><table><tr>{head}</tr>{body}</table>
<h3>Per component (mean CPU %, RAM MB, temperatures)</h3><table><tr>{head2}</tr>{body2}</table>
{img}</body></html>""")


# ======================================================================= main
def parse_values(text):
    """'25,50,75,100' or '25:100:25' (start:stop:step, stop included) -> [25, 50, 75, 100]"""
    text = (text or "").replace(" ", "")
    if not text:
        return []
    out = []
    for part in text.split(","):
        if ":" in part:
            b = [int(x) for x in part.split(":")]
            if len(b) != 3 or b[2] <= 0 or b[0] <= 0 or b[1] < b[0]:
                raise ValueError(f"bad range '{part}': use start:stop:step, e.g. 25:100:25")
            out += list(range(b[0], b[1] + 1, b[2]))
        else:
            v = int(part)
            if v < 0:
                raise ValueError("values must be >= 0 (0 = YOLO on every image)")
            out.append(v)
    return out


def main():
    ap = argparse.ArgumentParser(description="detector_every sweep: power, load, latency and tracking quality per N")
    ap.add_argument("--values", default="",
                    help='frame values to test: list "25,50,75,100" or range "25:100:25" (start:stop:step); '
                         'empty = asked at start')
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--warmup", type=float, default=25.0)
    ap.add_argument("--launch-args", default="video_device:=/dev/null_cam")
    ap.add_argument("--no-prompt", action="store_true")
    ap.add_argument("--out", default="")
    ap.add_argument("--weights", default="")
    ap.add_argument("--report", default="", help="only rebuild the report of an existing sweep folder")
    ap.add_argument("--ask-supply", action="store_true", help="after each case, type the bench supply V and A (whole system)")
    ap.add_argument("--tegrastats-log", default="", help="read power from a tegrastats --logfile written on the host")
    a = ap.parse_args([x for x in sys.argv[1:] if not x.startswith("--ros-args")])
    weights = parse_weights(a.weights)
    if a.report:
        report(a.report, weights)
        return
    if subprocess.run(["pgrep", "-f", "thermal_tracking.launch.py"], capture_output=True).returncode == 0:
        sys.exit("[STOP] a thermal_tracking launch is already running (C2): stop it first (Ctrl-C), then run the sweep")
    out = a.out or os.path.join(OUT_ROOT, time.strftime("sweep_%Y%m%d_%H%M%S"))
    os.makedirs(out, exist_ok=True)
    values = parse_values(a.values)
    while not values:
        try:
            values = parse_values(input('detector_every values to test (e.g. 25,50,75,100 or 25:100:25): '))
        except ValueError as e:
            print(f"  {e}")
            values = []
    print(f"cases: " + ", ".join(f"{i}: N={n}" for i, n in enumerate(values, 1)), flush=True)
    sysm = SysSampler(a.tegrastats_log)
    print(f"sweep N = {values} | {a.duration:.0f} s each | out {out}\n{sysm.describe()}", flush=True)
    ros = RosRecorder()
    summary = []
    try:
        for i, n in enumerate(values, 1):
            s = run_one(n, a, ros, sysm, out, case=i)
            if s:
                summary.append(s)
                write_consumption(out, summary)
                keys = sorted({k for r in summary for k in r}, key=lambda k: (k != "N", k))
                with open(os.path.join(out, "summary.csv"), "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=keys)
                    w.writeheader()
                    w.writerows(summary)
    except KeyboardInterrupt:
        print("\n[STOP] interrupted: report on the runs done so far", flush=True)
        subprocess.run(["pkill", "-INT", "-f", "thermal_tracking.launch.py"])
    finally:
        ros.close()
        sysm.close()
    if summary:
        report(out, weights)
        print(f"\nrestart C2 when done:  ros2 launch drone_target_tracker thermal_tracking.launch.py "
              f"detector_every:=<best N> {a.launch_args}")


if __name__ == "__main__":
    main()

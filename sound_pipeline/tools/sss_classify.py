#!/usr/bin/env python3
"""sss_classify.py - MobileNetV2 drone / no-drone on each of the 4 sources ODAS separates (UMA-8, no known-source drop).

Live (start this FIRST, then odaslive with the cfg made by tools/make_sss_cfg.py):
    python3 tools/sss_classify.py                      # odaslive on the same machine
    python3 tools/sss_classify.py --bind 0.0.0.0       # odaslive on the laptop (UMA-8 there), over Ethernet
  It listens on 127.0.0.1:9000 (SST tracked JSON) and 127.0.0.1:10000 (SSS separated audio, 4 ch int16 16 kHz),
  keeps 1 s of audio per tracked source and every 0.5 s prints, per active source:
    [12.5 s] slot 1  id 7   az  42.1  el  5.3  act 0.82 | p=0.93  DRONE
  and writes every decision to ~/dataset/sss_classify_<date_time>.csv

Offline (a raw file ODAS already wrote, e.g. ~/test_separated.raw; no tracking info -> every loud channel):
    python3 tools/sss_classify.py --file ~/test_separated.raw --channels 4 --fs 16000

Model and decision settings come from uma-8/config/drone_classifier.yaml (model_path, input_type, threshold,
smooth_n/m, win_s, hop_s, drone_index...). Override the model with --model.
"""
import argparse
import csv
import json
import math
import os
import socket
import sys
import threading
import time
from datetime import datetime

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "nodes"))
from mnv2_core import (ClassifierConfig, FeatureExtractor, Smoother, drone_probability,  # noqa: E402
                       load_model, resample)

GREEN, GREY, RESET = "\033[32m", "\033[90m", "\033[0m"


def load_params(path):
    d = yaml.safe_load(open(path))
    d = d.get("drone_classifier", d).get("ros__parameters", d)
    if d.get("resize") == [0]:
        d["resize"] = []
    return d


def az_el(x, y, z):
    n = math.sqrt(x * x + y * y + z * z) or 1.0
    return math.degrees(math.atan2(y, x)), math.degrees(math.asin(max(-1.0, min(1.0, z / n))))


class JsonStream:
    """Splits ODAS's back-to-back pretty-printed JSON objects."""

    def __init__(self):
        self.buf, self.depth, self.start, self.in_str, self.esc = "", 0, None, False, False

    def feed(self, text):
        out = []
        for ch in text:
            self.buf += ch
            if self.in_str:
                if self.esc:
                    self.esc = False
                elif ch == "\\":
                    self.esc = True
                elif ch == '"':
                    self.in_str = False
                continue
            if ch == '"':
                self.in_str = True
            elif ch == "{":
                if self.depth == 0:
                    self.start = len(self.buf) - 1
                self.depth += 1
            elif ch == "}" and self.depth:
                self.depth -= 1
                if self.depth == 0:
                    try:
                        out.append(json.loads(self.buf[self.start:]))
                    except ValueError:
                        pass
                    self.buf = ""
        return out


class SourceClassifier:
    """Per-slot 1 s buffers, per-track smoothing, one decision per slot every hop_s."""

    def __init__(self, cfg, model, fs, n_ch, min_activity, min_dbfs, csv_writer):
        self.cfg, self.model, self.fs, self.n_ch = cfg, model, fs, n_ch
        self.min_activity, self.min_dbfs = min_activity, min_dbfs
        self.feat = FeatureExtractor(cfg)
        self.win = int(round(cfg.win_s * fs))
        self.hop = int(round(cfg.hop_s * fs))
        self.buf = np.zeros((self.win, n_ch), np.float32)
        self.filled = np.zeros(n_ch, int)          # samples of the CURRENT track in each slot
        self.slot_id = [0] * n_ch
        self.since_hop = 0
        self.t = 0.0
        self.smooth = {}                           # track id -> Smoother
        self.label = {}                            # track id -> last printed is_drone
        self.tracks = None                         # latest SST frame: list of src dicts (None = offline)
        self.lock = threading.Lock()
        self.csv = csv_writer
        self.n_dec = 0
        self.results = {}                          # track id -> last decision (read by the ROS node)
        self.verbose = True

    def set_tracks(self, src):
        with self.lock:
            self.tracks = src

    def push(self, frames):
        """frames: (n, n_ch) float32 at fs."""
        n = len(frames)
        self.t += n / self.fs
        with self.lock:
            tracks = self.tracks
        for k in range(self.n_ch):                 # new track in a slot -> restart its buffer
            tid = int(tracks[k].get("id", 0)) if tracks is not None and k < len(tracks) else -1
            if tid != self.slot_id[k]:
                self.slot_id[k], self.filled[k] = tid, 0
        if n >= self.win:
            self.buf[:] = frames[-self.win:]
        else:
            self.buf = np.roll(self.buf, -n, axis=0)
            self.buf[-n:] = frames
        self.filled = np.minimum(self.filled + n, self.win)
        self.since_hop += n
        if self.since_hop >= self.hop:
            self.since_hop = 0
            self.decide(tracks)

    def decide(self, tracks):
        for k in range(self.n_ch):
            if self.filled[k] < self.win:
                continue
            w = self.buf[:, k]
            dbfs = 20 * np.log10(np.sqrt(np.mean(w ** 2)) + 1e-12)
            if tracks is not None:
                src = tracks[k] if k < len(tracks) else {}
                tid, act = int(src.get("id", 0)), float(src.get("activity", 0.0))
                if tid == 0 or act < self.min_activity:
                    continue
                az, el = az_el(float(src.get("x", 0)), float(src.get("y", 0)), float(src.get("z", 0)))
            else:                                   # offline: no tracking, use loud channels
                if dbfs < self.min_dbfs:
                    continue
                tid, act, az, el = k + 1, float("nan"), float("nan"), float("nan")
            t0 = time.perf_counter()
            try:
                y = resample(w - w.mean(), self.fs, self.cfg.model_fs)
                p = drone_probability(self.model(self.feat.tensor(y)), self.cfg)
            except Exception as e:  # noqa: BLE001 - keep the audio thread alive
                print(f"[ERR ] slot {k}: inference failed: {e}", flush=True)
                continue
            ms = (time.perf_counter() - t0) * 1000
            sm = self.smooth.setdefault(tid, Smoother(self.cfg))
            is_drone, _ = sm.update(p)
            self.n_dec += 1
            src_xyz = (tracks[k] if tracks is not None and k < len(tracks) else {})
            self.results[tid] = {"id": tid, "slot": k, "prob": round(float(p), 4), "is_drone": bool(is_drone),
                                 "rms_dbfs": round(float(dbfs), 1), "az": round(az, 1), "el": round(el, 1),
                                 "x": float(src_xyz.get("x", 0)), "y": float(src_xyz.get("y", 0)),
                                 "z": float(src_xyz.get("z", 0)), "t": time.time()}
            if not self.verbose:
                self.label[tid] = is_drone
                self.csv.writerow([f"{self.t:.2f}", k, tid, f"{az:.1f}", f"{el:.1f}", f"{act:.3f}",
                                   f"{dbfs:.1f}", f"{p:.4f}", int(is_drone), f"{ms:.1f}"])
                continue
            col = GREEN if is_drone else GREY
            change = "" if self.label.get(tid) == is_drone else "   <- changed"
            self.label[tid] = is_drone
            print(f"[{self.t:7.1f} s] slot {k}  id {tid:<4d} az {az:6.1f}  el {el:5.1f}  act {act:4.2f} "
                  f"| {dbfs:6.1f} dBFS  p={p:.2f}  {col}{'DRONE   ' if is_drone else 'no drone'}{RESET}"
                  f"  ({ms:.1f} ms){change}", flush=True)
            self.csv.writerow([f"{self.t:.2f}", k, tid, f"{az:.1f}", f"{el:.1f}", f"{act:.3f}",
                               f"{dbfs:.1f}", f"{p:.4f}", int(is_drone), f"{ms:.1f}"])


def serve(bind, port, name):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((bind, port))
    s.listen(1)
    print(f"[INFO] waiting for odaslive on {bind}:{port} ({name})", flush=True)
    return s


def tracked_loop(srv, clf):
    while True:
        conn, peer = srv.accept()
        print(f"[INFO] SST tracked stream connected from {peer[0]}", flush=True)
        js = JsonStream()
        with conn:
            while True:
                data = conn.recv(65536)
                if not data:
                    break
                for obj in js.feed(data.decode("utf-8", "ignore")):
                    if isinstance(obj.get("src"), list):
                        clf.set_tracks(obj["src"])
        print("[WARN] SST stream closed", flush=True)


def audio_loop(srv, clf, n_ch):
    frame_bytes = 2 * n_ch
    while True:
        conn, peer = srv.accept()
        print(f"[INFO] SSS audio stream connected from {peer[0]} ({n_ch} ch)", flush=True)
        pend = b""
        with conn:
            while True:
                data = conn.recv(65536)
                if not data:
                    break
                pend += data
                n = len(pend) // frame_bytes
                if n == 0:
                    continue
                x = np.frombuffer(pend[: n * frame_bytes], np.int16).reshape(n, n_ch).astype(np.float32) / 32768.0
                pend = pend[n * frame_bytes:]
                clf.push(x)
        print("[WARN] SSS stream closed", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--params", default=os.path.join(ROOT, "uma-8", "config", "drone_classifier.yaml"))
    ap.add_argument("--model")
    ap.add_argument("--channels", type=int, default=4, help="separated channels (= ODAS tracked slots)")
    ap.add_argument("--fs", type=int, default=16000, help="sample rate of the separated stream")
    ap.add_argument("--bind", default="127.0.0.1",
                    help="address to listen on: 127.0.0.1 = odaslive on this machine, 0.0.0.0 = odaslive on another "
                         "machine (e.g. the laptop with the UMA-8, over Ethernet)")
    ap.add_argument("--tracked-port", type=int, default=9000)
    ap.add_argument("--audio-port", type=int, default=10000)
    ap.add_argument("--min-activity", type=float, default=0.3, help="ignore ODAS tracks below this activity")
    ap.add_argument("--min-dbfs", type=float, default=-60.0, help="offline: ignore quieter channels")
    ap.add_argument("--file", help="offline: raw int16 file written by ODAS (sss separated/postfiltered)")
    ap.add_argument("--log-dir", default="~/dataset")
    a = ap.parse_args()

    p = load_params(a.params)
    if a.model:
        p["model_path"] = a.model
    cfg = ClassifierConfig.from_dict(p)
    model = load_model(cfg)
    print(f"[ OK ] model {cfg.model_path} ({model.name}) | input {cfg.input_type} | "
          f"window {cfg.win_s} s, decision every {cfg.hop_s} s, drone = {cfg.smooth_n} of last {cfg.smooth_m} "
          f"over p>={cfg.threshold}", flush=True)

    log_dir = os.path.expanduser(a.log_dir)
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, f"sss_classify_{datetime.now():%Y%m%d_%H%M%S}.csv")
    f = open(path, "w", newline="")
    w = csv.writer(f)
    w.writerow(["t_s", "slot", "track_id", "az_deg", "el_deg", "activity", "rms_dbfs", "prob", "is_drone", "ms"])
    clf = SourceClassifier(cfg, model, a.fs, a.channels, a.min_activity, a.min_dbfs, w)

    try:
        if a.file:
            raw = np.fromfile(os.path.expanduser(a.file), np.int16)
            n = raw.size // a.channels
            x = raw[: n * a.channels].reshape(n, a.channels).astype(np.float32) / 32768.0
            print(f"[INFO] {a.file}: {n / a.fs:.1f} s, {a.channels} ch @ {a.fs} Hz "
                  f"(per channel dBFS: {', '.join(f'{20*np.log10(np.sqrt(np.mean(x[:, k]**2))+1e-12):.0f}' for k in range(a.channels))})")
            step = int(a.fs * 0.008)               # same 8 ms blocks ODAS sends (hop 128 @ 16 kHz)
            for i in range(0, n, step):
                clf.push(x[i:i + step])
        else:
            t_srv = serve(a.bind, a.tracked_port, "SST tracked JSON")
            a_srv = serve(a.bind, a.audio_port, "SSS audio")
            threading.Thread(target=tracked_loop, args=(t_srv, clf), daemon=True).start()
            threading.Thread(target=audio_loop, args=(a_srv, clf, a.channels), daemon=True).start()
            print("[INFO] now start odaslive with the generated cfg. Ctrl+C to stop.", flush=True)
            while True:
                time.sleep(5)
                f.flush()
    except KeyboardInterrupt:
        pass
    finally:
        f.close()
        summary = {tid: ("DRONE" if d else "no drone") for tid, d in clf.label.items()}
        print(f"\n[DONE] {clf.n_dec} decisions | last label per track: {summary or 'none'}\n       csv: {path}")
        if hasattr(model, "close"):
            model.close()


if __name__ == "__main__":
    main()

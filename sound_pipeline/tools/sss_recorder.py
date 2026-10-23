#!/usr/bin/env python3
"""Record (and listen to) the sound MobileNet gets, BEFORE and AFTER the known-source drop.

Input = the 4 separated slots of /raw/sss (ODAS, 16 kHz), labelled with the /raw/sst tracks.
The drop (known point: track id + direction cone) comes from direction_gate_node (/gate/status,
/gate/known); a slot whose track is dropped is silenced in the AFTER files.
(gate:=false: drop read from target_direction_node /target/direction instead.)

Writes, until Ctrl+C, into ~/dataset/sss_rec/:
  sss_<date>_before_4ch.wav   the 4 slots exactly as MobileNet receives them
  sss_<date>_after_4ch.wav    same, dropped slots (known / echo) and empty slots silenced
  sss_<date>_after_mix.wav    mono sum of the kept slots = "what is left after the drop"
  sss_<date>_tracks.csv       per second: slot, id, az, el, activity, level, dropped
Live screen 5x/s: per slot the track, KEPT / DROPPED, and a level bar.

Listen live on the laptop (LAPTOP first: nc -l 6000 | aplay -t raw -f S16_LE -r 16000 -c 1):
  -p listen:=both     stereo: LEFT ear = before, RIGHT ear = after   (LAPTOP: aplay ... -c 2)
  -p listen:=after    the after-drop mix (default when listening)
  -p listen:=before   all 4 slots mixed, before the drop
  -p listen:=0..3     one slot, before the drop

    python3 tools/sss_recorder.py [--ros-args -p listen:=after]
    python3 tools/sss_recorder.py --selftest
"""
import csv
import json
import math
import os
import sys
import time
import wave

import numpy as np


def decode(raw, ch, fmt):
    """AudioFrame bytes -> int16 array [n, ch]."""
    fmt = str(fmt).lower()
    if "float" in fmt:
        x = np.clip(np.frombuffer(raw, "<f4") * 32767.0, -32768, 32767).astype("<i2")
    elif "32" in fmt:
        x = (np.frombuffer(raw, "<i4") >> 16).astype("<i2")
    else:
        x = np.frombuffer(raw, "<i2")
    if ch <= 0 or x.size % ch:
        return None
    return x.reshape(-1, ch)


def dbfs(x):
    if x.size == 0:
        return -120.0
    r = float(np.sqrt(np.mean((x.astype(np.float32) / 32768.0) ** 2)))
    return 20 * math.log10(r) if r > 1e-6 else -120.0


def mix(x):
    """[n, ch] int16 -> mono int16 (sum, clipped)."""
    return np.clip(x.astype(np.int32).sum(1), -32768, 32767).astype("<i2")


def _wav(path, ch, fs):
    w = wave.open(path, "wb")
    w.setnchannels(ch)
    w.setsampwidth(2)
    w.setframerate(fs)
    return w


class Recorder:
    """No ROS: slot state, drop state, WAV / CSV files."""

    def __init__(self, out_dir, stamp=None, min_activity=0.1, drop_max_age_s=2.0):
        os.makedirs(out_dir, exist_ok=True)
        self.base = os.path.join(out_dir, f"sss_{stamp or time.strftime('%Y%m%d_%H%M%S')}")
        self.min_act = min_activity
        self.drop_max_age = drop_max_age_s
        self.w = {}
        self.fs = self.ch = None
        self.frames = 0
        self.tracks = []              # per slot (id, az, el, act)
        self.drop_seen = {}           # track id -> last time target_direction_node listed it as dropped
        self.drop_hold = 1.5          # s: a dropped id stays dropped through short gaps in the list
        self.drop_t = None
        self.known = None             # (az, el, cone)
        self.gate_roles = None        # point mode: per-slot roles from direction_gate_node (/gate/status)
        self.gate_t = None
        self.lv_before = []
        self.lv_after = -120.0
        self.csv_f = open(self.base + "_tracks.csv", "w", newline="")
        self.csv = csv.writer(self.csv_f)
        self.csv.writerow(["t", "slot", "id", "az_deg", "el_deg", "activity", "level_dbfs", "dropped"])
        self.t0 = time.monotonic()
        self.last_csv = -1e9

    # -- inputs
    def on_tracks(self, sources):
        out = []
        for s in sources:
            try:
                x, y, z = float(s.x), float(s.y), float(s.z)
                n = math.sqrt(x * x + y * y + z * z) or 1.0
                out.append((int(s.id), math.degrees(math.atan2(y, x)),
                            math.degrees(math.asin(max(-1.0, min(1.0, z / n)))), float(s.activity)))
            except (AttributeError, TypeError, ValueError):
                out.append((0, 0.0, 0.0, 0.0))
        self.src = out                # ODAS lists only existing tracks, NOT one entry per slot

    def _pair(self, x):
        """Per-channel tracks, same rule as the classifier: k-th listed track <-> k-th channel with audio."""
        if self.ch is None:
            return
        rms = 20 * np.log10(np.sqrt(np.mean((x.astype(np.float32) / 32768.0) ** 2, axis=0)) + 1e-12)
        live = [k for k in range(self.ch) if rms[k] > -100.0]
        src = list(getattr(self, "src", []))
        if len(live) == len(src):
            per = [(0, 0.0, 0.0, 0.0)] * self.ch
            for k, c in enumerate(live):
                per[c] = src[k]
            self.tracks = per
        elif not self.tracks:
            self.tracks = [(0, 0.0, 0.0, 0.0)] * self.ch

    def on_direction(self, obj, now=None):
        k = obj.get("known_source_dropped") if isinstance(obj, dict) else None
        if not isinstance(k, dict):
            return
        now = time.monotonic() if now is None else now
        # /target/direction lists only the ids dropped in THAT frame (a track whose activity dips
        # below 0.1 for one frame is missing) -> remember each dropped id for drop_hold seconds
        for i in (k.get("dropped_track_ids") or []):
            self.drop_seen[int(i)] = now
        for i in [i for i, ts in self.drop_seen.items() if now - ts > 10.0]:
            del self.drop_seen[i]
        self.drop_t = now
        if k.get("azimuth_deg") is not None:
            self.known = (k["azimuth_deg"], k.get("elevation_deg"), k.get("cone_deg"))

    def on_gate_status(self, obj, now=None):
        """point mode: /gate/status {phase, slots: [{slot, id, role}]} -> roles per slot."""
        if isinstance(obj, dict) and isinstance(obj.get("slots"), list):
            self.gate_roles = {int(s["slot"]): s.get("role", "EMPTY") for s in obj["slots"]}
            self.gate_t = time.monotonic() if now is None else now

    def on_gate_known(self, obj):
        if isinstance(obj, dict) and obj.get("azimuth_deg") is not None:
            self.known = (obj["azimuth_deg"], obj.get("elevation_deg"), obj.get("cone_deg"))

    # -- decision per slot
    def state(self, i, now=None):
        """KEPT / DROPPED / EMPTY / NO-DROP-INFO for slot i."""
        now_ = time.monotonic() if now is None else now
        if self.gate_roles is not None and self.gate_t is not None and now_ - self.gate_t <= self.drop_max_age:
            if i >= len(self.tracks) or self.tracks[i][0] == 0 or self.tracks[i][3] < self.min_act:
                return "EMPTY"
            r = self.gate_roles.get(i, "EMPTY")
            return {"KNOWN": "DROPPED", "CAL": "DROPPED", "TARGET": "KEPT"}.get(r, "EMPTY")
        if i >= len(self.tracks):
            return "EMPTY"
        tid, _az, _el, act = self.tracks[i]
        if tid == 0 or act < self.min_act:
            return "EMPTY"
        now = time.monotonic() if now is None else now
        if self.drop_t is None or now - self.drop_t > self.drop_max_age:
            return "NO-DROP-INFO"
        ts = self.drop_seen.get(tid)
        return "DROPPED" if ts is not None and now - ts <= self.drop_hold else "KEPT"

    def _open(self, fs, ch):
        self.fs, self.ch = fs, ch
        self.w["before"] = _wav(self.base + "_before_4ch.wav", ch, fs)
        self.w["after"] = _wav(self.base + "_after_4ch.wav", ch, fs)
        self.w["mix"] = _wav(self.base + "_after_mix.wav", 1, fs)
        self.lv_before = [-120.0] * ch

    def on_frame(self, x, fs, now=None):
        """x: int16 [n, ch] as received by MobileNet. Returns (after [n, ch], after_mix [n])."""
        if self.ch is None:
            self._open(fs, x.shape[1])
        if x.shape[1] != self.ch:
            return None, None
        self._pair(x)
        keep = [self.state(i, now) in ("KEPT", "NO-DROP-INFO") for i in range(self.ch)]
        after = x.copy()
        for i, k in enumerate(keep):
            if not k:
                after[:, i] = 0
        am = mix(after)
        self.w["before"].writeframes(np.ascontiguousarray(x).tobytes())
        self.w["after"].writeframes(np.ascontiguousarray(after).tobytes())
        self.w["mix"].writeframes(am.tobytes())
        for i in range(self.ch):
            self.lv_before[i] = 0.7 * self.lv_before[i] + 0.3 * dbfs(x[:, i])
        self.lv_after = 0.7 * self.lv_after + 0.3 * dbfs(am)
        self.frames += x.shape[0]
        t = time.monotonic() - self.t0
        if t - self.last_csv >= 1.0:
            self.last_csv = t
            for i in range(self.ch):
                tid, az, el, act = self.tracks[i] if i < len(self.tracks) else (0, 0.0, 0.0, 0.0)
                self.csv.writerow([f"{t:.1f}", i, tid, f"{az:.1f}", f"{el:.1f}", f"{act:.2f}",
                                   f"{self.lv_before[i]:.1f}", int(self.state(i, now) == "DROPPED")])
            self.csv_f.flush()
        return after, am

    def screen(self, now=None):
        dur = self.frames / self.fs if self.fs else 0.0
        k = "" if self.known is None else f"   known az {self.known[0]:+.0f} el {self.known[1] or 0:+.0f} cone {self.known[2]}"
        lines = [f"MobileNet input, before / after the known-source drop   {dur:7.1f} s   (Ctrl+C to stop){k}",
                 f"-> {self.base}_before_4ch.wav | _after_4ch.wav | _after_mix.wav", ""]
        for i in range(self.ch or 0):
            tid, az, el, act = self.tracks[i] if i < len(self.tracks) else (0, 0.0, 0.0, 0.0)
            st = self.state(i, now)
            lv = self.lv_before[i]
            bar = "#" * max(0, min(40, int((lv + 90) / 1.5)))
            who = f"id {tid:<4d} az {az:+6.1f} el {el:+5.1f}" if st != "EMPTY" else "(empty slot)".ljust(28)
            lines.append(f"slot {i}  {who}  {st:<12s} | {lv:6.1f} dBFS {bar}")
        bar = "#" * max(0, min(40, int((self.lv_after + 90) / 1.5)))
        lines += ["", f"AFTER DROP (kept slots mixed)              {self.lv_after:6.1f} dBFS {bar}"]
        if any(self.state(i, now) == "NO-DROP-INFO" for i in range(self.ch or 0)):
            lines.append("NO-DROP-INFO: /target/direction not received -> is target_direction_node running?")
        return "\n".join(lines)

    def close(self):
        for w in self.w.values():
            w.close()
        self.csv_f.close()
        return self.base


def run_node():
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
    Audio = Sst = None
    try:
        from audio_utils_msgs.msg import AudioFrame as Audio
    except ImportError:
        pass
    for mod in ("odas_ros_msgs.msg", "odas_ros.msg"):
        try:
            Sst = __import__(mod, fromlist=["OdasSstArrayStamped"]).OdasSstArrayStamped
            break
        except (ImportError, AttributeError):
            pass
    if Audio is None or Sst is None:
        sys.exit("audio_utils_msgs / odas_ros_msgs not found: source ~/ros2_ws/install/setup.bash first")

    class RecNode(Node):
        def __init__(self):
            super().__init__("sss_recorder")

            def p(name, default):
                if not self.has_parameter(name):
                    self.declare_parameter(name, default)
                return self.get_parameter(name).value

            self.listen = str(p("listen", "none"))
            self.rec = Recorder(os.path.expanduser(p("out_dir", "~/dataset/sss_rec")))
            self.tty = sys.stderr if self.listen != "none" else sys.stdout   # stdout = audio when listening
            self.create_subscription(Audio, p("topic", "/raw/sss"), self.on_sss, 50)
            self.create_subscription(Sst, p("sst_topic", "/raw/sst"), lambda m: self.rec.on_tracks(m.sources), 10)
            if bool(p("gate", True)):
                # point mode: the drop is done by direction_gate_node -> its roles and known point
                self.create_subscription(String, "/gate/status", lambda m: self._js(m, self.rec.on_gate_status), 10)
                self.create_subscription(String, "/gate/known", lambda m: self._js(m, self.rec.on_gate_known), 10)
            else:
                self.create_subscription(String, p("direction_topic", "/target/direction"), self.on_dir, 10)
            self.create_timer(0.2, self.draw)

        def _js(self, msg, fn):
            try:
                fn(json.loads(msg.data))
            except ValueError:
                pass

        def on_dir(self, msg):
            try:
                self.rec.on_direction(json.loads(msg.data))
            except ValueError:
                pass

        def on_sss(self, msg):
            x = decode(bytes(msg.data), int(getattr(msg, "channel_count", 0) or 0), getattr(msg, "format", "signed_16"))
            if x is None:
                return
            after, am = self.rec.on_frame(x, int(getattr(msg, "sampling_frequency", 16000) or 16000))
            if after is None or self.listen == "none":
                return
            if self.listen == "both":
                # stereo: LEFT = before (all slots mixed), RIGHT = after the drop
                out = np.ascontiguousarray(np.stack([mix(x), am], 1))
            elif self.listen == "after":
                out = am
            elif self.listen == "before":
                out = mix(x)
            elif self.listen.isdigit() and int(self.listen) < x.shape[1]:
                out = np.ascontiguousarray(x[:, int(self.listen)])
            else:
                return
            try:
                sys.stdout.buffer.write(out.tobytes())
                sys.stdout.buffer.flush()
            except BrokenPipeError:
                self.listen = "none"

        def draw(self):
            if self.rec.fs:
                self.tty.write("\033[2J\033[H" + self.rec.screen() + "\n")
                self.tty.flush()

    rclpy.init()
    node = RecNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception:  # noqa: BLE001
        if rclpy.ok():                 # a real error while running -> show it
            raise
        # Ctrl+C mid-message: rclpy raises RuntimeError/ExternalShutdownException after shutdown; ignore
    finally:
        base = node.rec.close()
        print(f"\n[sss_rec] saved {base}_before_4ch.wav, _after_4ch.wav, _after_mix.wav, _tracks.csv "
              f"({node.rec.frames / (node.rec.fs or 1):.1f} s)", file=sys.stderr, flush=True)
        try:
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:  # noqa: BLE001
            pass


def selftest():
    import tempfile
    d = tempfile.mkdtemp()
    r = Recorder(d, stamp="test")
    fs, n, N = 16000, 384, 50
    t = np.arange(n * N) / fs
    # slot 0 = known source leftover (dropped), slot 1 = echo (dropped), slot 2 = target (kept), slot 3 empty
    sig = np.stack([np.sin(2 * np.pi * 300 * t) * 6000, np.sin(2 * np.pi * 300 * t) * 2000,
                    np.sin(2 * np.pi * 1000 * t) * 4000, np.zeros_like(t)], 1).astype("<i2")
    S = type("S", (), {})

    def src(i, az, act):
        s = S(); s.id = i; s.x = math.cos(math.radians(az)); s.y = math.sin(math.radians(az)); s.z = 0.1; s.activity = act
        return s
    r.on_tracks([src(9, -18, 0.9), src(6, 168, 0.9), src(12, 60, 0.9)])   # ODAS lists existing tracks only
    # before drop info: nothing silenced (NO-DROP-INFO)
    a0, _ = r.on_frame(sig[:n], fs, now=0.0)
    assert (a0[:, :3] == sig[:n, :3]).all()
    r.on_direction({"known_source_dropped": {"azimuth_deg": -18, "elevation_deg": 10, "cone_deg": 20,
                                             "dropped_track_ids": [9, 6]}}, now=0.0)
    for k in range(1, N):
        after, am = r.on_frame(sig[k * n:(k + 1) * n], fs, now=0.1)
        assert not after[:, 0].any() and not after[:, 1].any() and (after[:, 2] == sig[k * n:(k + 1) * n, 2]).all()
        assert (am == sig[k * n:(k + 1) * n, 2]).all()
    print(r.screen(now=0.1))
    base = r.close()
    with wave.open(base + "_before_4ch.wav") as w:
        assert w.getnchannels() == 4 and w.getnframes() == n * N
        assert (np.frombuffer(w.readframes(n * N), "<i2").reshape(-1, 4) == sig).all()
    with wave.open(base + "_after_mix.wav") as w:
        m = np.frombuffer(w.readframes(n * N), "<i2")
        assert w.getnchannels() == 1 and (m[n:] == sig[n:, 2]).all()
    rows = list(csv.reader(open(base + "_tracks.csv")))
    assert rows[0][-1] == "dropped" and len(rows) >= 5
    assert r.state(0, now=10.0) == "NO-DROP-INFO"                      # stale drop info
    # flicker: the drop list misses id 9 for one message -> still DROPPED (held 1.5 s)
    r.on_direction({"known_source_dropped": {"dropped_track_ids": [6]}}, now=0.5)
    assert r.state(0, now=0.6) == "DROPPED" and r.state(1, now=0.6) == "DROPPED"
    assert r.state(0, now=2.2) == "KEPT"                               # really gone from the list
    # point mode: roles from /gate/status
    r.on_gate_status({"phase": "RUN", "slots": [{"slot": 0, "role": "KNOWN"}, {"slot": 1, "role": "KNOWN"},
                                                {"slot": 2, "role": "TARGET"}, {"slot": 3, "role": "EMPTY"}]}, now=3.0)
    assert [r.state(i, now=3.1) for i in range(4)] == ["DROPPED", "DROPPED", "KEPT", "EMPTY"]
    print("selftest PASSED")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        run_node()

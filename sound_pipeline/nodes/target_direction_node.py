#!/usr/bin/env python3
"""ROS 2 node: log the target drone's direction from the ODAS SST tracker (x, y, z).

Subscribes
  /gate/sst              odas_ros_msgs/OdasSstArrayStamped  tracks AFTER the known-point drop
  /gate/known            std_msgs/String JSON                known point (its cone is drawn / dropped)
  /classifier/tracks     std_msgs/String JSON                classifier per track id:
                                                             is_drone, prob, rms_dbfs (marks DRONE, level gate)
Publishes
  /target/direction      std_msgs/String JSON  one entry per active target track
  /target/markers        visualization_msgs/MarkerArray (frame "odas") for RViz:
                         KNOWN red, ECHO orange, DRONE green, other targets yellow
Logs
  console  mode:=drone (default): the known source is shown once as DROPPED (and again if it moves
           or disappears); any track inside its cone is listed as dropped and never localised;
           ONLY MobileNet-confirmed drones get live x y z at rate_hz (10 Hz);
           one "no drone detected" line when there is none, then silence
           [   12.4 s] DRONE id 37   x +0.620  y -0.630  z +0.460   az   -45.5  el +27.4  p=0.91   -48.0 dBFS
  console  mode:=changes: every target, only when it changes:
           [  12.4 s] id 37   az   -45.3  el +27.7  act 0.96  x +0.62 y -0.63 z +0.46  DRONE p=0.91  <- moved 6.2°
           events: new | moved >= change_deg (5°) | now DRONE | no longer drone | LOST (> lost_s, 1 s)
           changes_only:=false prints every target every log_every_s (0.5 s)
  CSV      ~/dataset/target_direction_<date_time>.csv  (every accepted frame)

Direction: az = atan2(y, x), el = asin(z / |v|), in the ODAS array frame ("odas").
A track is a target when its activity >= min_activity and it is outside the known
source's cone. It is marked DRONE when /fusion/drone_targets reports the same track id
(or, if ids are missing, a bearing within match_deg) with prob >= min_prob.

Run (ROS 2 + odas_ros workspace sourced):
    python3 nodes/target_direction_node.py
    python3 nodes/target_direction_node.py --ros-args -p sst_topic:=/gate/sst -p min_activity:=0.2
Offline self-test:
    python3 nodes/target_direction_node.py --selftest
"""

import csv
import json
import math
import os
import sys
import time

AZ_KEYS = ("azimuth_deg", "az_deg", "azimuth", "az", "bearing_deg", "bearing")
ID_KEYS = ("id", "track_id", "tid", "sst_id")
PROB_KEYS = ("drone_prob", "prob", "p_drone", "score", "confidence", "conf")
LIST_KEYS = ("sources", "targets", "tracks", "items", "drones")


def circ_diff(a, b):
    return abs(((a - b + 180.0) % 360.0) - 180.0)


def items_of(obj):
    if isinstance(obj, list):
        return [o for o in obj if isinstance(o, dict)]
    if isinstance(obj, dict):
        for k in LIST_KEYS:
            if isinstance(obj.get(k), list):
                return [o for o in obj[k] if isinstance(o, dict)]
        vals = list(obj.values())
        if vals and all(isinstance(v, dict) for v in vals):
            return [dict(v, id=v.get("id", k)) for k, v in obj.items()]
        return [obj]
    return []


def num(d, keys):
    for k in keys:
        if k in d:
            try:
                v = float(d[k])
                if math.isfinite(v):
                    return v
            except (TypeError, ValueError):
                pass
    return None


def az_of(d):
    a = num(d, AZ_KEYS)
    if a is not None:
        return a % 360.0
    if "x" in d and "y" in d:
        try:
            return math.degrees(math.atan2(float(d["y"]), float(d["x"]))) % 360.0
        except (TypeError, ValueError):
            return None
    return None


def direction(x, y, z):
    n = math.sqrt(x * x + y * y + z * z)
    if n < 1e-9:
        return None, None
    az = math.degrees(math.atan2(y, x))                    # -180..180
    el = math.degrees(math.asin(max(-1.0, min(1.0, z / n))))
    return az, el


class TargetDirectionCore:
    """No ROS: turns SST tracks + known + classifier into target directions."""

    def __init__(self, min_activity=0.1, known_cone_deg=20.0, min_prob=0.5,
                 match_deg=15.0, drone_max_age=1.5, min_dbfs=None, known_lock_s=5.0,
                 learn_s=15.0, residual_cone_deg=20.0, residual_min_frames=20, opposite_cone_deg=25.0,
                 known_id_stick=True, cone_3d=False, upper_only=True, max_el_deg=90.0):
        self.min_activity = min_activity
        # drop zone shape: False = azimuth slice (|az - known_az| <= cone at ANY elevation, pole to pole)
        #                  True  = real 3D cone (angle to the known direction <= cone, az AND el)
        self.cone_3d = cone_3d
        # the UMA-8 is a flat array: it cannot tell "above" from "below" (mirror image through the
        # array plane), so every direction is folded into the upper hemisphere (el = |el|)
        self.upper_only = upper_only
        # zenith gate: a flat array resolves elevation badly near the top; reflections / diffuse
        # leftovers of the known source pile up there (el 60-80). Tracks above max_el are dropped.
        self.max_el = max_el_deg
        # ID tracker drop: a track id once seen inside the known cone stays dropped for its
        # whole life, even if its direction drifts out of the cone
        self.known_id_stick = known_id_stick
        self.known_ids = {}       # id -> last time seen
        self.known_cone = known_cone_deg
        self.min_prob = min_prob
        self.match_deg = match_deg
        self.drone_max_age = drone_max_age
        self.known_az = None
        self.known_el = None
        self.known_stamp = None
        # the known source does not move: lock its direction after known_lock_s of
        # the known topic (circular mean), so a nearby target cannot drag the drop cone
        self.known_lock_s = known_lock_s
        self.known_t0 = None
        self.known_samples = []
        self.known_locked = False
        # residual directions of the known source on the nulled path (echo off a wall,
        # null leakage): learned during the first learn_s of tracking, when ONLY the known
        # source plays, then dropped like the known source itself
        self.learn_s = learn_s
        self.residual_cone = residual_cone_deg
        self.residual_min_frames = residual_min_frames
        self.learn_t0 = None
        self.learning_done = learn_s <= 0
        self.residual_bins = []   # [az, el, frames]
        self.residuals = []       # [(az, el)] after learning
        # the known source's echo was measured twice at ~180° from it (az +8 -> -165,
        # az +2 -> -169): always drop that opposite cone (0 = off)
        self.opposite_cone = opposite_cone_deg
        self.dropped = []         # tracks dropped this frame because they sit on the known source
        self.min_dbfs = min_dbfs  # level gate on the separated audio (classifier rms_dbfs)
        self.drones = []          # [(stamp, id or None, az or None, prob)]
        self.cls = {}             # id -> (stamp, is_drone, prob, rms_dbfs) from /classifier/tracks

    # ---- persistence (valid for one calibration)
    def to_dict(self):
        return {"known_az": self.known_az, "known_el": self.known_el,
                "residuals": [[round(a, 1), round(e, 1)] for a, e in self.residuals]}

    def from_dict(self, d):
        if d.get("known_az") is not None:
            self.known_az, self.known_el = float(d["known_az"]), d.get("known_el")
            self.known_locked = True
        self.residuals = [(float(a), float(e)) for a, e in d.get("residuals", [])]
        self.learning_done = True

    def learning(self, now):
        if self.learning_done:
            return False
        return self.learn_t0 is None or now - self.learn_t0 < self.learn_s

    def learn_left(self, now):
        if self.learning_done or self.learn_t0 is None:
            return 0.0 if self.learning_done else self.learn_s
        return max(0.0, self.learn_s - (now - self.learn_t0))

    def in_known(self, az, el):
        if self.known_az is None:
            return False
        if self.cone_3d and self.known_el is not None:
            return ang_dist(az, el, self.known_az, self.known_el) <= self.known_cone
        return circ_diff(az % 360.0, self.known_az) <= self.known_cone

    def _in_residual(self, az, el):
        return any(ang_dist(az, el, ra, re) <= self.residual_cone for ra, re in self.residuals)

    def on_known(self, obj, now=None):
        for it in items_of(obj):
            a = az_of(it)
            if a is None:
                continue
            el = num(it, ("elevation_deg", "el_deg", "elevation", "el"))
            if el is None and all(k in it for k in ("x", "y", "z")):
                try:
                    el = direction(float(it["x"]), float(it["y"]), float(it["z"]))[1]
                except (TypeError, ValueError):
                    el = None
            if self.upper_only and el is not None:
                el = abs(el)
            self.known_stamp = now
            if self.known_locked:
                return                       # fixed direction, only refresh "still seen"
            if self.known_lock_s <= 0 or now is None:
                self.known_az, self.known_el = a, el
                return
            if self.known_t0 is None:
                self.known_t0 = now
            self.known_samples.append((a, el if el is not None else 0.0))
            sx = sum(math.cos(math.radians(x)) for x, _ in self.known_samples)
            sy = sum(math.sin(math.radians(x)) for x, _ in self.known_samples)
            self.known_az = math.degrees(math.atan2(sy, sx)) % 360.0
            els = sorted(e for _, e in self.known_samples)
            self.known_el = els[len(els) // 2]
            if now - self.known_t0 >= self.known_lock_s:
                self.known_locked = True
            return

    def on_drones(self, obj, now):
        """Accepts /classifier/tracks ([{id, prob, is_drone, rms_dbfs, ...}])
        or /fusion/drone_targets-style lists."""
        out = []
        for it in items_of(obj):
            p = num(it, PROB_KEYS)
            tid = it.get(next((k for k in ID_KEYS if k in it), ""), None)
            try:
                tid = int(str(tid).lstrip("nr")) if tid is not None else None
            except (TypeError, ValueError):
                tid = None
            if tid is not None and ("is_drone" in it or "rms_dbfs" in it):
                self.cls[tid] = (now, bool(it.get("is_drone")), p, num(it, ("rms_dbfs", "dbfs")))
                continue
            p = 1.0 if p is None else p
            if p < self.min_prob:
                continue
            out.append((now, tid, az_of(it), p))
        if out or not self.cls:
            self.drones = out

    def _cls_of(self, tid, now):
        c = self.cls.get(tid)
        if c is None or now - c[0] > self.drone_max_age:
            return None
        return c

    def _drone_match(self, tid, az, now):
        c = self._cls_of(tid, now)
        if c is not None:
            _, is_drone, p, _ = c
            return (p if p is not None else 1.0) if is_drone else None
        best = None
        for stamp, did, daz, p in self.drones:
            if now - stamp > self.drone_max_age:
                continue
            if did is not None and did == tid:
                return p
            if did is None and daz is not None and circ_diff(daz % 360.0, az % 360.0) <= self.match_deg:
                best = max(best or 0.0, p)
        return best

    def on_sst(self, sources, now):
        """sources: iterable of objects/dicts with id, x, y, z, activity."""
        targets = []
        self.dropped = []
        if not self.learning_done:
            if self.learn_t0 is None:
                self.learn_t0 = now
            elif now - self.learn_t0 >= self.learn_s:
                self.learning_done = True
                self.residuals = [(a, e) for a, e, n in self.residual_bins if n >= self.residual_min_frames]
        learning = not self.learning_done
        for s in sources:
            g = (lambda k: s[k]) if isinstance(s, dict) else (lambda k: getattr(s, k))
            try:
                tid, x, y, z = int(g("id")), float(g("x")), float(g("y")), float(g("z"))
                act = float(g("activity"))
            except (KeyError, AttributeError, TypeError, ValueError):
                continue
            if tid == 0 or act < self.min_activity:      # ODAS uses id 0 for an empty slot
                continue
            if self.upper_only:
                z = abs(z)
            az, el = direction(x, y, z)
            if az is None:
                continue
            if el > self.max_el:
                self.dropped.append({"id": tid, "azimuth_deg": round(az, 1), "elevation_deg": round(el, 1),
                                     "activity": round(act, 2), "why": "zenith"})
                continue
            if self.in_known(az, el):
                self.dropped.append({"id": tid, "azimuth_deg": round(az, 1), "elevation_deg": round(el, 1),
                                     "activity": round(act, 2), "why": "known"})
                if self.known_id_stick and self.known_locked:
                    self.known_ids[tid] = now
                continue
            if self.known_id_stick and tid in self.known_ids:
                self.known_ids[tid] = now
                self.dropped.append({"id": tid, "azimuth_deg": round(az, 1), "elevation_deg": round(el, 1),
                                     "activity": round(act, 2), "why": "known-id"})
                continue
            if (self.known_az is not None and self.opposite_cone > 0
                    and circ_diff(az % 360.0, (self.known_az + 180.0) % 360.0) <= self.opposite_cone):
                self.dropped.append({"id": tid, "azimuth_deg": round(az, 1), "elevation_deg": round(el, 1),
                                     "activity": round(act, 2), "why": "opposite"})
                continue
            if learning:                                   # known source alone: everything is its residual
                for b in self.residual_bins:
                    if ang_dist(az, el, b[0], b[1]) <= self.residual_cone / 2:
                        b[0] += (az - b[0]) * 0.05 if abs(az - b[0]) < 180 else 0.0
                        b[1] += (el - b[1]) * 0.05
                        b[2] += 1
                        break
                else:
                    self.residual_bins.append([az, el, 1])
                self.dropped.append({"id": tid, "azimuth_deg": round(az, 1), "elevation_deg": round(el, 1),
                                     "activity": round(act, 2), "why": "learning"})
                continue
            if self._in_residual(az, el):
                self.dropped.append({"id": tid, "azimuth_deg": round(az, 1), "elevation_deg": round(el, 1),
                                     "activity": round(act, 2), "why": "residual"})
                continue
            c = self._cls_of(tid, now)
            dbfs = c[3] if c is not None else None
            if self.min_dbfs is not None and dbfs is not None and dbfs < self.min_dbfs:
                continue                                  # too quiet: noise-floor ghost
            p = self._drone_match(tid, az, now)
            targets.append({"id": tid, "x": round(x, 3), "y": round(y, 3), "z": round(z, 3),
                            "azimuth_deg": round(az, 1), "elevation_deg": round(el, 1),
                            "activity": round(act, 2), "drone": p is not None,
                            "drone_prob": None if p is None else round(p, 2),
                            "rms_dbfs": None if dbfs is None else round(dbfs, 1),
                            "cls_prob": None if c is None or c[2] is None else round(c[2], 2)})
        for k in [k for k, ts in self.known_ids.items() if now - ts > 3.0]:
            del self.known_ids[k]                          # ODAS never reuses an id: forget dead ones
        targets.sort(key=lambda t: (not t["drone"], -t["activity"]))
        return targets


def fmt(t):
    tag = f"DRONE p={t['drone_prob']:.2f}" if t["drone"] else "target    "
    return (f"[TARGET] id={t['id']:<3d} {tag} x={t['x']:+.2f} y={t['y']:+.2f} z={t['z']:+.2f} "
            f"az={t['azimuth_deg']:+6.1f}° el={t['elevation_deg']:+5.1f}° act={t['activity']:.2f}")


def ang_dist(az1, el1, az2, el2):
    """Great-circle angle between two directions, degrees."""
    a1, e1, a2, e2 = map(math.radians, (az1, el1, az2, el2))
    c = math.sin(e1) * math.sin(e2) + math.cos(e1) * math.cos(e2) * math.cos(a1 - a2)
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


class ChangeFilter:
    """Report a track only when it appears, moves >= change_deg, changes DRONE status, or is lost."""

    def __init__(self, change_deg=5.0, lost_s=1.0):
        self.change_deg = change_deg
        self.lost_s = lost_s
        self.last = {}      # id -> (az, el, drone) last REPORTED
        self.seen = {}      # id -> last time seen

    def update(self, targets, now):
        events = []
        for t in targets:
            tid = t["id"]
            self.seen[tid] = now
            prev = self.last.get(tid)
            if prev is None:
                why = "new"
            elif t["drone"] != prev[2]:
                why = "now DRONE" if t["drone"] else "no longer drone"
            else:
                d = ang_dist(t["azimuth_deg"], t["elevation_deg"], prev[0], prev[1])
                why = f"moved {d:.1f}°" if d >= self.change_deg else None
            if why:
                self.last[tid] = (t["azimuth_deg"], t["elevation_deg"], t["drone"])
                events.append((t, why))
        for tid in [k for k, ts in self.seen.items() if now - ts > self.lost_s]:
            prev = self.last.pop(tid, None)
            self.seen.pop(tid)
            if prev is not None:
                events.append(({"id": tid, "azimuth_deg": prev[0], "elevation_deg": prev[1],
                                "drone": prev[2]}, "lost"))
        return events


def fmt_event(t, why, t_rel):
    if why == "lost":
        return (f"[{t_rel:7.1f} s] id {t['id']:<4d} LOST      last az {t['azimuth_deg']:+7.1f}  "
                f"el {t['elevation_deg']:+5.1f}")
    lvl = f"{t['rms_dbfs']:6.1f} dBFS" if t.get("rms_dbfs") is not None else "   -- dBFS"
    if t["drone"]:
        tag = f"p={t['drone_prob']:.2f}  DRONE   "
    elif t.get("cls_prob") is not None:
        tag = f"p={t['cls_prob']:.2f}  no drone"
    else:
        tag = "p=--    target  "
    return (f"[{t_rel:7.1f} s] id {t['id']:<4d} az {t['azimuth_deg']:+7.1f}  el {t['elevation_deg']:+5.1f}  "
            f"act {t['activity']:.2f}  x {t['x']:+.2f} y {t['y']:+.2f} z {t['z']:+.2f} | {lvl}  {tag}  <- {why}")


class DroneDisplay:
    """mode:=drone — live x, y, z of MobileNet-confirmed drones only; one line when none.
    Also shows the known source being dropped: its direction (when it appears or moves)
    and every track that ODAS starts inside its cone (dropped, never localised)."""

    def __init__(self, rate_hz=10.0, change_deg=5.0, known_timeout_s=3.0):
        self.period = 1.0 / rate_hz if rate_hz > 0 else 0.0
        self.last = -1e9
        self.had_drone = None     # None = nothing printed yet
        self.change_deg = change_deg
        self.known_timeout = known_timeout_s
        self.known_shown = None   # (az, el) last printed, or "lost"
        self.dropped_shown = {}   # id -> last time printed

    def known_lines(self, core, now, t_rel):
        lines = []
        cone = core.known_cone
        if core.known_az is None or core.known_stamp is None:
            return lines
        if now - core.known_stamp > self.known_timeout:
            if self.known_shown not in (None, "lost"):
                lines.append(f"[{t_rel:7.1f} s] KNOWN source not seen for {self.known_timeout:.0f} s")
                self.known_shown = "lost"
            return lines
        az = ((core.known_az + 180.0) % 360.0) - 180.0
        el = core.known_el if core.known_el is not None else 0.0
        prev = self.known_shown
        if prev in (None, "lost") or ang_dist(az, el, prev[0], prev[1]) >= self.change_deg:
            lines.append(f"[{t_rel:7.1f} s] KNOWN source  az {az:+7.1f}  el {el:+5.1f}  "
                         f"-> DROPPED (cone {cone:.0f}°), not localised")
            self.known_shown = (az, el)
        for d in core.dropped:
            if now - self.dropped_shown.get(d["id"], -1e9) > 5.0:
                lines.append(f"[{t_rel:7.1f} s] dropped track id {d['id']:<4d} az {d['azimuth_deg']:+7.1f}  "
                             f"el {d['elevation_deg']:+5.1f}  (inside known cone)")
            self.dropped_shown[d["id"]] = now
        return lines

    def update(self, targets, now, t_rel, core=None):
        drones = [t for t in targets if t["drone"]]
        lines = self.known_lines(core, now, t_rel) if core is not None else []
        if drones:
            if self.had_drone is not True or now - self.last >= self.period:
                self.last = now
                for t in drones:
                    lvl = f"{t['rms_dbfs']:6.1f} dBFS" if t.get("rms_dbfs") is not None else ""
                    lines.append(f"[{t_rel:7.1f} s] DRONE id {t['id']:<4d} x {t['x']:+.3f}  y {t['y']:+.3f}  "
                                 f"z {t['z']:+.3f}   az {t['azimuth_deg']:+7.1f}  el {t['elevation_deg']:+5.1f}  "
                                 f"p={t['drone_prob']:.2f}  {lvl}")
            self.had_drone = True
        elif self.had_drone is not False:
            lines.append(f"[{t_rel:7.1f} s] no drone detected")
            self.had_drone = False
        return drones, lines


def _unit(az, el, r=1.0):
    a, e = math.radians(az), math.radians(el)
    return (r * math.cos(e) * math.cos(a), r * math.cos(e) * math.sin(a), r * math.sin(e))


def update_trails(trails, targets, dropped=(), keep=60, forget=30):
    """trails: {id: {"pts": [(x,y,z)...], "miss": n, "kind": "drone|target|dropped"}}.
    Adds the current point of every track, forgets a track unseen for `forget` updates."""
    seen = set()
    for t in list(targets) + [dict(d, kind="dropped") for d in dropped]:
        tid = int(t["id"])
        seen.add(tid)
        kind = t.get("kind") or ("drone" if t.get("drone") else "target")
        tr = trails.setdefault(tid, {"pts": [], "miss": 0, "kind": kind})
        tr["kind"], tr["miss"] = kind, 0
        tr["pts"].append(_unit(t["azimuth_deg"], t["elevation_deg"], 1.0))
        del tr["pts"][:-keep]
    for tid in list(trails):
        if tid not in seen:
            trails[tid]["miss"] += 1
            if trails[tid]["miss"] > forget:
                del trails[tid]
    return trails


KIND_RGB = {"drone": (0.1, 0.9, 0.2), "target": (0.95, 0.85, 0.1), "dropped": (0.9, 0.3, 0.3)}


STATIC_NS = ("sphere", "ring")      # sent once per second, never expire


def build_markers(core, targets, trails=None, with_static=True, labels=True):
    """RViz scene as plain dicts (converted to visualization_msgs in the node).
    Grey ring + 0/90/180/270 labels, KNOWN arrow (red) + its cone, ECHO arrow (orange)
    + cone, learned residuals (orange), every target arrow: green = MobileNet DRONE,
    yellow = tracked but not a drone (e.g. a tone)."""
    m = []

    def arrow(ns, i, az, el, rgb, length=1.0, width=0.03):
        m.append(dict(ns=ns, id=i, type="arrow", points=[(0, 0, 0), _unit(az, el, length)],
                      scale=(width, 2.2 * width, 0.12), rgba=(*rgb, 1.0)))

    def text(ns, i, pos, txt, rgb, size=0.09):
        m.append(dict(ns=ns, id=i, type="text", pos=pos, text=txt, scale=(0, 0, size), rgba=(*rgb, 1.0)))

    def arc(ns, i, az0, half, rgb, r=1.0):
        pts = [_unit(az0 + d, 0.0, r) for d in range(int(-half), int(half) + 1, 2)]
        m.append(dict(ns=ns, id=i, type="line", points=[(0, 0, 0)] + pts + [(0, 0, 0)],
                      scale=(0.015, 0, 0), rgba=(*rgb, 0.8)))

    # omnisphere (unit sphere around the UMA-8, like ODAS Studio): parallels + meridians
    seg = []
    up = getattr(core, "upper_only", False)
    for el in ((30, 60) if up else (-60, -30, 30, 60)):
        for d in range(0, 360, 6):
            seg += [_unit(d, el), _unit(d + 6, el)]
    for az in range(0, 360, 30):
        for e in range(0 if up else -90, 90, 6):
            seg += [_unit(az, e), _unit(az, e + 6)]
    # omnisphere: translucent shell + light parallels / meridians (upper dome only for the flat UMA-8)
    if up:
        tri = []
        for e in range(0, 90, 10):
            for d in range(0, 360, 10):
                a0, a1, b0, b1 = _unit(d, e), _unit(d + 10, e), _unit(d, e + 10), _unit(d + 10, e + 10)
                tri += [a0, a1, b1, a0, b1, b0]
        m.append(dict(ns="sphere", id=1, type="triangles", points=tri, scale=(1.0, 1.0, 1.0),
                      rgba=(0.25, 0.45, 0.75, 0.18)))
    else:
        m.append(dict(ns="sphere", id=1, type="sphere", pos=(0.0, 0.0, 0.0), scale=(2.0, 2.0, 2.0),
                      rgba=(0.25, 0.45, 0.75, 0.18)))
    m.append(dict(ns="sphere", id=0, type="lines", points=seg, scale=(0.004, 0, 0), rgba=(0.6, 0.75, 0.95, 0.35)))
    ring = [_unit(d, 0.0, 1.0) for d in range(0, 361, 5)]
    m.append(dict(ns="ring", id=0, type="line", points=ring, scale=(0.01, 0, 0), rgba=(0.5, 0.5, 0.5, 0.6)))
    for i, d in enumerate((0, 90, 180, -90)):
        text("ring", 10 + i, _unit(d, 0.0, 1.18), f"{d}°", (0.7, 0.7, 0.7), 0.07)
    for i, (lab, pos, rgb) in enumerate((("x", (1.35, 0, 0), (1.0, 0.3, 0.3)), ("y", (0, 1.35, 0), (0.3, 1.0, 0.3)),
                                         ("z", (0, 0, 1.35), (0.4, 0.6, 1.0)))):
        text("ring", 20 + i, pos, lab, rgb, 0.12)
    if core.known_az is not None:
        kaz = ((core.known_az + 180.0) % 360.0) - 180.0
        kel = core.known_el or 0.0
        arrow("known", 0, kaz, kel, (0.9, 0.15, 0.15), 0.9, 0.04)
        red = (0.9, 0.15, 0.15)
        c = core.known_cone
        if core.cone_3d:
            # real 3D cone: circle on the sphere at angle c around the known direction + 4 rays
            u = _unit(kaz, kel)
            e1 = _unit(kaz + 90.0, 0.0)
            e2 = (u[1] * e1[2] - u[2] * e1[1], u[2] * e1[0] - u[0] * e1[2], u[0] * e1[1] - u[1] * e1[0])
            cr, sr = math.cos(math.radians(c)), math.sin(math.radians(c))
            up = getattr(core, "upper_only", False)
            circ = []
            for k in range(0, 361, 3):
                a = math.radians(k)
                circ.append(tuple(cr * u[j] + sr * (math.cos(a) * e1[j] + math.sin(a) * e2[j]) for j in range(3)))
            # drop zone border = the cone circle, cut at the equator on the dome (upper_only):
            # the part of the circle below the array plane is replaced by the equator arc inside the cone
            seg = []
            for p0, p1 in zip(circ[:-1], circ[1:]):
                if not up or (p0[2] >= 0 and p1[2] >= 0):
                    seg += [p0, p1]
            if up:
                for d in range(0, 360, 2):
                    q0, q1 = _unit(d, 0.0), _unit(d + 2, 0.0)
                    if (ang_dist(d, 0.0, kaz, kel) <= c) and (ang_dist(d + 2, 0.0, kaz, kel) <= c):
                        seg += [q0, q1]
            m.append(dict(ns="known", id=1, type="lines", points=seg, scale=(0.015, 0, 0), rgba=(*red, 0.9)))
            rays = []
            ends = [q for q in circ[::15] if not up or q[2] >= 0]
            for q in ends:
                rays += [(0, 0, 0), q]
            m.append(dict(ns="known", id=3, type="lines", points=rays or [(0, 0, 0), u],
                          scale=(0.006, 0, 0), rgba=(*red, 0.5)))
            label = f"KNOWN az {kaz:+.0f}° el {kel:+.0f}°  dropped: 3D cone {c:.0f}°"
        else:
            # azimuth slice: everything with |az - known| <= c is dropped, at ANY elevation
            # -> two meridians from pole to pole + the equator arc between them
            seg = []
            for edge in (kaz - c, kaz + c):
                for e in range(-90, 90, 5):
                    seg += [_unit(edge, e), _unit(edge, e + 5)]
            for d in range(int(-c), int(c), 2):
                seg += [_unit(kaz + d, 0.0), _unit(kaz + d + 2, 0.0)]
            m.append(dict(ns="known", id=1, type="lines", points=seg, scale=(0.012, 0, 0), rgba=(*red, 0.9)))
            fill = []
            for e in range(-80, 81, 10):                     # light hatching inside the slice
                for d in range(int(-c), int(c), 4):
                    fill += [_unit(kaz + d, e), _unit(kaz + d + 4, e)]
            m.append(dict(ns="known", id=3, type="lines", points=fill, scale=(0.004, 0, 0), rgba=(*red, 0.35)))
            label = f"KNOWN az {kaz:+.0f}°  dropped: az {kaz - c:+.0f}..{kaz + c:+.0f}° at ANY elevation"
        text("known", 2, _unit(kaz, kel, 1.08), label, (1.0, 0.4, 0.4))
        if core.opposite_cone > 0:
            oaz = kaz + 180.0
            arrow("echo", 0, oaz, kel, (1.0, 0.55, 0.0), 0.6, 0.025)
            arc("echo", 1, oaz, core.opposite_cone, (1.0, 0.55, 0.0))
            text("echo", 2, _unit(oaz, kel, 0.7), "ECHO (dropped)", (1.0, 0.65, 0.2))
    for i, (ra, re) in enumerate(core.residuals):
        arrow("residual", i, ra, re, (1.0, 0.55, 0.0), 0.5, 0.02)
    # live points on the sphere + their trail (last seconds): you see the point move
    for i, d in enumerate(getattr(core, "dropped", []) or []):
        m.append(dict(ns="dropped_pt", id=i, type="sphere", pos=_unit(d["azimuth_deg"], d["elevation_deg"]),
                      scale=(0.06, 0.06, 0.06), rgba=(*KIND_RGB["dropped"], 0.7)))
    for tid, tr in (trails or {}).items():
        if len(tr["pts"]) >= 2:
            m.append(dict(ns="trail", id=tid, type="points", points=list(tr["pts"]),
                          scale=(0.025, 0.025, 0), rgba=(*KIND_RGB[tr["kind"]], 0.5)))
    hud = []
    for i, t in enumerate(targets):
        rgb = (0.1, 0.9, 0.2) if t["drone"] else (0.95, 0.85, 0.1)
        px, py, pz = _unit(t["azimuth_deg"], t["elevation_deg"])
        x, y, z = (t.get("x", px), t.get("y", py), t.get("z", pz))
        # projection lines: point -> its x,y on the floor (shows z) -> origin (shows x,y)
        m.append(dict(ns="target_proj", id=i, type="lines",
                      points=[(px, py, pz), (px, py, 0.0), (px, py, 0.0), (0, 0, 0),
                              (px, py, 0.0), (px, 0.0, 0.0), (px, py, 0.0), (0.0, py, 0.0)],
                      scale=(0.006, 0, 0), rgba=(*rgb, 0.6)))
        m.append(dict(ns="target_shadow", id=i, type="sphere", pos=(px, py, 0.0),
                      scale=(0.05, 0.05, 0.01), rgba=(*rgb, 0.5)))
        name = f"DRONE id {t['id']}" if t["drone"] else f"target id {t['id']}"
        p = f" p={t['drone_prob']:.2f}" if t.get("drone_prob") is not None else ""
        hud.append(f"{name}{p}   x {x:+.2f}  y {y:+.2f}  z {z:+.2f}   "
                   f"az {t['azimuth_deg']:+.0f}°  el {t['elevation_deg']:+.0f}°")
        m.append(dict(ns="target_pt", id=i, type="sphere", pos=_unit(t["azimuth_deg"], t["elevation_deg"]),
                      scale=(0.12, 0.12, 0.12), rgba=(*rgb, 1.0)))
        arrow("target", i, t["azimuth_deg"], t["elevation_deg"], rgb, 1.0, 0.045)
        p = f" p={t['drone_prob']:.2f}" if t.get("drone_prob") is not None else ""
        lab = (f"DRONE id {t['id']}{p}" if t["drone"] else f"target id {t['id']}") + \
              f"\nx {x:+.2f} y {y:+.2f} z {z:+.2f}\naz {t['azimuth_deg']:+.0f}° el {t['elevation_deg']:+.0f}°"
        text("target", 100 + i, _unit(t["azimuth_deg"], t["elevation_deg"], 1.12), lab, rgb, 0.1)
    # live panel above the sphere: x y z of every target (or "no target")
    m.append(dict(ns="hud", id=0, type="text", pos=(0.0, 0.0, 1.75),
                  text="LIVE TARGETS (unit direction x y z)\n" + ("\n".join(hud) if hud else "no target"),
                  scale=(0, 0, 0.09), rgba=(1.0, 1.0, 1.0, 1.0)))
    if not labels:
        # omnisphere only: sphere, equator, known zone, track balls + trails; no text, no projections
        m = [d for d in m if d["type"] != "text" and d["ns"] not in ("target_proj", "target_shadow")]
    if not with_static:
        m = [d for d in m if d["ns"] not in STATIC_NS]
    return m


# ---------------------------------------------------------------- ROS node

def import_sst_msg():
    for mod in ("odas_ros_msgs.msg", "odas_ros.msg"):
        try:
            m = __import__(mod, fromlist=["OdasSstArrayStamped"])
            return m.OdasSstArrayStamped
        except (ImportError, AttributeError):
            continue
    sys.exit("OdasSstArrayStamped not found: source ~/ros2_ws/install/setup.bash first")


def run_node():
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
    Sst = import_sst_msg()
    try:
        from visualization_msgs.msg import Marker, MarkerArray
        from geometry_msgs.msg import Point
    except ImportError:
        Marker = MarkerArray = Point = None

    class TargetDirectionNode(Node):
        def __init__(self):
            super().__init__("target_direction_node")
            def p(k, v):   # declare once; later calls with the same name reuse it
                if self.has_parameter(k):
                    return self.get_parameter(k).value
                return self.declare_parameter(k, v).value
            self.core = TargetDirectionCore(
                min_activity=p("min_activity", 0.1), known_cone_deg=p("known_cone_deg", 20.0),
                min_prob=p("min_prob", 0.5), match_deg=p("match_deg", 15.0),
                min_dbfs=(lambda v: None if v <= -199 else v)(p("min_dbfs", -200.0)),
                known_lock_s=p("known_lock_s", 5.0), learn_s=p("learn_s", 15.0),
                residual_cone_deg=p("residual_cone_deg", 20.0),
                opposite_cone_deg=p("opposite_cone_deg", 25.0),
                known_id_stick=p("known_id_stick", True), cone_3d=p("cone_3d", False),
                upper_only=p("upper_only", True), max_el_deg=float(p("max_el_deg", 90.0)))
            root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.dirs_file = os.path.expanduser(p("known_dirs_file", os.path.join(root, "uma-8", "config", "known_dirs.json")))
            npz = os.path.join(os.path.dirname(self.dirs_file), "known_null.npz")
            self.saved = False
            try:
                if os.path.getmtime(self.dirs_file) >= os.path.getmtime(npz):
                    with open(self.dirs_file) as f:
                        self.core.from_dict(json.load(f))
                    self.saved = True
                    self.get_logger().info(f"known directions loaded from {self.dirs_file}: "
                                           f"{self.core.to_dict()}")
            except (OSError, ValueError):
                pass
            if not self.saved:
                self.get_logger().info(f"LEARNING the known source for {self.core.learn_s:g} s: "
                                       "keep ONLY the known source on, no target")
            self.log_every = p("log_every_s", 0.5)
            self.changes_only = p("changes_only", True)
            self.mode = p("mode", "drone")          # drone | changes | all
            self.display = DroneDisplay(rate_hz=p("rate_hz", 10.0), change_deg=p("change_deg", 5.0))
            self.changes = ChangeFilter(change_deg=p("change_deg", 5.0), lost_s=p("lost_s", 1.0))
            self.t0 = time.monotonic()
            self.only_drone = p("only_drone", False)
            log_dir = os.path.expanduser(p("log_dir", "~/dataset"))
            os.makedirs(log_dir, exist_ok=True)
            path = os.path.join(log_dir, f"target_direction_{time.strftime('%Y%m%d_%H%M%S')}.csv")
            self.csv_f = open(path, "w", newline="")
            self.csv = csv.writer(self.csv_f)
            self.csv.writerow(["t_unix", "id", "drone", "drone_prob", "x", "y", "z",
                               "azimuth_deg", "elevation_deg", "activity", "rms_dbfs", "event"])
            self.last_log = 0.0
            self.pub = self.create_publisher(String, "/target/direction", 10)
            self.frame = p("frame", "odas")
            self.mk_rate = max(1.0, float(p("marker_rate_hz", 25.0)))   # live: 25 updates/s
            self.mk_labels = bool(p("labels", True))                    # false = omnisphere only
            self.mk_period = 1.0 / self.mk_rate
            self.mk_life = max(0.3, 3.0 / self.mk_rate)   # a point vanishes 0.3 s after its track ends
            self.trail_s = float(p("trail_s", 6.0))
            self.mk_last = 0.0
            self.static_last = 0.0
            self.mk_first = True
            self.mk_err = False
            self.trails = {}
            self.mk_pub = self.create_publisher(MarkerArray, "/target/markers", 1) if MarkerArray else None
            sst_topic = p("sst_topic", "/gate/sst")
            self.create_subscription(Sst, sst_topic, self.on_sst, 10)
            self.create_subscription(String, p("known_topic", "/gate/known"), self.on_known, 10)
            self.create_subscription(String, p("drone_topic", "/classifier/tracks"), self.on_drones, 10)
            self.get_logger().info(f"{sst_topic} (+ known point, /classifier/tracks) -> "
                                   f"/target/direction | csv {path}")

        def publish_markers(self, targets):
            # live: no DELETEALL (flicker); moving items carry a short lifetime and simply
            # expire when not refreshed. Stamp left at 0 = "latest", so the Jetson/laptop
            # clock difference cannot hide or delay anything in RViz.
            arr = MarkerArray()
            now = time.monotonic()
            if self.mk_first:
                clear = Marker()
                clear.header.frame_id = self.frame
                clear.action = 3                               # DELETEALL old scene once
                arr.markers.append(clear)
                self.mk_first = False
            with_static = now - self.static_last >= 1.0
            if with_static:
                self.static_last = now
            update_trails(self.trails, targets, self.core.dropped,
                          keep=int(self.trail_s * self.mk_rate), forget=int(3 * self.mk_rate))
            life_s = int(self.mk_life)
            life_ns = int((self.mk_life - life_s) * 1e9)
            for d in build_markers(self.core, targets, self.trails, with_static, self.mk_labels):
                mk = Marker()
                mk.header.frame_id = self.frame
                if d["ns"] not in STATIC_NS:
                    mk.lifetime.sec, mk.lifetime.nanosec = life_s, life_ns
                mk.ns, mk.id = str(d["ns"]), int(d["id"])
                mk.type = {"arrow": Marker.ARROW, "text": Marker.TEXT_VIEW_FACING,
                           "line": Marker.LINE_STRIP, "lines": Marker.LINE_LIST,
                           "sphere": Marker.SPHERE, "points": Marker.POINTS,
                           "triangles": Marker.TRIANGLE_LIST}[d["type"]]
                mk.action = Marker.ADD
                mk.pose.orientation.w = 1.0
                mk.scale.x, mk.scale.y, mk.scale.z = map(float, d["scale"])
                mk.color.r, mk.color.g, mk.color.b, mk.color.a = map(float, d["rgba"])
                if "points" in d:
                    for x, y, z in d["points"]:
                        mk.points.append(Point(x=float(x), y=float(y), z=float(z)))
                if "pos" in d:
                    mk.pose.position.x, mk.pose.position.y, mk.pose.position.z = map(float, d["pos"])
                if "text" in d:
                    mk.text = str(d["text"])
                arr.markers.append(mk)
            self.mk_pub.publish(arr)

        def on_known(self, msg):
            try:
                self.core.on_known(json.loads(msg.data), time.monotonic())
            except ValueError:
                pass

        def on_drones(self, msg):
            try:
                self.core.on_drones(json.loads(msg.data), time.monotonic())
            except ValueError:
                pass

        def on_sst(self, msg):
            now = time.monotonic()
            targets = self.core.on_sst(msg.sources, now)
            if self.mk_pub is not None and now - self.mk_last >= self.mk_period:
                self.mk_last = now
                try:                                   # RViz output must never stop the localisation
                    self.publish_markers(targets)
                except Exception as e:  # noqa: BLE001
                    if not self.mk_err:
                        self.get_logger().warn(f"RViz markers disabled: {type(e).__name__}: {e}")
                    self.mk_err = True
                    self.mk_pub = None
            if not self.saved and self.core.learning_done and self.core.known_locked:
                try:
                    with open(self.dirs_file, "w") as f:
                        json.dump(self.core.to_dict(), f)
                    self.get_logger().info(f"known directions learned and saved: {self.core.to_dict()}")
                except OSError as e:
                    self.get_logger().warn(f"cannot save {self.dirs_file}: {e}")
                self.saved = True
            if self.mode == "drone":
                drones, lines = self.display.update(targets, now, now - self.t0, core=self.core)
                out = String()
                known = {"azimuth_deg": None if self.core.known_az is None else
                         round(((self.core.known_az + 180) % 360) - 180, 1),
                         "elevation_deg": self.core.known_el, "cone_deg": self.core.known_cone,
                         "dropped_track_ids": [d["id"] for d in self.core.dropped],
                         "residual_dirs": [[round(((a + 180) % 360) - 180, 1), round(e, 1)]
                                           for a, e in self.core.residuals],
                         "learning": self.core.learning(now),
                         "opposite_az": None if (self.core.known_az is None or self.core.opposite_cone <= 0)
                         else round(((self.core.known_az + 360.0) % 360.0) - 180.0, 1),
                         "opposite_cone_deg": self.core.opposite_cone,
                         "learn_left_s": round(self.core.learn_left(now), 1)}
                out.data = json.dumps({"stamp": time.time(), "drone": bool(drones), "targets": drones,
                                       "known_source_dropped": known})
                self.pub.publish(out)
                t_unix = time.time()
                for t in drones:
                    self.csv.writerow([f"{t_unix:.3f}", t["id"], 1, t["drone_prob"], t["x"], t["y"], t["z"],
                                       t["azimuth_deg"], t["elevation_deg"], t["activity"],
                                       t["rms_dbfs"], ""])
                for ln in lines:
                    print(ln, flush=True)
                    if ln.endswith("no drone detected"):
                        self.csv.writerow([f"{t_unix:.3f}", "", 0, "", "", "", "", "", "", "", "",
                                           "no drone"])
                if lines:
                    self.csv_f.flush()
                return
            if self.only_drone:
                targets = [t for t in targets if t["drone"]]
            out = String()
            out.data = json.dumps({"stamp": time.time(), "targets": targets})
            self.pub.publish(out)
            t_unix = time.time()
            events = self.changes.update(targets, now)
            why_by_id = {e[0]["id"]: e[1] for e in events if e[1] != "lost"}
            for t in targets:
                self.csv.writerow([f"{t_unix:.3f}", t["id"], int(t["drone"]), t["drone_prob"],
                                   t["x"], t["y"], t["z"], t["azimuth_deg"],
                                   t["elevation_deg"], t["activity"], t["rms_dbfs"],
                                   why_by_id.get(t["id"], "")])
            if self.changes_only:
                for t, why in events:
                    if why == "lost":
                        self.csv.writerow([f"{t_unix:.3f}", t["id"], int(t["drone"]), "", "", "", "",
                                           t["azimuth_deg"], t["elevation_deg"], "", "", "lost"])
                    self.get_logger().info(fmt_event(t, why, now - self.t0))
                if events:
                    self.csv_f.flush()
                return
            if now - self.last_log >= self.log_every:
                self.last_log = now
                self.csv_f.flush()
                if targets:
                    for t in targets:
                        self.get_logger().info(fmt(t))
                else:
                    self.get_logger().info("[TARGET] none (no active track outside the known cone)")

        def destroy_node(self):
            self.csv_f.close()
            super().destroy_node()

    rclpy.init()
    node = TargetDirectionNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, Exception) as e:   # Ctrl+C / pkill (ExternalShutdownException)
        if not isinstance(e, KeyboardInterrupt) and type(e).__name__ != "ExternalShutdownException":
            raise
    finally:
        try:
            node.destroy_node()
        except Exception:  # noqa: BLE001
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:  # noqa: BLE001  (already shut down by the signal handler)
            pass


# ---------------------------------------------------------------- self-test

def _selftest_cone():
    s3 = lambda i, az, el: {"id": i, "x": math.cos(math.radians(el)) * math.cos(math.radians(az)),
                            "y": math.cos(math.radians(el)) * math.sin(math.radians(az)),
                            "z": math.sin(math.radians(el)), "activity": 0.9}
    for c3, expect in ((False, [3]), (True, [2, 3])):
        q = TargetDirectionCore(learn_s=0, opposite_cone_deg=0, known_lock_s=0, cone_3d=c3, upper_only=False)
        q.on_known([{"azimuth_deg": 5.0, "elevation_deg": -10.0}], 0.0)
        q.known_locked = True
        out = q.on_sst([s3(1, 10.0, -5.0), s3(2, 20.0, 60.0), s3(3, -156.0, 28.0)], 1.0)
        assert sorted(t["id"] for t in out) == expect, (c3, out, q.dropped)
        mk = build_markers(q, out)
        assert any(d["ns"] == "known" and d["id"] == 1 for d in mk)
        bare = build_markers(q, out, labels=False)
        assert not any(d["type"] == "text" for d in bare) and any(d["ns"] == "sphere" for d in bare)
    print("drop zone: az slice drops id 2 (az 20, el 60); 3D cone keeps it; labels off -> ok")
    # upper hemisphere: known measured at el -31 -> +31; a track mirrored below (el -40) = above (+40)
    h = TargetDirectionCore(learn_s=0, opposite_cone_deg=0, known_lock_s=0, cone_3d=True)
    h.on_known([{"azimuth_deg": 22.0, "elevation_deg": -31.0}], 0.0)
    assert h.known_el == 31.0, h.known_el
    h.known_locked = True
    out = h.on_sst([s3(1, 22.0, -35.0), s3(2, -64.0, -50.0)], 1.0)
    assert [t["id"] for t in out] == [2] and out[0]["elevation_deg"] > 0 and out[0]["z"] > 0, out
    mk = build_markers(h, out)
    assert all(pt[2] >= -1e-9 for d in mk if d["ns"] in ("sphere", "known") for pt in d.get("points", [])), "below equator"
    print("upper hemisphere only: known el -31 -> +31, tracks folded up, dome drawn -> ok")
    z = TargetDirectionCore(learn_s=0, opposite_cone_deg=0, known_lock_s=0, cone_3d=True, max_el_deg=60.0)
    z.on_known([{"azimuth_deg": 33.0, "elevation_deg": 42.0}], 0.0)
    z.known_locked = True
    out = z.on_sst([s3(3, -102.0, 77.0), s3(4, -150.0, 20.0)], 1.0)
    assert [t["id"] for t in out] == [4] and z.dropped[0]["why"] == "zenith", (out, z.dropped)
    print("zenith gate: el 77 dropped above max_el 60 -> ok")


def selftest():
    _selftest_cone()
    # ID tracker drop: id 4 starts in the known cone, then drifts out -> still dropped by id
    k = TargetDirectionCore(learn_s=0, opposite_cone_deg=0, known_lock_s=0)
    k.on_known([{"azimuth_deg": 10.0}], 0.0)
    k.known_locked = True
    s = lambda i, az: {"id": i, "x": math.cos(math.radians(az)), "y": math.sin(math.radians(az)), "z": 0.1, "activity": 0.9}
    assert k.on_sst([s(4, 12.0), s(9, 80.0)], 0.1)[0]["id"] == 9
    out = k.on_sst([s(4, 45.0), s(9, 80.0)], 0.2)
    assert [t["id"] for t in out] == [9] and k.dropped[0]["why"] == "known-id", (out, k.dropped)
    assert [t["id"] for t in k.on_sst([s(4, 45.0)], 4.0)] == []      # still alive -> still dropped
    assert [t["id"] for t in k.on_sst([s(6, 45.0)], 8.0)] == [6]     # new id outside the cone -> target
    print("ID tracker drop: id stays dropped after leaving the cone -> ok")
    c = TargetDirectionCore(learn_s=0)
    c.on_known({"az": 0.0, "label": "known"})
    s45 = math.radians(45)
    tracks = [
        {"id": 1, "x": 1.0, "y": 0.02, "z": 0.0, "activity": 0.9},                 # known direction
        {"id": 3, "x": math.cos(s45) * 0.8, "y": -math.sin(s45) * 0.8, "z": 0.6, "activity": 0.8},
        {"id": 5, "x": 0.0, "y": 1.0, "z": 0.0, "activity": 0.05},                 # too weak
        {"id": 0, "x": 0.0, "y": 0.0, "z": 0.0, "activity": 0.0},                  # empty slot
    ]
    t = c.on_sst(tracks, 0.0)
    assert [x["id"] for x in t] == [3], t
    assert abs(t[0]["azimuth_deg"] + 45.0) < 0.2 and abs(t[0]["elevation_deg"] - 36.9) < 0.2, t[0]
    assert not t[0]["drone"]
    print("known cone dropped, weak + empty skipped ->", fmt(t[0]))

    c.on_drones({"targets": [{"track_id": 3, "drone_prob": 0.91}]}, 0.1)
    t = c.on_sst(tracks, 0.2)
    assert t[0]["drone"] and t[0]["drone_prob"] == 0.91
    print("classifier marks track 3 (by id)        ->", fmt(t[0]))

    c.on_drones([{"azimuth_deg": 312.0, "prob": 0.8}], 0.3)                    # no id, bearing only
    t = c.on_sst(tracks, 0.4)
    assert t[0]["drone"] and t[0]["drone_prob"] == 0.8
    print("classifier marks it (by bearing, no id) ->", fmt(t[0]))

    t = c.on_sst(tracks, 5.0)                                                   # classifier stale
    assert not t[0]["drone"]
    print("classifier stale > 1.5 s                ->", fmt(t[0]))
    cf = ChangeFilter(change_deg=5.0, lost_s=1.0)
    def T(az, el, drone=False):
        x = math.cos(math.radians(el)) * math.cos(math.radians(az))
        y = math.cos(math.radians(el)) * math.sin(math.radians(az))
        return {"id": 7, "x": x, "y": y, "z": math.sin(math.radians(el)), "azimuth_deg": az,
                "elevation_deg": el, "activity": 0.9, "drone": drone, "drone_prob": 0.9 if drone else None}
    ev = [w for _, w in cf.update([T(10, 20)], 0.0)];  assert ev == ["new"], ev
    ev = [w for _, w in cf.update([T(12, 21)], 0.5)];  assert ev == [], ev          # 2° jitter: silent
    ev = [w for _, w in cf.update([T(17, 21)], 1.0)];  assert ev and ev[0].startswith("moved"), ev
    ev = [w for _, w in cf.update([T(17, 21, True)], 1.5)]; assert ev == ["now DRONE"], ev
    ev = [w for _, w in cf.update([], 3.0)];            assert ev == ["lost"], ev
    print("changes only: new / jitter silent / moved / now DRONE / lost -> ok")
    dd = DroneDisplay(rate_hz=10.0)
    _, l0 = dd.update([T(10, 20)], 0.0, 0.0);       assert l0 == ["[    0.0 s] no drone detected"], l0
    _, l1 = dd.update([T(10, 20)], 0.05, 0.05);     assert l1 == [], l1          # still none: silent
    _, l2 = dd.update([T(30, 20, True)], 0.10, 0.1); assert len(l2) == 1 and "DRONE id 7" in l2[0], l2
    _, l3 = dd.update([T(31, 20, True)], 0.15, 0.15); assert l3 == [], l3        # within 0.1 s
    _, l4 = dd.update([T(32, 20, True)], 0.21, 0.21); assert len(l4) == 1, l4
    _, l5 = dd.update([T(32, 20, False)], 0.30, 0.3); assert l5 == ["[    0.3 s] no drone detected"], l5
    print("mode drone: only DRONE with live x y z, 'no drone detected' once -> ok")
    kc = TargetDirectionCore(learn_s=0)
    kd = DroneDisplay(rate_hz=10.0)
    kc.on_known({"az": -3.0, "el": 11.0, "label": "known"}, 0.0)
    sources = [{"id": 61, "x": 0.98, "y": -0.05, "z": 0.19, "activity": 0.9},     # on the known source
               {"id": 62, "x": 0.0, "y": 0.9, "z": 0.43, "activity": 0.9}]        # target at +90°
    kc.on_drones([{"id": 62, "prob": 0.93, "is_drone": True, "rms_dbfs": -50.0}], 0.0)
    tg = kc.on_sst(sources, 0.1)
    _, kl = kd.update(tg, 0.1, 0.1, core=kc)
    assert [t["id"] for t in tg] == [62], tg
    assert any("KNOWN source" in l and "DROPPED" in l for l in kl), kl
    assert any("dropped track id 61" in l for l in kl), kl
    assert any("DRONE id 62" in l for l in kl), kl
    _, kl2 = kd.update(kc.on_sst(sources, 0.3), 0.3, 0.3, core=kc)
    assert not any("KNOWN" in l or "dropped" in l for l in kl2), kl2              # printed once, not spammed
    print("known source shown as DROPPED, its track never localised, target only -> ok")
    lc = TargetDirectionCore(known_lock_s=5.0, learn_s=0)
    for i in range(52):                                     # 5 s of a steady known source near +8°
        lc.on_known({"az": 8.0 + (i % 3) - 1, "el": 3.0}, i * 0.1)
    assert lc.known_locked and abs(((lc.known_az + 180) % 360) - 180 - 8.0) < 1.0, lc.known_az
    for a in (13.3, 20.6, 25.8, 11.4):                       # live estimate dragged by a target
        lc.on_known({"az": a, "el": 0.0}, 6.0)
    assert abs(((lc.known_az + 180) % 360) - 180 - 8.0) < 1.0, lc.known_az
    tgt = lc.on_sst([{"id": 70, "x": math.cos(math.radians(32)), "y": math.sin(math.radians(32)),
                      "z": 0.0, "activity": 0.9}], 6.1)
    assert [t["id"] for t in tgt] == [70], tgt               # target at +32° kept (cone stays at +8°)
    print("known direction locked after 5 s: a nearby target cannot drag the cone -> ok")
    rc = TargetDirectionCore(learn_s=15.0)
    rc.on_known({"az": 8.0, "el": 3.0}, 0.0)
    def xyz(az, el):
        return {"x": math.cos(math.radians(el)) * math.cos(math.radians(az)),
                "y": math.cos(math.radians(el)) * math.sin(math.radians(az)), "z": math.sin(math.radians(el))}
    t = 0.0
    while t < 15.2:                                         # known source alone: a side echo at -60
        assert rc.on_sst([dict(id=3, activity=0.9, **xyz(-60 + (t % 2), 16))], t) == []
        t += 0.01
    assert rc.learning_done and len(rc.residuals) == 1, rc.residual_bins
    out = rc.on_sst([dict(id=3, activity=0.9, **xyz(-62, 14)), dict(id=9, activity=0.9, **xyz(90, 20))], t)
    assert [o["id"] for o in out] == [9] and any(d["why"] == "residual" for d in rc.dropped), (out, rc.dropped)
    rc2 = TargetDirectionCore(learn_s=0); rc2.from_dict(rc.to_dict())
    assert rc2.learning_done and rc2.known_locked and len(rc2.residuals) == 1
    print("echo of the known source learned in 15 s and dropped; target at +90 kept; saved/loaded -> ok")
    oc = TargetDirectionCore(learn_s=0)
    oc.on_known({"az": 2.1, "el": 1.1}, 0.0)
    out = oc.on_sst([dict(id=2, activity=0.9, **xyz(-168.5, 16.4)), dict(id=5, activity=0.9, **xyz(-100, 10))], 0.1)
    assert [o["id"] for o in out] == [5] and oc.dropped[0]["why"] == "opposite", (out, oc.dropped)
    print("your log: echo at -168.5 (known +2.1) dropped as opposite; target at -100 kept -> ok")
    out[0]["drone"] = True; out[0]["drone_prob"] = 0.9
    tgts = out + [dict(id=8, azimuth_deg=60.0, elevation_deg=5.0, drone=False, drone_prob=None)]
    tr = {}
    for k in range(5):
        update_trails(tr, [dict(tg, azimuth_deg=tg["azimuth_deg"] + k) for tg in tgts])
    assert all(len(v["pts"]) == 5 for v in tr.values())
    for k in range(31):
        update_trails(tr, [])
    assert tr == {}, "trail forgotten after 3 s"
    for k in range(3):
        update_trails(tr, tgts)
    mk = build_markers(oc, tgts, tr)
    kinds = {d["type"] for d in mk}
    assert {"lines", "sphere", "points"} <= kinds, kinds
    ns = [d["ns"] for d in mk]
    assert ns.count("known") == 4 and ns.count("echo") == 3 and ns.count("target") == 4, ns
    txt = [d["text"] for d in mk if d["type"] == "text"]
    assert any(t.startswith("DRONE id 5") for t in txt) and any(t.startswith("target id 8") for t in txt), txt
    print("RViz markers: ring, KNOWN (red), ECHO (orange), DRONE (green), target (yellow) -> ok")
    for l in kl: print(l)
    print(l2[0])
    c2 = TargetDirectionCore(min_dbfs=-65.0, learn_s=0)
    c2.on_drones(json.loads('[{"id": 37, "prob": 0.92, "is_drone": true, "rms_dbfs": -48.0},'
                                ' {"id": 48, "prob": 0.66, "is_drone": true, "rms_dbfs": -77.4},'
                                ' {"id": 55, "prob": 0.40, "is_drone": false, "rms_dbfs": -50.0}]'), 0.0)
    out = c2.on_sst([{"id": 37, "x": 0.72, "y": -0.53, "z": 0.45, "activity": 1.0},
                     {"id": 48, "x": 0.44, "y": 0.75, "z": 0.50, "activity": 1.0},
                     {"id": 55, "x": -0.78, "y": 0.6, "z": 0.2, "activity": 1.0}], 0.1)
    ids = {t["id"]: t for t in out}
    assert 48 not in ids, "ghost at -77 dBFS must be gated"
    assert ids[37]["drone"] and ids[37]["drone_prob"] == 0.92 and not ids[55]["drone"], ids
    print("/classifier/tracks: DRONE by id, -77 dBFS ghost gated, is_drone=false kept as target -> ok")
    print(fmt_event(ids[37], "new", 1.0)); print(fmt_event(ids[55], "new", 1.0))
    print(fmt_event(T(17, 21, True), "moved 6.6°", 12.4))
    print("\nselftest PASSED")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        run_node()

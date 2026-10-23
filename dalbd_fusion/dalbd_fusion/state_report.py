#!/usr/bin/env python3
"""state_report.py - timeline of a fusion run from fusion_<date>.csv.

Prints the state segments (start, end, duration, mean bearing), the share of time per output source and the
vision-sound gap, and saves <csv>_timeline.png (bands = state, dots = vision / sound / fused azimuth).

Usage:  ros2 run dalbd_fusion state_report                    (latest file in /workspaces/isaac_ros-dev/fusion_logs)
        ros2 run dalbd_fusion state_report /path/fusion_20261020_150000.csv
"""
import csv
import glob
import os
import sys

DEF_DIRS = ["/workspaces/isaac_ros-dev/fusion_logs", os.path.expanduser("~/workspaces/isaac_ros-dev/fusion_logs"),
            os.path.expanduser("~/fusion_logs")]
COL = {"VISUAL_TRACK": "#b9b3ee", "SOUND_TRACK": "#a8e0cd", "SEARCH": "#e4e2dc"}
SRC = {"0": "none", "1": "vision", "2": "sound"}


def f(x):
    try:
        v = float(x)
        return None if v != v else v
    except (TypeError, ValueError):
        return None


def main(args=None):
    argv = [a for a in sys.argv[1:] if not a.startswith("--ros-args")]
    if argv:
        path = argv[0]
    else:
        files = [x for d in DEF_DIRS for x in glob.glob(os.path.join(d, "fusion_*.csv"))]
        if not files:
            raise SystemExit(f"[FAIL] no fusion_*.csv in {DEF_DIRS}")
        path = max(files, key=os.path.getmtime)
    rows = list(csv.DictReader(open(path)))
    if not rows:
        raise SystemExit("[FAIL] empty log")
    t0 = float(rows[0]["t"])
    for r in rows:
        r["ts"] = float(r["t"]) - t0

    segs, cur = [], None
    for r in rows:
        if cur is None or r["state"] != cur["state"]:
            if cur:
                cur["end"] = r["ts"]
                segs.append(cur)
            cur = {"state": r["state"], "start": r["ts"], "az": []}
        if f(r["az_deg"]) is not None:
            cur["az"].append(f(r["az_deg"]))
    cur["end"] = rows[-1]["ts"]
    segs.append(cur)

    print(f"{path}\n{len(rows)} steps over {rows[-1]['ts']:.1f} s\n")
    print(f"{'start s':>8} {'end s':>8} {'dur s':>7}  {'state':<13} {'mean az':>8}")
    for s in segs:
        az = f"{sum(s['az']) / len(s['az']):8.1f}" if s["az"] else "       -"
        print(f"{s['start']:8.1f} {s['end']:8.1f} {s['end'] - s['start']:7.1f}  {s['state']:<13} {az}")
    tot = {}
    for r in rows:
        k = SRC.get(r["valid"], r["valid"])
        tot[k] = tot.get(k, 0) + 1
    print("\nshare of output: " + ", ".join(f"{k} {100 * v / len(rows):.0f}%" for k, v in tot.items()))
    print(f"transitions: {len(segs) - 1}")
    gaps = []
    for r in rows:
        va, sa = f(r["vision_az"]), f(r["sound_az"])
        if r["vision"] == "1" and r["sound"] == "1" and va is not None and sa is not None:
            gaps.append(abs((va - sa + 180) % 360 - 180))
    if gaps:
        gaps.sort()
        print(f"vision-sound azimuth gap while both present: median {gaps[len(gaps) // 2]:.1f} deg, "
              f"p95 {gaps[int(0.95 * (len(gaps) - 1))]:.1f} deg  (constant -> set sound_yaw_offset_deg; "
              f"opposite sign when moving -> flip sound_az_sign)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(pip3 install --user matplotlib for the timeline plot)")
        return
    fig, ax = plt.subplots(figsize=(11, 4))
    for s in segs:
        ax.axvspan(s["start"], s["end"], color=COL.get(s["state"], "#eee"), alpha=0.6, lw=0)
    ts = [r["ts"] for r in rows]
    ax.plot(ts, [f(r["vision_az"]) if r["vision"] == "1" else None for r in rows], ".", ms=2, color="#534AB7",
            label="vision az")
    ax.plot(ts, [f(r["sound_az"]) if r["sound"] == "1" else None for r in rows], ".", ms=2, color="#0F6E56",
            label="sound az")
    for v, c, name in (("1", "#3C3489", "fused (vision)"), ("2", "#085041", "fused (sound)")):
        pts = [(r["ts"], f(r["az_deg"])) for r in rows if r["valid"] == v and f(r["az_deg"]) is not None]
        if pts:
            ax.plot(*zip(*pts), "o", ms=2.5, color=c, label=name)
    for s in [x for x in segs if x["end"] - x["start"] >= 1.5]:
        ax.text((s["start"] + s["end"]) / 2, ax.get_ylim()[1], s["state"].split("_")[0], ha="center", va="top",
                fontsize=8)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("azimuth (deg, camera frame)")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("Fusion timeline: band = state, dots = bearings")
    fig.tight_layout()
    out = os.path.splitext(path)[0] + "_timeline.png"
    fig.savefig(out, dpi=130)
    print(f"[ OK ] {out}")


if __name__ == "__main__":
    main()

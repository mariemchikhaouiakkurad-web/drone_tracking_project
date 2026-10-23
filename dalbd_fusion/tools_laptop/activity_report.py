#!/usr/bin/env python3
"""activity_report.py - LAPTOP: WHEN each part of DALBD worked, and what it cost (power, load, temperature).

Reads, for every run recorded by run_recorder:
    run_<date>_<time>_N<N>_activity.csv   one row per message (each YOLO inference, each tracked image,
                                          each ODAS / classifier / warning output, state changes)
    run_<date>_<time>_N<N>.csv            one row per second (power, GPU / CPU load, temperatures, rates)
and writes <folder>/activity/activity_report.html with, per run:
    - a full timeline: YOLO / tracker / sound activity bars, tracker status and fusion state bands,
      rates, latency of each message, power, GPU / CPU load, temperatures - all on the same time axis
    - a 15 s zoom where every single YOLO inference and tracked image is visible
    - a table: how often and how long YOLO was active (searching vs tracking), the real time between two
      YOLO checks, tracker / sound timing, power and temperature by mode, power with vs without YOLO
and <folder>/activity/activity_summary.csv (one line per run).

Usage:  python3 activity_report.py ~/rec_test1        (sudo apt install -y python3-matplotlib)
Run it AFTER the test, not while the pipeline runs: it uses the laptop CPU, and cam_sender on the same laptop
then sends fewer camera frames. If you must: nice -n 19 python3 activity_report.py ~/rec_test1
"""
import bisect
import csv
import glob
import html
import os
import statistics as st
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PARTS = [("yolo", "YOLO", "tab:blue"), ("tracker", "Tracker (DyHiT)", "tab:orange"),
         ("sound_odas", "Sound: ODAS", "tab:green"), ("sound_classifier", "Sound: classifier", "tab:olive"),
         ("sound_warning", "Sound: warning", "tab:red")]
STATE_COL = {"tracking": "tab:orange", "searching": "lightgrey", "VISUAL_TRACK": "tab:blue",
             "SOUND_TRACK": "tab:green", "SEARCH": "lightgrey"}


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(q * len(v)))] if v else None


def r2(x, n=1):
    return "" if x is None else round(x, n)


def segments(state_rows, t_end, first=None):
    """[(t0, t1, state)] from change rows [(t, state)]."""
    out, cur, t0 = [], first, 0.0
    for t, s in state_rows:
        if cur is not None and t > t0:
            out.append((t0, t, cur))
        cur, t0 = s, t
    if cur is not None and t_end > t0:
        out.append((t0, t_end, cur))
    return out


def state_at(segs, t):
    for a, b, s in segs:
        if a <= t < b:
            return s
    return None


def union_len(iv):
    tot, end = 0.0, None
    for a, b in sorted(iv):
        if end is None or a > end:
            tot += b - a
            end = b
        elif b > end:
            tot += b - end
            end = b
    return tot


def load_run(act_path):
    sec_path = act_path.replace("_activity.csv", ".csv")
    act = read(act_path)
    sec = read(sec_path) if os.path.exists(sec_path) else []
    name = os.path.basename(sec_path)[:-4]
    n = name.rsplit("_N", 1)[-1]
    by = {}
    for r in act:
        by.setdefault(r["part"], []).append(r)
    t_end = max([fnum(r["t_s"]) or 0 for r in act] + [fnum(r.get("t_s")) or 0 for r in sec] + [1.0])
    first_trk = next((r.get("track_status") for r in sec if r.get("track_status")), None)
    first_fus = next((r.get("fusion_state") for r in sec if r.get("fusion_state")), None)
    trk = segments([(fnum(r["t_s"]), r["info"]) for r in by.get("tracker_status", [])], t_end, first_trk)
    fus = segments([(fnum(r["t_s"]), r["info"]) for r in by.get("fusion_state", [])], t_end, first_fus)
    if not trk and sec:                                  # fall back on the per-second column
        trk = segments([(fnum(r["t_s"]), r.get("track_status")) for r in sec if r.get("track_status")], t_end)
    return {"name": name, "N": n, "act": by, "sec": sec, "t_end": t_end, "trk": trk, "fus": fus}


def stats(R):
    by, sec, trk, T = R["act"], R["sec"], R["trk"], R["t_end"]
    out = {"run": R["name"], "detector_every": R["N"], "duration_s": round(T, 1)}
    mode_s = {}
    for a, b, s in trk:
        mode_s[s] = mode_s.get(s, 0.0) + (b - a)
    out["tracking_s"] = round(mode_s.get("tracking", 0.0), 1)
    out["searching_s"] = round(mode_s.get("searching", 0.0), 1)
    # ---- YOLO
    y = [(fnum(r["t_s"]), fnum(r["start_s"]), fnum(r["busy_ms"])) for r in by.get("yolo", [])]
    out["yolo_inferences"] = len(y)
    out["yolo_busy_pct"] = r2(100.0 * union_len([(s, t) for t, s, _ in y if s is not None]) / T)
    busy = [b for _, _, b in y if b is not None]
    out["yolo_busy_ms_median"] = r2(pct(busy, 0.5))
    out["yolo_busy_ms_p95"] = r2(pct(busy, 0.95))
    for mode in ("searching", "tracking"):
        ts = [t for t, _, _ in y if state_at(trk, t) == mode]
        out[f"yolo_hz_{mode}"] = r2(len(ts) / mode_s[mode], 2) if mode_s.get(mode) else ""
    # real time between two YOLO checks while tracking (consecutive inferences inside one tracking segment)
    gaps = []
    for a, b, s in trk:
        if s != "tracking":
            continue
        ts = sorted(t for t, _, _ in y if a <= t < b)
        gaps += [ts[i + 1] - ts[i] for i in range(len(ts) - 1)]
    out["yolo_check_period_s_median"] = r2(pct(gaps, 0.5), 2)
    out["yolo_check_period_s_p95"] = r2(pct(gaps, 0.95), 2)
    # ---- tracker
    tr = [(fnum(r["t_s"]), fnum(r["busy_ms"])) for r in by.get("tracker", [])]
    out["tracker_images"] = len(tr)
    out["tracker_hz_tracking"] = r2(len(tr) / mode_s["tracking"], 1) if mode_s.get("tracking") else ""
    tb = [b for _, b in tr if b is not None]
    out["tracker_busy_ms_median"] = r2(pct(tb, 0.5))
    out["tracker_busy_ms_p95"] = r2(pct(tb, 0.95))
    try:
        n = int(R["N"])
        if out["tracker_hz_tracking"]:
            out["yolo_check_period_s_expected"] = round(n / out["tracker_hz_tracking"], 2)
    except ValueError:
        pass
    # ---- sound
    for p in ("sound_odas", "sound_classifier", "sound_warning"):
        out[f"{p}_msgs"] = len(by.get(p, []))
        out[f"{p}_hz"] = r2(len(by.get(p, [])) / T, 2)
    w = by.get("sound_warning", [])
    out["sound_drone_msgs"] = sum(1 for r in w if r["info"].startswith("DRONE"))
    wl = [fnum(r["busy_ms"]) for r in w if fnum(r["busy_ms"]) is not None]
    out["sound_warning_latency_ms_median"] = r2(pct(wl, 0.5))
    # ---- power / load / temperature by mode
    def col_mean(rows, c):
        v = [fnum(r.get(c)) for r in rows if fnum(r.get(c)) is not None]
        return r2(st.mean(v), 2) if v else ""
    def col_max(rows, c):
        v = [fnum(r.get(c)) for r in rows if fnum(r.get(c)) is not None]
        return r2(max(v), 2) if v else ""
    for mode in ("tracking", "searching"):
        rows = [r for r in sec if r.get("track_status") == mode]
        out[f"P_VDD_IN_W_{mode}"] = col_mean(rows, "P_VDD_IN_W")
        out[f"gpu_load_pct_{mode}"] = col_mean(rows, "gpu_load_pct")
        out[f"cpu_total_pct_{mode}"] = col_mean(rows, "cpu_total_pct")
    trk_rows = [r for r in sec if r.get("track_status") == "tracking"]
    on = [r for r in trk_rows if (fnum(r.get("yolo_hz")) or 0) > 0]
    off = [r for r in trk_rows if (fnum(r.get("yolo_hz")) or 0) == 0]
    out["P_VDD_IN_W_tracking_yolo_on_s"] = col_mean(on, "P_VDD_IN_W")
    out["P_VDD_IN_W_tracking_yolo_off_s"] = col_mean(off, "P_VDD_IN_W")
    out["P_VDD_IN_W_mean"] = col_mean(sec, "P_VDD_IN_W")
    out["P_VDD_IN_W_max"] = col_max(sec, "P_VDD_IN_W")
    e = [fnum(r.get("P_VDD_IN_W")) for r in sec if fnum(r.get("P_VDD_IN_W")) is not None]
    out["energy_VDD_IN_Wh"] = round(sum(e) / 3600.0, 4) if e else ""
    for c in ("T_tj_C", "T_gpu_C", "T_cpu_C"):
        v = [fnum(r.get(c)) for r in sec if fnum(r.get(c)) is not None]
        if v:
            out[f"{c}_mean"] = round(st.mean(v), 1)
            out[f"{c}_max"] = round(max(v), 1)
            out[f"{c}_rise"] = round(v[-1] - v[0], 1)
    out["fan_pct_mean"] = col_mean(sec, "fan_pct")
    return out


def band(ax, segs, y, h, label):
    for a, b, s in segs:
        ax.broken_barh([(a, b - a)], (y, h), color=STATE_COL.get(s, "white"), lw=0)
    ax.text(-0.01, y + h / 2, label, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=8)


def merge_bars(bars, gap):
    """join bars closer than `gap` s: 9000 tracker bars of a 5 min run become a few segments (fast to draw)."""
    out = []
    for a, w in sorted(bars):
        if out and a - (out[-1][0] + out[-1][1]) <= gap:
            out[-1][1] = max(out[-1][1], a + w - out[-1][0])
        else:
            out.append([a, w])
    return [tuple(b) for b in out]


def timeline_axes(ax, R, x0=None, x1=None):
    by = R["act"]
    rows = len(PARTS)
    for i, (p, lab, col) in enumerate(PARTS):
        yb = rows - i + 1
        ev = by.get(p, [])
        bars = [(fnum(r["start_s"]), fnum(r["t_s"]) - fnum(r["start_s"])) for r in ev
                if fnum(r["start_s"]) is not None and fnum(r["t_s"]) is not None]
        if bars and x0 is None:                        # full run: merge (zoom keeps every single bar)
            bars = merge_bars(bars, 0.12)
        elif bars:
            bars = [b for b in bars if b[0] + b[1] >= x0 and b[0] <= x1]
        if bars:
            ax.broken_barh(bars, (yb, 0.8), color=col, lw=0)
        tk = [(fnum(r["t_s"]), "tab:red" if r["info"].startswith("DRONE") else col) for r in ev
              if fnum(r["start_s"]) is None and fnum(r["t_s"]) is not None
              and (x0 is None or x0 <= fnum(r["t_s"]) <= x1)]
        if tk:
            ax.vlines([t for t, _ in tk], yb, yb + 0.8, colors=[c for _, c in tk], lw=0.8, rasterized=True)
        ax.text(-0.01, yb + 0.4, f"{lab} ({len(ev)})", transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=8)
    band(ax, R["trk"], 1.0, 0.8, "tracker status")
    band(ax, R["fus"], 0.0, 0.8, "fusion state")
    ax.set_yticks([])
    ax.set_ylim(-0.2, rows + 2)
    if x0 is not None:
        ax.set_xlim(x0, x1)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c, label=k) for k, c in STATE_COL.items()], fontsize=7, ncol=5,
              loc="upper right", bbox_to_anchor=(1, 1.18))


def series(sec, c):
    x, y = [], []
    for r in sec:
        t, v = fnum(r.get("t_s")), fnum(r.get(c))
        if t is not None and v is not None:
            x.append(t)
            y.append(v)
    return x, y


def full_figure(R, png):
    sec, by = R["sec"], R["act"]
    fig, ax = plt.subplots(6, 1, figsize=(18, 17), sharex=True,
                           gridspec_kw={"height_ratios": [3, 1.2, 1.4, 1.4, 1.2, 1.4]})
    timeline_axes(ax[0], R)
    ax[0].set_title(f"{R['name']}  -  WHEN each part worked (bar = processing window of one message)")
    for c, lab in (("yolo_hz", "YOLO"), ("track_hz", "tracker"), ("sound_hz", "sound warning")):
        x, y = series(sec, c)
        ax[1].plot(x, y, label=lab, lw=1)
    ax[1].set_ylabel("rate (Hz)")
    for p, lab, col in PARTS:
        pts = [(fnum(r["t_s"]), fnum(r["busy_ms"])) for r in by.get(p, []) if fnum(r["busy_ms"]) is not None]
        if pts:
            if len(pts) > 6000:                          # thin very long runs (keeps the shape, draws fast)
                pts = pts[::len(pts) // 6000 + 1]
            ax[2].scatter([a for a, _ in pts], [b for _, b in pts], s=3, color=col, label=lab, rasterized=True)
    ax[2].set_ylabel("latency per message (ms)")
    ax[2].set_yscale("log")
    for c in ("P_VDD_IN_W", "P_VDD_CPU_GPU_CV_W", "P_VDD_SOC_W"):
        x, y = series(sec, c)
        if y:
            ax[3].plot(x, y, label=c[2:-2], lw=1)
    ax[3].set_ylabel("power (W)")
    for c, lab in (("gpu_load_pct", "GPU load"), ("cpu_total_pct", "CPU load"), ("fan_pct", "fan")):
        x, y = series(sec, c)
        if y:
            ax[4].plot(x, y, label=lab, lw=1)
    ax[4].set_ylabel("%")
    for c in ("T_tj_C", "T_gpu_C", "T_cpu_C", "T_soc0_C"):
        x, y = series(sec, c)
        if y:
            ax[5].plot(x, y, label=c[2:-2], lw=1)
    ax[5].set_ylabel("temperature (C)")
    ax[5].set_xlabel("time in run (s)")
    for a in ax[1:]:
        a.grid(alpha=0.3)
        if a.get_legend_handles_labels()[0]:
            a.legend(fontsize=7, loc="upper right", ncol=3)
    fig.tight_layout()
    fig.savefig(png, dpi=75)
    plt.close(fig)


def zoom_figure(R, png, width=15.0):
    trk = R["trk"]
    start = next((a for a, b, s in trk if s == "tracking" and b - a > 3 and a >= 10), None)
    if start is None:
        start = next((a for a, b, s in trk if s == "tracking"), 10.0)
    x0, x1 = start - 2.0, start - 2.0 + width
    sec = R["sec"]
    fig, ax = plt.subplots(3, 1, figsize=(18, 8.5), sharex=True, gridspec_kw={"height_ratios": [3, 1.3, 1.3]})
    timeline_axes(ax[0], R, x0, x1)
    ax[0].set_title(f"{R['name']}  -  zoom {x0:.0f}-{x1:.0f} s: every YOLO inference (blue) and tracked image (orange)")
    for p, lab, col in PARTS:
        pts = [(fnum(r["t_s"]), fnum(r["busy_ms"])) for r in R["act"].get(p, [])
               if fnum(r["busy_ms"]) is not None and x0 <= (fnum(r["t_s"]) or -1) <= x1]
        if pts:
            ax[1].plot([a for a, _ in pts], [b for _, b in pts], ".", ms=4, color=col, label=lab)
    ax[1].set_ylabel("latency (ms)")
    x, y = series(sec, "P_VDD_IN_W")
    ax[2].step(x, y, where="post", label="VDD_IN (1 value per second)")
    x, y = series(sec, "gpu_load_pct")
    if y:
        a2 = ax[2].twinx()
        a2.step(x, y, where="post", color="tab:purple", alpha=0.6, label="GPU load %")
        a2.set_ylabel("GPU %")
    ax[2].set_ylabel("power (W)")
    ax[2].set_xlabel("time in run (s)")
    for a in ax[1:]:
        a.grid(alpha=0.3)
        if a.get_legend_handles_labels()[0]:
            a.legend(fontsize=7, loc="upper left")
    ax[2].set_xlim(x0, x1)
    fig.tight_layout()
    fig.savefig(png, dpi=75)
    plt.close(fig)


GROUPS = [
    ("Run", ["detector_every", "duration_s", "tracking_s", "searching_s"]),
    ("YOLO - when and how long", ["yolo_inferences", "yolo_busy_pct", "yolo_hz_searching", "yolo_hz_tracking",
                                  "yolo_check_period_s_median", "yolo_check_period_s_expected",
                                  "yolo_check_period_s_p95", "yolo_busy_ms_median", "yolo_busy_ms_p95"]),
    ("Tracker", ["tracker_images", "tracker_hz_tracking", "tracker_busy_ms_median", "tracker_busy_ms_p95"]),
    ("Sound", ["sound_odas_hz", "sound_classifier_hz", "sound_warning_hz", "sound_drone_msgs",
               "sound_warning_latency_ms_median"]),
    ("Power (module VDD_IN)", ["P_VDD_IN_W_mean", "P_VDD_IN_W_max", "P_VDD_IN_W_tracking", "P_VDD_IN_W_searching",
                               "P_VDD_IN_W_tracking_yolo_on_s", "P_VDD_IN_W_tracking_yolo_off_s",
                               "energy_VDD_IN_Wh"]),
    ("Load", ["gpu_load_pct_tracking", "gpu_load_pct_searching", "cpu_total_pct_tracking", "cpu_total_pct_searching"]),
    ("Temperature", ["T_tj_C_mean", "T_tj_C_max", "T_tj_C_rise", "T_gpu_C_max", "T_cpu_C_max", "fan_pct_mean"]),
]


def main():
    d = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/rec_test1")
    acts = sorted(glob.glob(os.path.join(d, "**", "run_*_activity.csv"), recursive=True))
    if not acts:
        sys.exit(f"no run_*_activity.csv in {d} (record with the run_recorder that writes _activity.csv)")
    out = os.path.join(d, "activity")
    os.makedirs(out, exist_ok=True)
    runs = [load_run(p) for p in acts]
    summ = [stats(R) for R in runs]
    page = ["<html><head><meta charset='utf-8'><title>DALBD activity</title><style>"
            "body{font-family:sans-serif;margin:16px}table{border-collapse:collapse;font-size:12px;margin-bottom:8px}"
            "td,th{border:1px solid #ccc;padding:3px 6px;text-align:right}th{background:#eee}"
            "td.k{text-align:left;font-weight:bold;background:#fafafa}img{max-width:100%}</style></head><body>"
            "<h1>DALBD - when each part worked, and what it cost</h1>"
            "<p>Bars = processing window of each message (from the data stamp to the output). "
            "Ticks = outputs without a stamp. Bands = tracker status and fusion state. "
            "<b>yolo_check_period</b> = real time between two YOLO inferences while tracking "
            "(expected = N / tracker rate). <b>…_yolo_on_s / _off_s</b> = module power in tracking seconds "
            "with / without a YOLO inference.</p>"]
    page.append("<h2>All runs</h2>")
    for g, keys in GROUPS:
        page.append(f"<h3>{g}</h3><table><tr><th>run</th>" + "".join(f"<th>{k}</th>" for k in keys) + "</tr>")
        for s in summ:
            page.append(f"<tr><td class='k'>{html.escape(s['run'])}</td>"
                        + "".join(f"<td>{html.escape(str(s.get(k, '')))}</td>" for k in keys) + "</tr>")
        page.append("</table>")
    for R in runs:
        full_figure(R, os.path.join(out, R["name"] + "_timeline.png"))
        zoom_figure(R, os.path.join(out, R["name"] + "_zoom.png"))
        page.append(f"<h2>{html.escape(R['name'])}</h2><img src='{R['name']}_timeline.png'>"
                    f"<img src='{R['name']}_zoom.png'>")
    page.append("</body></html>")
    with open(os.path.join(out, "activity_report.html"), "w") as f:
        f.write("\n".join(page))
    cols = []
    for s in summ:
        cols += [k for k in s if k not in cols]
    with open(os.path.join(out, "activity_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(summ)
    print(f"report:  {os.path.join(out, 'activity_report.html')}\nsummary: {os.path.join(out, 'activity_summary.csv')}")


if __name__ == "__main__":
    main()

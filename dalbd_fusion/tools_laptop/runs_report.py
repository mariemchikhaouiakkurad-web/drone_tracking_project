#!/usr/bin/env python3
"""runs_report.py - LAPTOP: turn the run_recorder CSVs into one HTML page with charts (open it in a browser).

Usage:  python3 runs_report.py ~/dalbd_runs            -> ~/dalbd_runs/report/runs_report.html
Needs:  python3-matplotlib   (sudo apt install -y python3-matplotlib)

Page content:
  1. comparison of all runs (by detector_every N): power, energy per minute, GPU / CPU load, max temperature,
     YOLO / tracker / sound latency, rates, % of time tracking and in each fusion state
  2. table = runs_summary.csv
  3. per run: power, load, temperatures, latency and rate over time (one row of charts per run)
"""
import csv
import glob
import html
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def series(rows, col):
    x, y = [], []
    for r in rows:
        t, v = fnum(r.get("t_s")), fnum(r.get(col))
        if t is not None and v is not None:
            x.append(t)
            y.append(v)
    return x, y


def label(s):
    return f"N={s.get('detector_every', '?')}\n{s.get('start', '')[11:16]}"


def compare(summary, out):
    charts = [
        ("Power (W)", [("mean_P_VDD_IN_W", "VDD_IN mean"), ("tracking_mean_P_VDD_IN_W", "mean while tracking"),
                       ("max_P_VDD_IN_W", "VDD_IN max")]),
        ("Energy per minute (Wh/min)", [("_wh_per_min", "VDD_IN")]),
        ("Load (%)", [("mean_gpu_load_pct", "GPU"), ("mean_cpu_total_pct", "CPU")]),
        ("Max temperature (C)", [("max_T_tj_C", "tj"), ("max_T_gpu_C", "GPU"), ("max_T_cpu_C", "CPU")]),
        ("Latency median (ms)", [("p50_yolo_lat_ms", "YOLO"), ("p50_track_lat_ms", "tracker"), ("p50_sound_lat_ms", "sound"),
                                 ("mean_yolo_lat_ms", "YOLO mean")]),
        ("Rate (Hz)", [("mean_yolo_hz", "YOLO"), ("tracking_mean_yolo_hz", "YOLO while tracking"),
                       ("mean_track_hz", "tracker"), ("mean_sound_hz", "sound")]),
        ("Time share (%)", [("tracking_pct", "tracking"), ("visual_track_pct", "VISUAL_TRACK"),
                            ("sound_track_pct", "SOUND_TRACK"), ("search_pct", "SEARCH")]),
        ("Tracker (end of run)", [("lost_total", "lost"), ("reacquired_total", "reacquired"), ("corrected_total", "corrected")]),
    ]
    for s in summary:
        e, d = fnum(s.get("energy_VDD_IN_Wh")), fnum(s.get("duration_s"))
        s["_wh_per_min"] = round(e / d * 60.0, 4) if e and d else ""
    fig, axes = plt.subplots(2, 4, figsize=(20, 8.5))
    names = [label(s) for s in summary]
    for ax, (title, cols) in zip(axes.flat, charts):
        cols = [(c, l) for c, l in cols if any(fnum(s.get(c)) is not None for s in summary)]
        if not cols:
            ax.set_title(title + " (no data)")
            ax.axis("off")
            continue
        w = 0.8 / len(cols)
        for i, (c, l) in enumerate(cols):
            vals = [fnum(s.get(c)) or 0.0 for s in summary]
            xs = [k + (i - (len(cols) - 1) / 2) * w for k in range(len(summary))]
            b = ax.bar(xs, vals, w, label=l)
            ax.bar_label(b, fmt="%.3g", fontsize=7)
        ax.set_xticks(range(len(summary)))
        ax.set_xticklabels(names, fontsize=8)
        ax.set_title(title)
        ax.legend(fontsize=7)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("DALBD runs compared (one bar group per run)")
    fig.tight_layout()
    fig.savefig(out, dpi=90)
    plt.close(fig)


def per_run(rows, title, out):
    cols = rows[0].keys() if rows else []
    power = [c for c in cols if c.startswith("P_") and c.endswith("_W")]
    temps = [c for c in cols if c.startswith("T_") and c.endswith("_C")]
    parts = [c for c in cols if c.startswith("cpu_") and c.endswith("_pct") and c != "cpu_total_pct"]
    fig, axes = plt.subplots(1, 5, figsize=(25, 4))
    plots = [
        ("Power (W)", power),
        ("Load (%)", ["gpu_load_pct", "cpu_total_pct", "fan_pct"]),
        ("Temperature (C)", temps),
        ("Latency (ms)", ["yolo_lat_ms", "track_lat_ms", "sound_lat_ms"]),
        ("CPU per part (% of one core)", parts),
    ]
    for ax, (t, cs) in zip(axes, plots):
        for c in cs:
            x, y = series(rows, c)
            if y and (max(y) > 0 or t != "CPU per part (% of one core)"):
                ax.plot(x, y, label=c.replace("_pct", "").replace("_ms", "").replace("_W", "").replace("_C", ""), lw=1)
        ax.set_title(t)
        ax.set_xlabel("time in run (s)")
        ax.grid(alpha=0.3)
        if ax.lines:
            ax.legend(fontsize=6, ncol=2)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out, dpi=80)
    plt.close(fig)


def main():
    d = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/dalbd_runs")
    runs = sorted(p for p in glob.glob(os.path.join(d, "run_*.csv")) if not p.endswith(("_events.csv", "_activity.csv")))
    if not runs:
        sys.exit(f"no run_*.csv in {d}")
    rep = os.path.join(d, "report")
    os.makedirs(rep, exist_ok=True)
    sp = os.path.join(d, "runs_summary.csv")
    summary = read_csv(sp) if os.path.exists(sp) else []
    parts = ["<html><head><meta charset='utf-8'><title>DALBD runs</title><style>"
             "body{font-family:sans-serif;margin:16px}table{border-collapse:collapse;font-size:12px}"
             "td,th{border:1px solid #ccc;padding:3px 6px;text-align:right}th{background:#eee;position:sticky;top:0}"
             "img{max-width:100%}</style></head><body><h1>DALBD runs</h1>"]
    if summary:
        compare(summary, os.path.join(rep, "compare.png"))
        parts.append("<h2>1. Runs compared</h2><img src='compare.png'>")
        key = ["run_file", "detector_every", "start", "duration_s", "mean_P_VDD_IN_W", "max_P_VDD_IN_W",
               "energy_VDD_IN_Wh", "mean_gpu_load_pct", "mean_cpu_total_pct", "max_T_tj_C", "mean_yolo_hz",
               "mean_yolo_lat_ms", "p50_yolo_lat_ms", "tracking_mean_P_VDD_IN_W", "tracking_mean_yolo_hz", "mean_track_hz", "mean_track_lat_ms", "mean_trk_ms_mean", "mean_sound_lat_ms",
               "tracking_pct", "visual_track_pct", "sound_track_pct", "search_pct", "lost_total", "reacquired_total"]
        key = [k for k in key if any(s.get(k) not in (None, "") for s in summary)]
        parts.append("<h2>2. Main numbers per run</h2><table><tr>" + "".join(f"<th>{html.escape(k)}</th>" for k in key)
                     + "</tr>" + "".join("<tr>" + "".join(f"<td>{html.escape(str(s.get(k, '')))}</td>" for k in key)
                                         + "</tr>" for s in summary) + "</table>"
                     "<p>All columns: <code>runs_summary.csv</code> (open it in LibreOffice Calc).</p>")
    parts.append("<h2>3. Each run over time</h2>")
    for p in runs:
        rows = read_csv(p)
        if not rows:
            continue
        name = os.path.splitext(os.path.basename(p))[0]
        per_run(rows, f"{name}  ({len(rows)} s)", os.path.join(rep, name + ".png"))
        parts.append(f"<h3>{html.escape(name)}</h3><img src='{html.escape(name)}.png'>")
    parts.append("</body></html>")
    out = os.path.join(rep, "runs_report.html")
    with open(out, "w") as f:
        f.write("\n".join(parts))
    print(f"report: {out}\nopen:   xdg-open {out}")


if __name__ == "__main__":
    main()

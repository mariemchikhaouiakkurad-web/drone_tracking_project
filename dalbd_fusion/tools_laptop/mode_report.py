#!/usr/bin/env python3
"""mode_report.py - LAPTOP: cost and timing of each MODE of the pipeline, from the run_recorder CSVs.

Every second of a run is put in one vision mode (from the tracker status):
  SEARCH    YOLO runs on every image, tracker stopped          (track_status = searching)
  TRACKING  DyHiT on every image + YOLO once every N images    (track_status = tracking)
and, for sound, in:
  SOUND DRONE   the sound pipeline hears a drone               (sound_level = DRONE)
  SOUND CLEAR   sound running, no drone                        (sound_level = CLEAR)
  SOUND_TRACK   the fusion uses the sound direction (vision lost, drone heard)
For each mode: seconds, power (module and CPU+GPU rail), GPU / CPU load, CPU and RAM of every part (YOLO chain,
tracker, camera, fusion, sound rx / ODAS / classifier / warning), rates and median latencies, max temperature.

Usage:  python3 mode_report.py ~/rec_test1          -> ~/rec_test1/report/mode_report.html (+ mode_summary.csv)
Needs:  python3-matplotlib
Skips the first 10 s of every run (start-up); change with --warmup 20.
"""
import csv
import glob
import html
import os
import statistics as st
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PARTS = ["yolo_chain", "tracker", "net_camera", "fusion", "tools", "sound_rx", "sound_odas", "sound_classifier",
         "sound_warning"]
PART_LABEL = {"yolo_chain": "YOLO", "tracker": "tracker", "net_camera": "camera rx", "fusion": "fusion",
              "tools": "tools", "sound_rx": "sound rx", "sound_odas": "ODAS", "sound_classifier": "classifier",
              "sound_warning": "warning"}
VISION_MODES = [("SEARCH", "YOLO every image", lambda r: r.get("track_status") == "searching"),
                ("TRACKING", "DyHiT + YOLO every N", lambda r: r.get("track_status") == "tracking")]
SOUND_MODES = [("SOUND DRONE", "drone heard", lambda r: r.get("sound_level") == "DRONE"),
               ("SOUND CLEAR", "sound on, no drone", lambda r: r.get("sound_level") == "CLEAR"),
               ("SOUND_TRACK", "fusion uses sound", lambda r: r.get("fusion_state") == "SOUND_TRACK")]
COLORS = {"SEARCH": "#d62728", "TRACKING": "#1f77b4", "SOUND DRONE": "#ff7f0e", "SOUND CLEAR": "#7f7f7f",
          "SOUND_TRACK": "#2ca02c"}


def f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def vals(rows, col):
    return [x for x in (f(r.get(col)) for r in rows) if x is not None]


def mean(v):
    return round(sum(v) / len(v), 2) if v else ""


def med(v):
    return round(st.median(v), 1) if v else ""


def stats(rows, label, desc, n):
    o = {"N": n, "mode": label, "what": desc, "seconds": len(rows)}
    o["P_module_W"] = mean(vals(rows, "P_VDD_IN_W"))
    o["P_cpu_gpu_W"] = mean(vals(rows, "P_VDD_CPU_GPU_CV_W"))
    o["gpu_load_pct"] = mean(vals(rows, "gpu_load_pct"))
    o["cpu_total_pct"] = mean(vals(rows, "cpu_total_pct"))
    o["ram_used_MB"] = mean(vals(rows, "ram_used_MB"))
    t = vals(rows, "T_tj_C") or vals(rows, "T_tjal_C") or vals(rows, "T_cpu_C")
    o["T_max_C"] = round(max(t), 1) if t else ""
    for p in PARTS:
        o[f"cpu_{p}_pct"] = mean(vals(rows, f"cpu_{p}_pct"))
        o[f"ram_{p}_MB"] = mean(vals(rows, f"ram_{p}_MB"))
    for k in ("yolo", "track", "sound"):
        o[f"{k}_hz"] = mean(vals(rows, f"{k}_hz"))
        o[f"{k}_lat_p50_ms"] = med(vals(rows, f"{k}_lat_ms"))
        lm = vals(rows, f"{k}_lat_max_ms")
        o[f"{k}_lat_max_ms"] = round(max(lm), 1) if lm else ""
    o["sound_prob_mean"] = mean(vals(rows, "sound_prob"))
    return o


def segments(rows):
    """contiguous blocks of the same vision mode -> [(mode, t_start, t_end)]"""
    seg, cur, t0, tp = [], None, None, None
    for r in rows:
        t = f(r.get("t_s"))
        m = r.get("track_status") or "?"
        if t is None:
            continue
        if m != cur:
            if cur is not None:
                seg.append((cur, t0, tp))
            cur, t0 = m, t
        tp = t
    if cur is not None:
        seg.append((cur, t0, tp))
    return seg


def timeline(rows, title, png):
    fig, axes = plt.subplots(4, 1, figsize=(16, 9), sharex=True)
    for m, a, b in segments(rows):
        c = {"searching": COLORS["SEARCH"], "tracking": COLORS["TRACKING"]}.get(m, "#cccccc")
        for ax in axes:
            ax.axvspan(a, b + 1, color=c, alpha=0.12, lw=0)
    for r in rows:                                       # sound DRONE marks on the top axis
        if r.get("sound_level") == "DRONE" and f(r.get("t_s")) is not None:
            axes[0].axvspan(f(r["t_s"]), f(r["t_s"]) + 1, ymin=0.9, ymax=1.0, color=COLORS["SOUND DRONE"])
    plots = [("Power (W)", [("P_VDD_IN_W", "module"), ("P_VDD_CPU_GPU_CV_W", "CPU+GPU")]),
             ("Load (%)", [("gpu_load_pct", "GPU"), ("cpu_total_pct", "CPU")] +
              [(f"cpu_{p}_pct", f"CPU {PART_LABEL[p]}") for p in ("yolo_chain", "tracker", "sound_odas", "sound_classifier")]),
             ("Rate (Hz)", [("yolo_hz", "YOLO"), ("track_hz", "tracker"), ("sound_hz", "sound")]),
             ("Latency (ms)", [("yolo_lat_ms", "YOLO"), ("track_lat_ms", "tracker"), ("sound_lat_ms", "sound")])]
    for ax, (t, cols) in zip(axes, plots):
        for c, lab in cols:
            x = [f(r.get("t_s")) for r in rows if f(r.get(c)) is not None and f(r.get("t_s")) is not None]
            y = [f(r.get(c)) for r in rows if f(r.get(c)) is not None and f(r.get("t_s")) is not None]
            if y and max(y) > 0:
                ax.plot(x, y, lw=1, label=lab)
        ax.set_ylabel(t)
        ax.grid(alpha=0.3)
        if ax.lines:
            ax.legend(fontsize=7, ncol=6, loc="upper right")
    if axes[3].lines:
        lat = [v for c in ("yolo_lat_ms", "track_lat_ms", "sound_lat_ms") for v in vals(rows, c)]
        if lat:
            axes[3].set_ylim(0, min(max(lat), max(300.0, 3 * st.median(lat))))   # spikes above are cut
    axes[-1].set_xlabel("time in run (s)   background: red = SEARCH (YOLO every image), blue = TRACKING; "
                        "orange marks at top = sound hears a drone")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(png, dpi=80)
    plt.close(fig)


def bars(summary, png):
    pooled = [s for s in summary if s["N"] == "all" and s["seconds"]]
    if not pooled:
        return False
    names = [s["mode"] for s in pooled]
    cols = [COLORS.get(n, "#999999") for n in names]
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.8))
    for ax, (k, t) in zip(axes[:2], (("P_module_W", "Module power (W)"), ("gpu_load_pct", "GPU load (%)"))):
        v = [f(s[k]) or 0 for s in pooled]
        b = ax.bar(names, v, color=cols)
        ax.bar_label(b, fmt="%.3g", fontsize=8)
        ax.set_title(t)
        ax.tick_params(axis="x", labelsize=8, rotation=15)
    ax = axes[2]                                         # CPU per part, stacked
    bottom = [0.0] * len(pooled)
    for p in PARTS:
        v = [f(s[f"cpu_{p}_pct"]) or 0 for s in pooled]
        if max(v) > 0:
            ax.bar(names, v, bottom=bottom, label=PART_LABEL[p])
            bottom = [a + b for a, b in zip(bottom, v)]
    ax.set_title("CPU per part (% of one core, stacked)")
    ax.legend(fontsize=7)
    ax.tick_params(axis="x", labelsize=8, rotation=15)
    ax = axes[3]
    w = 0.27
    for i, (k, lab) in enumerate((("yolo", "YOLO"), ("track", "tracker"), ("sound", "sound"))):
        v = [f(s[f"{k}_lat_p50_ms"]) or 0 for s in pooled]
        b = ax.bar([j + (i - 1) * w for j in range(len(pooled))], v, w, label=lab)
        ax.bar_label(b, fmt="%.0f", fontsize=7)
    ax.set_xticks(range(len(pooled)))
    ax.set_xticklabels(names, fontsize=8, rotation=15)
    ax.set_title("Median latency (ms)")
    ax.legend(fontsize=7)
    for a in axes:
        a.grid(axis="y", alpha=0.3)
    fig.suptitle("Cost of each mode (all runs together)")
    fig.tight_layout()
    fig.savefig(png, dpi=85)
    plt.close(fig)
    return True


def table(rows, keys):
    keys = [k for k in keys if any(r.get(k) not in ("", None) for r in rows)]
    h = "<table><tr>" + "".join(f"<th>{html.escape(k)}</th>" for k in keys) + "</tr>"
    for r in rows:
        h += "<tr>" + "".join(f"<td>{html.escape(str(r.get(k, '')))}</td>" for k in keys) + "</tr>"
    return h + "</table>"


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    warm = 10.0
    if "--warmup" in sys.argv:
        warm = float(sys.argv[sys.argv.index("--warmup") + 1])
        args = [a for a in args if a != sys.argv[sys.argv.index("--warmup") + 1]]
    d = os.path.expanduser(args[0] if args else "~/dalbd_runs")
    runs = sorted(p for p in glob.glob(os.path.join(d, "run_*.csv")) if not p.endswith("_events.csv"))
    if not runs:
        sys.exit(f"no run_*.csv in {d}")
    rep = os.path.join(d, "report")
    os.makedirs(rep, exist_ok=True)
    summary, allrows, pics = [], [], []
    for p in runs:
        with open(p, newline="") as fh:
            rows = [r for r in csv.DictReader(fh) if (f(r.get("t_s")) or 0) >= warm]
        if not rows:
            continue
        n = rows[0].get("detector_every", "?")
        allrows += rows
        for label, desc, sel in VISION_MODES + SOUND_MODES:
            summary.append(stats([r for r in rows if sel(r)], label, desc, n))
        name = os.path.splitext(os.path.basename(p))[0]
        timeline(rows, f"{name}  (N = {n}, {len(rows)} s after the first {warm:.0f} s)", os.path.join(rep, name + "_modes.png"))
        pics.append(name)
    for label, desc, sel in VISION_MODES + SOUND_MODES:
        summary.append(stats([r for r in allrows if sel(r)], label, desc, "all"))
    out_csv = os.path.join(rep, "mode_summary.csv")
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    have_bars = bars(summary, os.path.join(rep, "modes_compare.png"))
    main_keys = ["N", "mode", "what", "seconds", "P_module_W", "P_cpu_gpu_W", "gpu_load_pct", "cpu_total_pct", "T_max_C",
                 "yolo_hz", "yolo_lat_p50_ms", "track_hz", "track_lat_p50_ms", "sound_hz", "sound_lat_p50_ms",
                 "sound_prob_mean"]
    part_keys = ["N", "mode", "seconds"] + [f"cpu_{p}_pct" for p in PARTS] + [f"ram_{p}_MB" for p in PARTS]
    pooled = [s for s in summary if s["N"] == "all"]
    per_run = [s for s in summary if s["N"] != "all" and s["seconds"]]
    page = ["<html><head><meta charset='utf-8'><title>DALBD modes</title><style>body{font-family:sans-serif;margin:16px}"
            "table{border-collapse:collapse;font-size:12px;margin-bottom:12px}td,th{border:1px solid #ccc;padding:3px 6px;"
            "text-align:right}th{background:#eee}img{max-width:100%}</style></head><body><h1>DALBD — cost and timing of each mode</h1>"
            "<p>SEARCH = YOLO on every image (tracker stopped). TRACKING = DyHiT on every image + YOLO once every N images. "
            "Sound modes overlap with the vision modes (sound runs in parallel). First "
            f"{warm:.0f} s of every run left out.</p>"]
    if have_bars:
        page.append("<h2>1. Modes compared (all runs)</h2><img src='modes_compare.png'>")
    page.append("<h2>2. All runs together</h2>" + table(pooled, main_keys))
    page.append("<h3>Load of each part (CPU % of one core, RAM MB)</h3>" + table(pooled, part_keys))
    page.append("<h2>3. Per run (per N)</h2>" + table(per_run, main_keys))
    page.append("<h3>Load of each part per run</h3>" + table(per_run, part_keys))
    page.append("<h2>4. Each run over time</h2>")
    for name in pics:
        page.append(f"<h3>{html.escape(name)}</h3><img src='{html.escape(name)}_modes.png'>")
    page.append(f"<p>Numbers: <code>{html.escape(out_csv)}</code></p></body></html>")
    out = os.path.join(rep, "mode_report.html")
    with open(out, "w") as fh:
        fh.write("\n".join(page))
    print(f"report: {out}\ntable:  {out_csv}\nopen:   xdg-open {out}")


if __name__ == "__main__":
    main()

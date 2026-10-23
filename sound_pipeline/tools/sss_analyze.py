#!/usr/bin/env python3
"""Analyse one sss_recorder take: what MobileNet received (before) and what is left after the drop.

    python3 tools/sss_analyze.py                       latest take in ~/dataset/sss_rec
    python3 tools/sss_analyze.py ~/dataset/sss_rec/sss_20261016_100430
    python3 tools/sss_analyze.py --selftest

Report:
  1. per slot: level before / after (dBFS), dominant frequencies
  2. timeline in 5 s windows: loudest slot before vs after-mix level (where sound reached MobileNet)
  3. every track: time span, direction range, level, DROPPED / KEPT share
  4. verdict for a known-source-only take
Also writes <take>_report.png (spectrograms before / after) when matplotlib is installed.
"""
import collections
import csv
import glob
import math
import os
import sys
import wave

import numpy as np

SILENT = -100.0


def load(path):
    with wave.open(path) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), "<i2").reshape(-1, w.getnchannels())
        return x.astype(np.float32) / 32768.0, w.getframerate()


def dbfs(v):
    if v.size == 0:
        return -120.0
    r = float(np.sqrt(np.mean(v ** 2)))
    return 20 * math.log10(r) if r > 1e-6 else -120.0


def peaks(v, fs, n=3):
    if dbfs(v) < SILENT or v.size < 1024:
        return []
    seg = v[: fs * 10]
    S = np.abs(np.fft.rfft(seg * np.hanning(seg.size)))
    f = np.fft.rfftfreq(seg.size, 1.0 / fs)
    S[f < 80] = 0
    out = []
    for i in np.argsort(S)[::-1]:
        if all(abs(f[i] - g) > 40 for g in out):
            out.append(float(f[i]))
        if len(out) == n:
            break
    return out


def latest(d):
    c = sorted(glob.glob(os.path.join(d, "sss_*_before_4ch.wav")))
    return c[-1][: -len("_before_4ch.wav")] if c else None


def analyse(base):
    before, fs = load(base + "_before_4ch.wav")
    after, _ = load(base + "_after_4ch.wav")
    mix, _ = load(base + "_after_mix.wav")
    dur = before.shape[0] / fs
    L = [f"take {os.path.basename(base)}   {dur:.1f} s   {before.shape[1]} slots @ {fs} Hz", ""]
    L.append("1) per slot            before      after    dominant frequencies (before)")
    for c in range(before.shape[1]):
        b, a = dbfs(before[:, c]), dbfs(after[:, c])
        pk = ", ".join(f"{p:.0f} Hz" for p in peaks(before[:, c], fs)) or "-"
        L.append(f"   slot {c}          {b:7.1f}   {a:7.1f}    {pk}")
    L.append(f"   after-drop mix                {dbfs(mix[:, 0]):7.1f}")
    L.append("")
    L.append("2) timeline (5 s): loudest slot BEFORE | AFTER mix   (# = sound present)")
    win = fs * 5
    heard_b = heard_a = 0
    for i in range(0, before.shape[0], win):
        bl = max(dbfs(before[i:i + win, c]) for c in range(before.shape[1]))
        al = dbfs(mix[i:i + win, 0])
        heard_b += bl > SILENT
        heard_a += al > SILENT
        L.append(f"   {i / fs:5.0f}-{min(dur, (i + win) / fs):5.0f} s   {bl:7.1f} {'#' if bl > SILENT else ' '} | "
                 f"{al:7.1f} {'#' if al > SILENT else ' '}")
    nwin = math.ceil(before.shape[0] / win)
    L.append("")
    L.append("3) tracks seen by MobileNet's input")
    rows = []
    if os.path.exists(base + "_tracks.csv"):
        rows = [r for r in csv.DictReader(open(base + "_tracks.csv")) if r["id"] != "0" and float(r["activity"]) >= 0.1]
    kept_rows = 0
    if not rows:
        L.append("   none: no active track during the whole take")
    by = collections.defaultdict(list)
    for r in rows:
        by[r["id"]].append(r)
    for tid, rs in by.items():
        az = [float(r["az_deg"]) for r in rs]
        el = [float(r["el_deg"]) for r in rs]
        lv = [float(r["level_dbfs"]) for r in rs]
        dr = sum(r.get("dropped", "0") == "1" for r in rs)
        kept_rows += len(rs) - dr
        L.append(f"   id {tid:>4}  t {float(rs[0]['t']):5.1f}-{float(rs[-1]['t']):5.1f} s  az {min(az):+6.0f}..{max(az):+4.0f}"
                 f"  el {min(el):+4.0f}..{max(el):+3.0f}  {np.mean(lv):6.1f} dBFS  "
                 f"dropped {100 * dr / len(rs):3.0f}%  kept {len(rs) - dr}x")
    L.append("")
    L.append("4) verdict (known source only)")
    if heard_b == 0:
        L.append("   MobileNet's input was SILENT the whole take: the null removed the known source completely,")
        L.append("   nothing had to be dropped. Valid only if the known source really played (check /raw/sst),")
        L.append("   and only if a second source (target) DOES appear in a take where it plays.")
    elif heard_a == 0 and kept_rows == 0:
        L.append(f"   The known source leaked into MobileNet's input in {heard_b}/{nwin} windows, and the drop removed")
        L.append("   ALL of it: after-drop is silent. PASS.")
    else:
        L.append(f"   Sound remained AFTER the drop in {heard_a}/{nwin} windows ({kept_rows} kept track-seconds):")
        L.append("   with the known source only, that is leakage the drop missed -> see the KEPT tracks in 3)")
        L.append("   (direction outside the cone: reflection / echo / zenith ghost).")
    return "\n".join(L), before, mix, fs


def plot(base, before, mix, fs):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    n = before.shape[1] + 1
    fig, ax = plt.subplots(n, 1, figsize=(11, 1.8 * n), sharex=True)
    sigs = [(before[:, c], f"before slot {c}") for c in range(before.shape[1])] + [(mix[:, 0], "AFTER drop (mix)")]
    for a, (s, t) in zip(ax, sigs):
        if dbfs(s) > SILENT:
            a.specgram(s + 1e-9, NFFT=512, Fs=fs, noverlap=256, cmap="magma", vmin=-160, vmax=-60)
        else:
            a.text(0.5, 0.5, "silent", ha="center", va="center", transform=a.transAxes)
        a.set_ylabel(t, fontsize=8)
        a.set_ylim(0, fs / 2)
    ax[-1].set_xlabel("s")
    fig.tight_layout()
    out = base + "_report.png"
    fig.savefig(out, dpi=90)
    return out


def selftest():
    import tempfile
    d = tempfile.mkdtemp()
    base = os.path.join(d, "sss_test")
    fs, n = 16000, 16000 * 12
    t = np.arange(n) / fs
    b = np.zeros((n, 4), np.float32)
    b[: n // 2, 0] = 0.01 * np.sin(2 * np.pi * 2330 * t[: n // 2])     # known leftover, first 6 s
    a = b.copy()
    a[:, 0] = 0                                                            # dropped
    for name, x in (("_before_4ch", b), ("_after_4ch", a), ("_after_mix", a.sum(1, keepdims=True))):
        with wave.open(base + name + ".wav", "wb") as w:
            w.setnchannels(x.shape[1]); w.setsampwidth(2); w.setframerate(fs)
            w.writeframes((x * 32767).astype("<i2").tobytes())
    with open(base + "_tracks.csv", "w") as f:
        f.write("t,slot,id,az_deg,el_deg,activity,level_dbfs,dropped\n")
        for s in range(6):
            f.write(f"{s}.0,0,4,-175.0,9.0,0.9,-43.0,1\n")
    rep, before, mix, fs2 = analyse(base)
    print(rep)
    assert "2330 Hz" in rep and "PASS" in rep and "dropped 100%" in rep
    print("selftest PASSED")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
        sys.exit(0)
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    base = arg if arg else latest(os.path.expanduser("~/dataset/sss_rec"))
    if not base:
        sys.exit("no take found in ~/dataset/sss_rec")
    for suf in ("_before_4ch.wav", "_after_4ch.wav", "_after_mix.wav", "_tracks.csv"):
        if base.endswith(suf):
            base = base[: -len(suf)]
    rep, before, mix, fs = analyse(os.path.expanduser(base))
    print(rep)
    png = plot(os.path.expanduser(base), before, mix, fs)
    print(f"\nspectrograms: {png}" if png else "\n(matplotlib not installed: no spectrogram PNG)")

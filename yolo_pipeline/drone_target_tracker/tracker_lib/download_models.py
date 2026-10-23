#!/usr/bin/env python3
"""
download_models.py - fetch the trackers' model files into tracker_lib/trackers/*/models/ and CLIP into ~/.cache/clip,
so the node starts without internet afterwards. Run it inside the container (it needs the Python packages of
docker/Dockerfile.tracker):

    python3 /workspaces/isaac_ros-dev/src/drone_target_tracker/tracker_lib/download_models.py
    ... --skip-clip       (CLIP is then downloaded by SUTrack on its first start)

  SUTrack-T224  1.0 GB  Hugging Face (authors)      HiT-Base  511 MB  Google Drive (authors, needs gdown)
  DyHiT         182 MB  Google Drive (authors)      CLIP ViT-L/14  ~900 MB  (used inside SUTrack)
Files already there are skipped.
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def run(name):
    print(f"\n== {name}", flush=True)
    return subprocess.run([sys.executable, os.path.join(HERE, "trackers", name, "download_model.py")]).returncode == 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-clip", action="store_true")
    args = ap.parse_args()
    ok = {n: run(n) for n in ("sutrack", "hit_base", "dyhit")}
    if not args.skip_clip:
        print("\n== CLIP ViT-L/14", flush=True)
        try:
            import clip
            clip.load("ViT-L/14", device="cpu")
            ok["CLIP"] = True
        except Exception as e:
            print(f"   FAILED: {e}")
            ok["CLIP"] = False
    print("\n" + "\n".join(f"  {'OK     ' if v else 'MISSING'}  {k}" for k, v in ok.items()))
    sys.exit(0 if all(ok.values()) else 1)


if __name__ == "__main__":
    main()

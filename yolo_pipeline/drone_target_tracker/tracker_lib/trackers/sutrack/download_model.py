#!/usr/bin/env python3
"""Download the SUTrack-T224 checkpoint (1 GB) into trackers/sutrack/models/ from the authors' Hugging Face page
(https://huggingface.co/xche32/SUTrack, linked from external/SUTrack/MODEL_ZOO.md). Same file as the one used on the
laptop: size 1,044,947,583 bytes, sha256 starts with 5815a447480fba64."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from common.download import fetch  # noqa: E402

URL = ("https://huggingface.co/xche32/SUTrack/resolve/main/"
       "checkpoints/train/sutrack/sutrack_t224/SUTRACK_ep0180.pth.tar")

if __name__ == "__main__":
    sys.exit(0 if fetch({"SUTRACK_t224.pth.tar": URL}, os.path.join(HERE, "models")) else 1)

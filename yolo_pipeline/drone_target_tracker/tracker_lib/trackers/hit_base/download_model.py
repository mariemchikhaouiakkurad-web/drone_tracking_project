#!/usr/bin/env python3
"""Download the hit_base checkpoint into trackers/hit_base/models/ (Google Drive, needs gdown)."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from common.download import fetch_gdrive  # noqa: E402

FOLDER = "https://drive.google.com/drive/folders/15VTIJnUtJTdU6TcmGOixSEcErYV-h_xL"
FILES = {"HiT_Base.pth.tar": "1pK5Fo1-TNg0f_KbCndAXUDdprImb5zzt"}

if __name__ == "__main__":
    sys.exit(0 if fetch_gdrive(FILES, os.path.join(HERE, "models"), FOLDER) else 1)

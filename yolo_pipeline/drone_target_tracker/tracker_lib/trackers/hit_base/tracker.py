"""hit_base: HiT-Base (ICCV 2023, MIT), hierarchical transformer on a LeViT backbone; the most accurate HiT
(61 FPS Jetson AGX, LaSOT 64.6). PyTorch, GPU or CPU. Setup: trackers/hit_base/README.md."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from common.hit_family import make_cached  # noqa: E402


def make(cfg):
    return make_cached(cfg, "hit_base")

"""sutrack: SUTrack (AAAI 2025, MIT), one-stream transformer; our current pick (SUTrack-T224: LaSOT 69.6,
34 FPS on Jetson AGX). Needs an NVIDIA GPU, a CUDA build of PyTorch, a clone of the SUTrack repo and a
checkpoint: set "sutrack_repo", "sutrack_config", "sutrack_ckpt" in config.json. See trackers/README.md."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from common.bench_core import BaseTracker, cv2, rootpath  # noqa: E402


class Tracker(BaseTracker):
    name = "sutrack"

    def __init__(self, repo, config, ckpt):
        repo = os.path.abspath(os.path.expanduser(repo or ""))
        if not os.path.isdir(os.path.join(repo, "lib")):
            sys.exit('"sutrack_repo" in config.json must point to a clone of https://github.com/chenxin-dlut/SUTrack')
        if not ckpt or not os.path.exists(ckpt):
            sys.exit('"sutrack_ckpt" in config.json must point to a SUTrack checkpoint (.pth.tar) from the model zoo')
        sys.path.insert(0, repo)
        try:
            import torch
        except ImportError:
            sys.exit("SUTrack needs PyTorch, which is only in its own environment. Run it with\n"
                     "  " + os.path.join(".venv-sutrack", "Scripts" if os.name == "nt" else "bin", "python")
                     + " " + " ".join(sys.argv))
        if not torch.cuda.is_available():
            sys.exit("SUTrack's tracker code runs on CUDA only (NVIDIA GPU + CUDA build of PyTorch)")
        # PyTorch >= 2.6 loads checkpoints with weights_only=True by default, which rejects the extra
        # objects stored in SUTrack's .pth.tar. The file comes from the SUTrack authors, so allow it here.
        _orig_load = torch.load

        def _load(*a, **k):
            k.setdefault("weights_only", False)
            return _orig_load(*a, **k)
        torch.load = _load
        # Building the network normally also loads Fast-iTPN pre-training weights from pretrained/itpn/.
        # For inference the SUTrack checkpoint overwrites them, so skip that step.
        import lib.models.sutrack.encoder as sut_encoder
        sut_encoder.is_main_process = lambda: False
        from lib.test.utils import TrackerParams
        from lib.config.sutrack.config import cfg, update_config_from_file
        from lib.test.tracker.sutrack import SUTRACK

        update_config_from_file(os.path.join(repo, "experiments", "sutrack", config + ".yaml"))
        p = TrackerParams()
        p.cfg = cfg
        p.yaml_name = config
        p.template_factor = cfg.TEST.TEMPLATE_FACTOR
        p.template_size = cfg.TEST.TEMPLATE_SIZE
        p.search_factor = cfg.TEST.SEARCH_FACTOR
        p.search_size = cfg.TEST.SEARCH_SIZE
        p.checkpoint = ckpt
        p.save_all_boxes = False
        p.debug = 0
        self.torch = torch
        self.device = f"cuda ({torch.cuda.get_device_name(0)})"
        self.t = SUTRACK(p, "default")
        self.name = "sutrack_" + config.replace("sutrack_", "")

    @staticmethod
    def _rgb(frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if cv2 is not None else frame[:, :, ::-1].copy()

    def init(self, frame, box):
        self.t.initialize(self._rgb(frame), {"init_bbox": list(box)})

    def update(self, frame):
        out = self.t.track(self._rgb(frame))
        self.torch.cuda.synchronize()  # so the timing includes the GPU work
        try:
            score = float(out.get("best_score"))
        except Exception:
            score = None
        return True, tuple(float(v) for v in out["target_bbox"]), score

_CACHE = {}


def make(cfg):
    # building SUTrack is slow (model + CLIP): build once per process, re-initialise per sequence
    key = (cfg.get("sutrack_repo"), cfg.get("sutrack_config", "sutrack_t224"), cfg.get("sutrack_ckpt"))
    if key not in _CACHE:
        _CACHE[key] = Tracker(rootpath(key[0]), key[1], rootpath(key[2]))
    return _CACHE[key]

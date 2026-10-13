# The three trackers

All three are single-object trackers: `init(frame, box)` once, then `update(frame) -> (ok, box, score)` on every
frame. `tracker.py` in each folder wraps the authors' own test-time code (in `../external/`) so that pipeline.py can
use them all the same way. The checkpoints go in each `models/` folder: they are not in git, `python
download_models.py` (in the folder above) downloads all three.

| tracker | paper / code | LaSOT AUC (published) | published speed* | weights licence | checkpoint |
|---|---|---|---|---|---|
| `sutrack` (SUTrack-T224) | AAAI 2025, [chenxin-dlut/SUTrack](https://github.com/chenxin-dlut/SUTrack), MIT | 69.6 | 34 FPS Jetson AGX | MIT | `sutrack/models/SUTRACK_t224.pth.tar` (1 GB) |
| `hit_base` (HiT-Base) | ICCV 2023, [kangben258/HiT](https://github.com/kangben258/HiT), MIT | 64.6 | 61 FPS Jetson AGX | MIT (LeViT, Apache-2.0, start) | `hit_base/models/HiT_Base.pth.tar` (511 MB) |
| `dyhit` (DyHiT) | same repo as HiT, MIT | 62.4 | 111 FPS Jetson AGX | MIT | `dyhit/models/DyHiT_ep0060.pth.tar` (182 MB) |

\* Published figures, on other boards and settings: only compare timings measured on the same machine, as
pipeline.py does (`trk_ms` = the tracker's own time per frame).

Settings (checkpoint paths, device, lost-score thresholds): `../config.json`.

## Things to know
- **SUTrack**: CUDA only (NVIDIA GPU). The first run downloads CLIP ViT-L/14 (about 900 MB, cached in the user's
  home folder) because the model contains a text encoder the checkpoint expects; it is not used for tracking.
  It has a confidence score (`lost_score` 0.3 in config.json), which makes the pipeline run the detector at once
  when it drops. The Fast-iTPN pre-training weights are not needed (the checkpoint replaces them).
- **hit_base, dyhit**: GPU or CPU (`hit_base_device` / `dyhit_device`: auto, cuda, cpu). They have **no score**
  (`lost_score` null), so only the periodic detector check notices a lost target.
- **Downloading one checkpoint**: `python trackers/<name>/download_model.py`. SUTrack comes from the authors'
  Hugging Face page (https://huggingface.co/xche32/SUTrack, file `sutrack_t224/SUTRACK_ep0180.pth.tar`, saved as
  `SUTRACK_t224.pth.tar`); HiT-Base and DyHiT from the HiT authors' Google Drive folder
  (https://drive.google.com/drive/folders/15VTIJnUtJTdU6TcmGOixSEcErYV-h_xL, needs gdown).
- **`../external/`** holds the two research repos without their git history: SUTrack at commit `d65052d1`
  (github.com/chenxin-dlut/SUTrack, 2025-06-16) and HiT at commit `ca806400` (github.com/kangben258/HiT,
  2025-09-26), each with its MIT LICENSE file. They are used unchanged;
  repo problems are worked around in `../common/` (`torch_tracker.py`, `hit_family.py`) and in `tracker.py`:
  checkpoints are loaded with `weights_only=False` (PyTorch >= 2.6), training-only imports are replaced by stand-ins
  (`torch._six`, `jpeg4py`, `visdom`), and no LeViT / Fast-iTPN pre-training download is attempted.
- Every result summary records the `tracker_device` it ran on.

"""mnv2_core.py - MobileNetV2 drone-sound classifier, without ROS.

audio block (n, channels) -> mic selection + mono mix -> sliding window -> resample
-> log-mel spectrogram -> model input tensor -> MobileNetV2 (ONNX / TensorRT / TorchScript)
-> drone probability -> N-of-M smoothing.

Used by nodes/drone_classifier_node.py (live) and tools/classify_wav.py (offline).
All preprocessing values come from the params file (uma-8/config/drone_classifier.yaml):
they MUST match what the model was trained with.
"""
import json
import os
import time
from collections import deque
from dataclasses import dataclass, field, fields
from math import gcd

import numpy as np


# --------------------------------------------------------------------------- config
@dataclass
class ClassifierConfig:
    # model
    model_path: str = ""
    backend: str = "auto"            # auto | onnx | tensorrt | torch
    onnx_providers: list = field(default_factory=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])
    output_activation: str = "softmax"  # softmax | sigmoid | none
    drone_index: int = 1             # index of the "drone" class in the model output
    class_names: list = field(default_factory=lambda: ["no_drone", "drone"])

    # audio input (what the feeder writes / what the WAV contains)
    fs_in: int = 48000
    channels: int = 8
    mic_channels: list = field(default_factory=lambda: [1, 2, 3, 4, 5, 6, 7])  # 0-based
    dtype: str = "int16"             # int16 | int32 | float32

    # windowing
    win_s: float = 1.0
    hop_s: float = 0.5

    # model input: "waveform" = raw audio [1, model_fs*win_s] (mel computed inside the model)
    #              "logmel"   = spectrogram computed here with the feature settings below
    input_type: str = "waveform"
    wave_norm: str = "none"          # waveform only: none | peak (scale to max |x| = 1) | rms (to -20 dBFS)

    # features (must match training)
    model_fs: int = 16000
    n_fft: int = 1024
    win_length: int = 1024
    hop_length: int = 256
    n_mels: int = 64
    fmin: float = 50.0
    fmax: float = 8000.0
    center: bool = True
    pad_mode: str = "constant"       # librosa >= 0.10 default
    mel_backend: str = "auto"        # auto | librosa | numpy
    log_scale: str = "db"            # db (power_to_db, ref=1) | ln (log(S + eps)) | none
    top_db: float = 80.0

    # tensor layout
    time_major: bool = False         # False: (mels, frames) ; True: (frames, mels)
    norm: str = "clip"               # none | clip (per-window mean/std) | fixed | minmax | imagenet
    norm_mean: float = 0.0
    norm_std: float = 1.0
    resize: list = field(default_factory=list)   # [] or [H, W], e.g. [224, 224]
    in_channels: int = 1             # 1 or 3 (3 = replicate, for ImageNet MobileNetV2)
    layout: str = "NCHW"             # NCHW | NHWC

    # decision
    threshold: float = 0.5
    smooth_n: int = 3                # N windows over threshold ...
    smooth_m: int = 5                # ... among the last M -> is_drone

    @classmethod
    def from_dict(cls, d):
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


# --------------------------------------------------------------------------- audio
_SCALE = {"int16": 32768.0, "int32": 2147483648.0, "float32": 1.0}


def bytes_to_frames(buf, cfg):
    """raw interleaved PCM bytes -> float32 (n, channels)."""
    x = np.frombuffer(buf, dtype=np.dtype(cfg.dtype))
    n = x.size // cfg.channels
    x = x[: n * cfg.channels].reshape(n, cfg.channels).astype(np.float32)
    return x / _SCALE[cfg.dtype]


def to_mono(frames, cfg):
    """(n, ch) -> (n,) mean of the microphone channels."""
    mics = [c for c in cfg.mic_channels if c < frames.shape[1]]
    if not mics:
        raise ValueError(f"mic_channels {cfg.mic_channels} not in a {frames.shape[1]}-channel stream")
    return frames[:, mics].mean(axis=1)


def resample(x, fs_in, fs_out):
    if fs_in == fs_out:
        return x
    try:
        from scipy.signal import resample_poly
        g = gcd(int(fs_in), int(fs_out))
        return resample_poly(x, int(fs_out) // g, int(fs_in) // g).astype(np.float32)
    except ImportError:
        n_out = int(round(len(x) * fs_out / fs_in))
        return np.interp(np.linspace(0, len(x) - 1, n_out), np.arange(len(x)), x).astype(np.float32)


class Windower:
    """Collects mono samples at fs_in, yields win_s windows every hop_s."""

    def __init__(self, cfg):
        self.win = int(round(cfg.win_s * cfg.fs_in))
        self.hop = int(round(cfg.hop_s * cfg.fs_in))
        self.buf = np.zeros(0, dtype=np.float32)

    def push(self, mono):
        self.buf = np.concatenate([self.buf, mono.astype(np.float32)])
        out = []
        while len(self.buf) >= self.win:
            out.append(self.buf[: self.win].copy())
            self.buf = self.buf[self.hop:]
        return out


# --------------------------------------------------------------------------- features
def _hz_to_mel(f):  # Slaney (librosa default, htk=False)
    f = np.asarray(f, dtype=np.float64)
    f_sp = 200.0 / 3
    mels = f / f_sp
    min_log_hz, min_log_mel, logstep = 1000.0, 1000.0 / f_sp, np.log(6.4) / 27.0
    return np.where(f >= min_log_hz, min_log_mel + np.log(np.maximum(f, 1e-10) / min_log_hz) / logstep, mels)


def _mel_to_hz(m):
    m = np.asarray(m, dtype=np.float64)
    f_sp = 200.0 / 3
    freqs = f_sp * m
    min_log_hz, min_log_mel, logstep = 1000.0, 1000.0 / f_sp, np.log(6.4) / 27.0
    return np.where(m >= min_log_mel, min_log_hz * np.exp(logstep * (m - min_log_mel)), freqs)


def mel_filterbank(sr, n_fft, n_mels, fmin, fmax):
    """Same as librosa.filters.mel(sr, n_fft, n_mels, fmin, fmax) (Slaney scale + Slaney norm)."""
    fftfreqs = np.linspace(0, sr / 2, 1 + n_fft // 2)
    mel_f = _mel_to_hz(np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2))
    fdiff = np.diff(mel_f)
    ramps = mel_f[:, None] - fftfreqs[None, :]
    w = np.zeros((n_mels, len(fftfreqs)))
    for i in range(n_mels):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        w[i] = np.maximum(0, np.minimum(lower, upper))
    w *= (2.0 / (mel_f[2: n_mels + 2] - mel_f[:n_mels]))[:, None]
    return w.astype(np.float32)


class FeatureExtractor:
    def __init__(self, cfg):
        self.cfg = cfg
        self.use_librosa = False
        if cfg.input_type == "waveform":
            return
        if cfg.mel_backend in ("auto", "librosa"):
            try:
                import librosa  # noqa: F401
                self.use_librosa = True
            except ImportError:
                if cfg.mel_backend == "librosa":
                    raise
        if not self.use_librosa:
            self.fb = mel_filterbank(cfg.model_fs, cfg.n_fft, cfg.n_mels, cfg.fmin, cfg.fmax)
            win = np.hanning(cfg.win_length + 1)[:-1]          # periodic Hann, like scipy/librosa
            pad = (cfg.n_fft - cfg.win_length) // 2
            self.window = np.pad(win, (pad, cfg.n_fft - cfg.win_length - pad)).astype(np.float32)

    def melspec(self, y):
        c = self.cfg
        if self.use_librosa:
            import librosa
            return librosa.feature.melspectrogram(
                y=y, sr=c.model_fs, n_fft=c.n_fft, hop_length=c.hop_length, win_length=c.win_length,
                center=c.center, pad_mode=c.pad_mode, power=2.0, n_mels=c.n_mels, fmin=c.fmin, fmax=c.fmax)
        if c.center:
            y = np.pad(y, c.n_fft // 2, mode=c.pad_mode)
        n_frames = 1 + (len(y) - c.n_fft) // c.hop_length
        idx = np.arange(c.n_fft)[None, :] + c.hop_length * np.arange(n_frames)[:, None]
        spec = np.abs(np.fft.rfft(y[idx] * self.window, axis=1)) ** 2   # (frames, bins)
        return (self.fb @ spec.T).astype(np.float32)                     # (mels, frames)

    def log(self, S):
        c = self.cfg
        if c.log_scale == "db":
            L = 10.0 * np.log10(np.maximum(S, 1e-10))
            return np.maximum(L, L.max() - c.top_db) if c.top_db else L
        if c.log_scale == "ln":
            return np.log(S + 1e-6)
        return S

    def tensor(self, y):
        """mono window at model_fs -> model input tensor (batch 1)."""
        c = self.cfg
        if c.input_type == "waveform":
            n = int(round(c.win_s * c.model_fs))
            y = y[:n] if len(y) >= n else np.pad(y, (0, n - len(y)))
            if c.wave_norm == "peak":
                y = y / (np.abs(y).max() + 1e-9)
            elif c.wave_norm == "rms":
                y = y * (0.1 / (np.sqrt(np.mean(y ** 2)) + 1e-9))
            return np.ascontiguousarray(y[None].astype(np.float32))          # (1, n)
        X = self.log(self.melspec(y))                         # (mels, frames)
        if c.time_major:
            X = X.T
        if c.norm == "clip":
            X = (X - X.mean()) / (X.std() + 1e-6)
        elif c.norm == "fixed":
            X = (X - c.norm_mean) / (c.norm_std + 1e-12)
        elif c.norm in ("minmax", "imagenet"):
            X = (X - X.min()) / (X.max() - X.min() + 1e-6)
        if c.resize:
            X = _resize(X, int(c.resize[0]), int(c.resize[1]))
        X = np.repeat(X[None], c.in_channels, axis=0)          # (C, H, W)
        if c.norm == "imagenet" and c.in_channels == 3:
            mean = np.array([0.485, 0.456, 0.406], np.float32)[:, None, None]
            std = np.array([0.229, 0.224, 0.225], np.float32)[:, None, None]
            X = (X - mean) / std
        if c.layout == "NHWC":
            X = X.transpose(1, 2, 0)
        return np.ascontiguousarray(X[None].astype(np.float32))


def _resize(a, H, W):
    h, w = a.shape
    xs, ys = np.linspace(0, w - 1, W), np.linspace(0, h - 1, H)
    tmp = np.stack([np.interp(xs, np.arange(w), row) for row in a])               # (h, W)
    return np.stack([np.interp(ys, np.arange(h), tmp[:, j]) for j in range(W)], 1)  # (H, W)


# --------------------------------------------------------------------------- backends
class OnnxModel:
    def __init__(self, path, providers):
        import onnxruntime as ort
        avail = ort.get_available_providers()
        prov = [p for p in providers if p in avail] or ["CPUExecutionProvider"]
        self.sess = ort.InferenceSession(path, providers=prov)
        self.inp = self.sess.get_inputs()[0]
        self.name = f"onnx {prov[0]}"

    def __call__(self, x):
        if "float16" in self.inp.type:
            x = x.astype(np.float16)
        return np.asarray(self.sess.run(None, {self.inp.name: x})[0], dtype=np.float32)


class TrtModel:
    """TensorRT 10 engine (JetPack 6). Must be created and called in the same thread."""

    def __init__(self, path):
        import tensorrt as trt
        import pycuda.driver as cuda
        cuda.init()
        self.cuda, self.trt = cuda, trt
        self.cuctx = cuda.Device(0).make_context()
        logger = trt.Logger(trt.Logger.WARNING)
        with open(path, "rb") as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"cannot load engine {path}: rebuild it on this Jetson (trtexec)")
        self.ctx = self.engine.create_execution_context()
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.inp = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT][0]
        self.out = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT][0]
        self.stream = cuda.Stream()
        self.shape = None
        self.name = "tensorrt"
        self.cuctx.pop()          # don't leave the context bound to this thread: every call pushes it itself

    def _alloc(self, x):
        cuda, trt = self.cuda, self.trt
        if -1 in tuple(self.engine.get_tensor_shape(self.inp)):
            self.ctx.set_input_shape(self.inp, x.shape)
        self.in_dtype = trt.nptype(self.engine.get_tensor_dtype(self.inp))
        out_shape = tuple(self.ctx.get_tensor_shape(self.out))
        self.h_out = np.empty(out_shape, dtype=trt.nptype(self.engine.get_tensor_dtype(self.out)))
        self.d_in = cuda.mem_alloc(int(np.prod(x.shape)) * np.dtype(self.in_dtype).itemsize)
        self.d_out = cuda.mem_alloc(self.h_out.nbytes)
        self.ctx.set_tensor_address(self.inp, int(self.d_in))
        self.ctx.set_tensor_address(self.out, int(self.d_out))
        self.shape = x.shape

    def __call__(self, x):
        self.cuctx.push()         # works from any thread (the live tools run inference in their audio thread)
        try:
            if self.shape != x.shape:
                self._alloc(x)
            x = np.ascontiguousarray(x.astype(self.in_dtype))
            self.cuda.memcpy_htod_async(self.d_in, x, self.stream)
            self.ctx.execute_async_v3(self.stream.handle)
            self.cuda.memcpy_dtoh_async(self.h_out, self.d_out, self.stream)
            self.stream.synchronize()
            return self.h_out.astype(np.float32).copy()
        finally:
            self.cuctx.pop()

    def close(self):
        try:
            self.cuctx.detach()
        except Exception:
            pass


class TorchModel:
    def __init__(self, path):
        import torch
        self.torch = torch
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.m = torch.jit.load(path, map_location=self.dev)
        except Exception:
            self.m = torch.load(path, map_location=self.dev, weights_only=False)
        self.m.eval()
        self.name = f"torch {self.dev}"

    def __call__(self, x):
        with self.torch.no_grad():
            y = self.m(self.torch.from_numpy(x).to(self.dev))
        return y.float().cpu().numpy()


def load_model(cfg):
    b = cfg.backend
    if b == "auto":
        p = cfg.model_path.lower()
        b = "tensorrt" if p.endswith((".engine", ".plan", ".trt")) else "torch" if p.endswith((".pt", ".pth", ".ts")) else "onnx"
    if b == "onnx":
        return OnnxModel(cfg.model_path, cfg.onnx_providers)
    if b == "tensorrt":
        try:
            return TrtModel(cfg.model_path)
        except ImportError as e:                    # pycuda missing: run the same model's ONNX instead
            onnx = os.path.splitext(cfg.model_path)[0] + ".onnx"
            if not os.path.isfile(onnx):
                raise
            print(f"[WARN] TensorRT engine needs pycuda ({e}); using {onnx} with onnxruntime. "
                  f"For the engine: pip3 install pycuda", flush=True)
            return OnnxModel(onnx, cfg.onnx_providers)
    if b == "torch":
        return TorchModel(cfg.model_path)
    raise ValueError(f"unknown backend {b}")


# --------------------------------------------------------------------------- decision
def drone_probability(y, cfg):
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if y.size == 1:                                     # single-logit binary model
        return float(1 / (1 + np.exp(-y[0]))) if cfg.output_activation != "none" else float(y[0])
    if cfg.output_activation == "softmax":
        e = np.exp(y - y.max())
        y = e / e.sum()
    elif cfg.output_activation == "sigmoid":
        y = 1 / (1 + np.exp(-y))
    return float(y[cfg.drone_index])


class Smoother:
    """is_drone = at least N of the last M windows over threshold."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.hist = deque(maxlen=max(1, cfg.smooth_m))

    def update(self, p):
        self.hist.append(p >= self.cfg.threshold)
        return sum(self.hist) >= self.cfg.smooth_n, float(np.mean(self.hist))


class Classifier:
    """Feeds raw frames, returns one result per window."""

    def __init__(self, cfg, model=None):
        self.cfg = cfg
        self.win = Windower(cfg)
        self.feat = FeatureExtractor(cfg)
        self.model = model
        self.smooth = Smoother(cfg)

    def classify_window(self, w):
        t0 = time.perf_counter()
        y = resample(w - w.mean(), self.cfg.fs_in, self.cfg.model_fs)
        out = self.model(self.feat.tensor(y))
        p = drone_probability(out, self.cfg)
        is_drone, ratio = self.smooth.update(p)
        return {"prob": round(float(p), 4), "is_drone": bool(is_drone), "vote_ratio": round(float(ratio), 3),
                "rms_dbfs": round(float(20 * np.log10(np.sqrt(np.mean(w ** 2)) + 1e-12)), 1),
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}

    def push_frames(self, frames):
        return [self.classify_window(w) for w in self.win.push(to_mono(frames, self.cfg))]


def annotate_targets(targets_json, result, max_age_s, now):
    """Adds the classifier result to every target of /fusion/targets (scene-level: one mono mix)."""
    try:
        targets = json.loads(targets_json)
    except ValueError:
        return [], []
    fresh = result is not None and (now - result["t"]) <= max_age_s
    out = []
    for tg in targets if isinstance(targets, list) else []:
        tg = dict(tg)
        tg["drone_prob"] = result["prob"] if fresh else None
        tg["is_drone"] = bool(result["is_drone"]) if fresh else False
        tg["class"] = ("drone" if tg["is_drone"] else "other") if fresh else "unknown"
        out.append(tg)
    return out, [t for t in out if t["is_drone"]]

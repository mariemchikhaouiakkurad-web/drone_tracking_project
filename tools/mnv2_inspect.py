#!/usr/bin/env python3
"""mnv2_inspect.py - print the input/output of the MobileNetV2 model and the yaml lines that match it.

Usage:  python3 tools/mnv2_inspect.py /path/to/model.onnx      (.onnx | .engine/.plan | .pt/.ts)
"""
import sys


def onnx_io(path):
    import onnxruntime as ort
    s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    ins = [(i.name, i.shape, i.type) for i in s.get_inputs()]
    outs = [(o.name, o.shape, o.type) for o in s.get_outputs()]
    return ins, outs


def trt_io(path):
    import tensorrt as trt
    eng = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(open(path, "rb").read())
    if eng is None:
        raise SystemExit("engine does not load here: rebuild it on this Jetson with trtexec --onnx=... --saveEngine=...")
    ins, outs = [], []
    for i in range(eng.num_io_tensors):
        n = eng.get_tensor_name(i)
        row = (n, list(eng.get_tensor_shape(n)), str(eng.get_tensor_dtype(n)))
        (ins if eng.get_tensor_mode(n) == trt.TensorIOMode.INPUT else outs).append(row)
    return ins, outs


def torch_io(path):
    import torch
    m = torch.jit.load(path, map_location="cpu")
    print(m.graph if hasattr(m, "graph") else m)
    return [], []


def suggest(shape):
    dims = [d if isinstance(d, int) and d > 0 else None for d in shape]
    if len(dims) == 2:
        n = dims[1]
        print("\nRaw-waveform model (spectrogram computed inside). Lines for drone_classifier.yaml:")
        print('    input_type: "waveform"')
        print(f"    # {n} samples per window: win_s * model_fs = {n}  (e.g. win_s: 1.0, model_fs: {n})")
        print("Still to confirm from the training code: sample rate, wave_norm, drone_index.")
        return
    if len(dims) != 4:
        print(f"  input has {len(dims)} dims: check the training code.")
        return
    _, a, b, c = dims
    if a in (1, 3):
        layout, ch, H, W = "NCHW", a, b, c
    elif c in (1, 3):
        layout, ch, H, W = "NHWC", c, a, b
    else:
        print("  cannot tell channels from the shape; check the training code.")
        return
    print("\nSuggested lines for uma-8/config/drone_classifier.yaml:")
    print(f'    layout: "{layout}"')
    print(f"    in_channels: {ch}")
    if H and W and (H, W) in ((224, 224), (160, 160), (192, 192), (128, 128), (96, 96)):
        print(f"    resize: [{H}, {W}]            # square image-size input: spectrogram is resized")
        if ch == 3:
            print('    norm: "imagenet"              # if the training used ImageNet mean/std')
    else:
        print(f"    n_mels: {H}                    # if H is the mel axis (time_major: false)")
        print(f"    # W = {W} frames -> frames = 1 + win_s*model_fs/hop_length (center: true)")
        print("    resize: [0]")
    print("Still to confirm from the training code: model_fs, win_s, n_fft, hop_length, fmin, fmax, log, norm.")


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    p = sys.argv[1]
    low = p.lower()
    ins, outs = (trt_io(p) if low.endswith((".engine", ".plan", ".trt"))
                 else torch_io(p) if low.endswith((".pt", ".pth", ".ts")) else onnx_io(p))
    print("INPUTS");  [print(f"  {n}  shape={s}  type={t}") for n, s, t in ins]
    print("OUTPUTS"); [print(f"  {n}  shape={s}  type={t}") for n, s, t in outs]
    if outs:
        last = outs[0][1][-1] if outs[0][1] else None
        print(f"\nOutput classes: {last}  -> set drone_index / class_names; "
              f"{'single logit: binary sigmoid' if last == 1 else 'softmax over classes'}")
    if ins:
        suggest(ins[0][1])


if __name__ == "__main__":
    main()

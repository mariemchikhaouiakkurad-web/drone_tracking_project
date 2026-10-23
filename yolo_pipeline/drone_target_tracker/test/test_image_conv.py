"""Tests of the ROS image -> BGR conversion for the encodings a thermal camera may send (no ROS needed)."""
import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from drone_target_tracker.image_conv import bgr_fields, to_bgr  # noqa: E402

H, W = 4, 6
GREY = (np.arange(H * W, dtype=np.uint8).reshape(H, W) * 10)


def msg(enc, data, step, big=0):
    return SimpleNamespace(encoding=enc, height=H, width=W, step=step, data=bytes(data), is_bigendian=big)


def test_mono8_and_rgb8():
    out = to_bgr(msg("mono8", GREY.tobytes(), W))
    assert out.shape == (H, W, 3) and (out[:, :, 0] == GREY).all()
    rgb = np.dstack([GREY, GREY // 2, GREY // 3])
    out = to_bgr(msg("rgb8", rgb.tobytes(), W * 3))
    assert (out[:, :, 0] == GREY // 3).all() and (out[:, :, 2] == GREY).all()


def test_yuyv_grey_stays_grey():
    yuyv = np.zeros((H, W * 2), np.uint8)
    yuyv[:, 0::2] = GREY          # Y of every pixel
    yuyv[:, 1::2] = 128           # U and V neutral = no colour, as from a thermal sensor
    out = to_bgr(msg("yuv422_yuy2", yuyv.tobytes(), W * 2))
    assert out.shape == (H, W, 3)
    assert np.abs(out[:, :, 1].astype(int) - out[:, :, 0].astype(int)).max() <= 1   # grey: B = G = R
    assert np.abs(out[:, :, 1].astype(int) - GREY.astype(int)).max() <= 30          # same brightness order


def test_mono16_is_stretched_to_8_bits():
    raw = (GREY.astype(np.uint16) * 100 + 1000)
    out = to_bgr(msg("mono16", raw.astype("<u2").tobytes(), W * 2))
    assert out[:, :, 0].min() == 0 and out[:, :, 0].max() == 255


def test_row_padding_and_debug_image_fields():
    padded = np.zeros((H, W + 2), np.uint8)
    padded[:, :W] = GREY
    out = to_bgr(msg("mono8", padded.tobytes(), W + 2))
    assert (out[:, :, 0] == GREY).all()
    f = bgr_fields(out)
    assert f["encoding"] == "bgr8" and f["step"] == W * 3 and len(f["data"]) == H * W * 3

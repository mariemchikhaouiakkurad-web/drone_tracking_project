"""
image_conv.py - sensor_msgs/Image <-> BGR uint8 array (what the trackers take), without cv_bridge, so every
encoding the thermal camera may send is handled the same way on any machine:

  rgb8, bgr8, rgba8, bgra8   colour (a thermal camera usually sends grey in all three channels)
  mono8                      grey, 0..255
  mono16 / 16UC1             raw thermal, 0..65535: scaled to 0..255 per image (min..max of that image)
  yuv422_yuy2 / yuyv         YUYV, the sensor's native format (v4l2_camera publishes it if asked to)
  yuv422 / uyvy              UYVY
These functions only need numpy and OpenCV, no ROS: msg is anything with height, width, step, encoding, data,
is_bigendian.
"""
import cv2
import numpy as np

ENCODINGS = ("rgb8", "bgr8", "rgba8", "bgra8", "mono8", "8uc1", "mono16", "16uc1", "yuv422_yuy2", "yuyv",
             "yuv422", "uyvy")


def to_bgr(msg):
    enc = msg.encoding.lower()
    h, w, step = msg.height, msg.width, msg.step
    try:
        buf = np.frombuffer(msg.data, np.uint8)  # rclpy gives bytes or array.array: no copy
    except TypeError:
        buf = np.asarray(msg.data, np.uint8)
    rows = buf[:h * step].reshape(h, step)
    if enc in ("rgb8", "bgr8"):
        img = rows[:, :w * 3].reshape(h, w, 3)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if enc == "rgb8" else img.copy()
    if enc in ("rgba8", "bgra8"):
        img = rows[:, :w * 4].reshape(h, w, 4)
        return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR if enc == "rgba8" else cv2.COLOR_BGRA2BGR)
    if enc in ("mono8", "8uc1"):
        return cv2.cvtColor(rows[:, :w].copy(), cv2.COLOR_GRAY2BGR)
    if enc in ("mono16", "16uc1"):
        img = rows[:, :w * 2].copy().view(">u2" if msg.is_bigendian else "<u2").reshape(h, w).astype(np.float32)
        lo, hi = float(img.min()), float(img.max())
        img8 = ((img - lo) * (255.0 / (hi - lo)) if hi > lo else np.zeros_like(img)).astype(np.uint8)
        return cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)
    if enc in ("yuv422_yuy2", "yuyv"):
        return cv2.cvtColor(rows[:, :w * 2].reshape(h, w, 2), cv2.COLOR_YUV2BGR_YUYV)
    if enc in ("yuv422", "uyvy"):
        return cv2.cvtColor(rows[:, :w * 2].reshape(h, w, 2), cv2.COLOR_YUV2BGR_UYVY)
    raise ValueError(f"image encoding '{msg.encoding}' is not supported ({', '.join(ENCODINGS)})")


def bgr_fields(img):
    """Fields of a sensor_msgs/Image holding this BGR image (the caller sets the header)."""
    h, w = img.shape[:2]
    return dict(height=h, width=w, encoding="bgr8", is_bigendian=0, step=w * 3,
                data=np.ascontiguousarray(img).tobytes())

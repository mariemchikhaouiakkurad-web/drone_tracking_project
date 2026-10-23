"""sound_parsing.py - reads the sound pipeline's messages (DALBD, branch pipeline_sound) and converts the direction
to the camera frame.

Sound pipeline output (domain 42), see DALBD/TOPICS.md:
    /drone_warning      std_msgs/String JSON  (drone_warning_node, 10 Hz)
        CLEAR: {"level": 0, "level_name": "CLEAR", "stamp": .., "input_ok": true}
        DRONE: {"level": .., "level_name": "DRONE", "stamp": .., "input_ok": true, "id": 37, "prob": 0.91,
                "x": .., "y": .., "z": .., "azimuth_deg": .., "elevation_deg": ..}
    /target/direction   std_msgs/String JSON  (target_direction_node)
        {"stamp": .., "drone": true, "targets": [{"id", "drone_prob", "x", "y", "z", "azimuth_deg",
                                                    "elevation_deg", ...}], "known_source_dropped": {...}}
The older text format "[DRONE] id 3 p 0.91 x .. y .. z .." / "[CLEAR] no drone" is still accepted.

Array frame -> camera frame:
    az_s = atan2(y, x), el_s = atan2(z, hypot(x, y))            (array / ODAS frame, degrees)
    az   = az_sign * az_s + yaw_offset_deg                       (camera frame: right positive)
    el   = el_s + pitch_offset_deg                               (camera frame: up positive)
"""
import json
import math
import re

NUM = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"


def xyz_to_azel_deg(x, y, z):
    return math.degrees(math.atan2(y, x)), math.degrees(math.atan2(z, math.hypot(x, y)))


def array_to_camera(az_s, el_s, az_sign=-1.0, yaw_offset_deg=0.0, pitch_offset_deg=0.0):
    az = az_sign * az_s + yaw_offset_deg
    return (az + 180.0) % 360.0 - 180.0, el_s + pitch_offset_deg


def _num(v):
    try:
        f = float(v)
        return None if f != f else f
    except (TypeError, ValueError):
        return None


def azel_from_dict(d):
    """{x, y, z} (preferred: same convention as the conversion above) or {azimuth_deg, elevation_deg} or {az, el}
    -> (az_deg, el_deg) in the array frame, or None."""
    x, y, z = (_num(d.get(k)) for k in "xyz")
    if x is not None and y is not None and z is not None and (x or y or z):
        return xyz_to_azel_deg(x, y, z)
    for a, e in (("azimuth_deg", "elevation_deg"), ("az", "el"), ("azimuth", "elevation")):
        az, el = _num(d.get(a)), _num(d.get(e))
        if az is not None and el is not None:
            return az, el
    return None


def parse_warning(text):
    """/drone_warning data -> (is_drone, (az_deg, el_deg) array frame or None)."""
    s = text.strip()
    if s.startswith("{"):
        try:
            d = json.loads(s)
        except ValueError:
            return False, None
        if not isinstance(d, dict):
            return False, None
        name = str(d.get("level_name", "")).upper()
        is_drone = name == "DRONE" or (not name and bool(d.get("drone")))
        if d.get("input_ok") is False:
            is_drone = False
        return is_drone, (azel_from_dict(d) if is_drone else None)
    is_drone, xyz = parse_drone_text(s)
    return is_drone, (xyz_to_azel_deg(*xyz) if (is_drone and xyz is not None) else None)


def parse_drone_text(text):
    """Old text format: '[DRONE] id 3 p 0.91 x 0.2 y -0.5 z 0.8' -> (is_drone, (x, y, z) or None)."""
    up = text.strip().upper().lstrip("[ ")
    if up.startswith("CLEAR") or "NO DRONE" in up:
        return False, None
    is_drone = up.startswith("DRONE") or bool(re.search(r"\bDRONE\b\s+(ID|\d)", up))
    m = {k: float(v) for k, v in re.findall(r"\b([xyz])\s*[=:]?\s*(" + NUM + r")", text, re.I)}
    if all(k in m for k in "xyz"):
        return is_drone, (m["x"], m["y"], m["z"])
    nums = [float(v) for v in re.findall(NUM, text)]
    if is_drone and len(nums) >= 3:
        return is_drone, tuple(nums[-3:])
    return is_drone, None


def extract_azel(msg):
    """Any direction message -> (az_deg, el_deg) in the ARRAY frame, or None.
    Handles /target/direction JSON {"drone", "targets": [...]} (best drone_prob target)."""
    for path in ("vector", "point", "pose.position", "position"):
        obj = msg
        for p in path.split("."):
            obj = getattr(obj, p, None)
            if obj is None:
                break
        if obj is not None and all(hasattr(obj, c) for c in "xyz"):
            return xyz_to_azel_deg(obj.x, obj.y, obj.z)
    data = getattr(msg, "data", None)
    if isinstance(data, str):
        try:
            d = json.loads(data)
        except (ValueError, TypeError):
            d = None
        if isinstance(d, dict) and "targets" in d:
            tg = [t for t in (d.get("targets") or []) if isinstance(t, dict)]
            if "drone" in d and not d.get("drone"):
                return None
            tg = [t for t in tg if t.get("drone", True)]
            if not tg:
                return None
            best = max(tg, key=lambda t: _num(t.get("drone_prob")) or 0.0)
            return azel_from_dict(best)
        if isinstance(d, list) and d and isinstance(d[0], dict):
            return azel_from_dict(d[0])
        if isinstance(d, dict):
            if "level_name" in d:
                is_drone, azel = parse_warning(data)
                return azel if is_drone else None
            return azel_from_dict(d)
        is_drone, xyz = parse_drone_text(data)
        if xyz is not None:
            return xyz_to_azel_deg(*xyz)
    return None


# ======================================================================= vertical / any mounting: axis remap
# The UMA-8 (ODAS frame, see uma8.cfg): +x -> MIC1 (12 o'clock), +y -> 9 o'clock (seen from the LED side),
# +z out of the LED face. The camera frame used here is the ROS one: forward, left, up.
# The mounting is described by which ODAS axis (with sign) points FORWARD, LEFT and UP, e.g. for
# "MIC1 up, LED face looking where the camera looks":  "forward=+z,left=-y,up=+x"
# (seen from behind the board, i.e. from the camera, the 9 o'clock side is on the camera's RIGHT -> left = -y).
_AX = {"x": 0, "y": 1, "z": 2}


def parse_axes(spec):
    """'forward=+z,left=-y,up=+x' -> 3x3 rows (forward, left, up) of +-1/0, or None if spec is empty.
    Raises ValueError if the spec is not a proper right-handed rotation."""
    spec = (spec or "").replace(" ", "").lower()
    if not spec:
        return None
    rows = {}
    for part in spec.split(","):
        name, ax = part.split("=")
        sign = -1.0 if ax.startswith("-") else 1.0
        ax = ax.lstrip("+-")
        r = [0.0, 0.0, 0.0]
        r[_AX[ax]] = sign
        rows[name] = r
    M = [rows["forward"], rows["left"], rows["up"]]
    det = (M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1]) - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0])
           + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]))
    if abs(det - 1.0) > 1e-6:
        raise ValueError(f"sound_axes '{spec}' is not a right-handed rotation (det {det:+.0f}): flip one sign")
    return M


def xyz_to_camera(xyz, M):
    """ODAS unit vector -> (az_deg, el_deg) in the camera frame: az = RIGHT positive, el = UP positive."""
    f = sum(M[0][i] * xyz[i] for i in range(3))
    left = sum(M[1][i] * xyz[i] for i in range(3))
    up = sum(M[2][i] * xyz[i] for i in range(3))
    return math.degrees(math.atan2(-left, f)), math.degrees(math.atan2(up, math.hypot(f, left)))


def warning_xyz(text):
    """/drone_warning JSON -> (x, y, z) of the best DRONE target, or None."""
    try:
        d = json.loads(text)
    except (ValueError, TypeError):
        is_drone, xyz = parse_drone_text(text or "")
        return xyz if is_drone else None
    if not isinstance(d, dict) or str(d.get("level_name", "")).upper() != "DRONE" or d.get("input_ok") is False:
        return None
    x, y, z = (_num(d.get(k)) for k in "xyz")
    return (x, y, z) if None not in (x, y, z) else None

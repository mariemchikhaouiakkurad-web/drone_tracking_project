"""Unit tests of every transition of State_machine_drone_pipeline (no ROS needed).

Run:  python3 -m pytest -q test/test_state_machine.py      or      python3 test/test_state_machine.py
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from dalbd_fusion.sound_parsing import array_to_camera, extract_azel, parse_drone_text, parse_warning  # noqa: E402,E501
from dalbd_fusion.state_machine import (SEARCH, SOUND, VALID_NONE, VALID_SOUND, VALID_VISION,  # noqa: E402
                                             VISUAL, StateMachine)

DT = 0.05   # chart sample time


class Sim:
    """Drives the chart at 20 Hz with scripted inputs."""

    def __init__(self, **kw):
        self.sm = StateMachine(**kw)
        self.t = 0.0
        self.vis = None      # (az, el) or None
        self.snd = None      # (az, el) or None
        self.drone = True
        self.out = self.sm.step(self.t)

    def run(self, seconds):
        n = int(round(seconds / DT))
        for _ in range(n):
            self.t += DT
            if self.vis is not None:
                self.sm.on_vision_status(self.t, True)
                self.sm.on_vision_bearing(self.t, *self.vis)
            else:
                self.sm.on_vision_status(self.t, False)
            if self.snd is not None:
                self.sm.on_sound(self.t, *self.snd)
                self.sm.on_drone_label(self.t, self.drone)
            self.out = self.sm.step(self.t)
        return self

    @property
    def state(self):
        return self.sm.state


def test_starts_in_search():
    s = Sim()
    assert s.state == SEARCH and s.out[2] == VALID_NONE and math.isnan(s.out[0])


def test_sound_alone_at_start_is_ignored():
    s = Sim()
    s.snd = (10.0, 5.0)
    s.run(3.0)
    assert s.state == SEARCH and s.sm.target_lost == 0


def test_search_to_visual():
    s = Sim()
    s.vis = (3.0, 1.0)
    s.run(0.1)
    assert s.state == VISUAL and s.out == (3.0, 1.0, VALID_VISION) and s.sm.target_lost == 0


def test_visual_ignores_sound():
    s = Sim()
    s.vis = (3.0, 1.0)
    s.snd = (40.0, 10.0)
    s.run(1.0)
    assert s.state == VISUAL and s.out[0] == 3.0


def test_short_vision_dropout_is_debounced():
    s = Sim()
    s.vis = (3.0, 1.0)
    s.run(0.5)
    s.vis = None
    s.run(0.4)                       # tracker says "searching" for 0.4 s < lost_s (0.5 s)
    assert s.state == VISUAL and s.out[2] == VALID_VISION and s.out[0] == 3.0


def test_visual_lost_to_search_sets_target_lost():
    s = Sim()
    s.vis = (3.0, 1.0)
    s.run(0.5)
    s.vis = None
    s.run(1.0)
    assert s.state == SEARCH and s.sm.target_lost == 1 and s.out[2] == VALID_NONE


def test_lost_then_sound_goes_to_sound_track():
    s = Sim()
    s.vis = (3.0, 1.0)
    s.run(0.5)
    s.vis = None
    s.snd = (30.0, 5.0)
    s.run(1.2)
    assert s.state == SOUND and s.out == (30.0, 5.0, VALID_SOUND)


def test_sound_track_to_visual_when_vision_back():
    s = Sim()
    s.vis = (3.0, 1.0)
    s.run(0.5)
    s.vis = None
    s.snd = (30.0, 5.0)
    s.run(1.5)
    s.vis = (28.0, 4.0)
    s.run(0.1)
    assert s.state == VISUAL and s.out == (28.0, 4.0, VALID_VISION) and s.sm.target_lost == 0
    assert not any(e[1] == SOUND and e[2] == SEARCH for e in s.sm.events), "must go SOUND -> VISUAL directly"


def test_sound_short_gap_holds_direction():
    s = Sim()
    s.vis = (3.0, 1.0)
    s.run(0.5)
    s.vis = None
    s.snd = (30.0, 5.0)
    s.run(1.5)
    s.snd = None
    s.run(3.0)                       # sound == 0 for ~1 s (after its 2 s timeout) < search_timeout
    assert s.state == SOUND and s.out == (30.0, 5.0, VALID_SOUND)


def test_sound_timeout_to_search_resets_target_lost():
    s = Sim()
    s.vis = (3.0, 1.0)
    s.run(0.5)
    s.vis = None
    s.snd = (30.0, 5.0)
    s.run(1.5)
    s.snd = None
    s.run(2.0 + 5.0 + 0.2)           # 2 s until sound == 0, then 5 s
    assert s.state == SEARCH and s.sm.target_lost == 0
    s.snd = (30.0, 5.0)              # sound again, but target_lost == 0 -> ignored
    s.run(1.0)
    assert s.state == SEARCH


def test_sound_without_drone_label_is_not_sound():
    s = Sim()
    s.vis = (3.0, 1.0)
    s.run(0.5)
    s.vis = None
    s.snd = (30.0, 5.0)
    s.drone = False
    s.run(2.0)
    assert s.state == SEARCH


def test_option_sound_can_start_tracking():
    s = Sim(sound_requires_loss=False)
    s.snd = (10.0, 5.0)
    s.run(0.2)
    assert s.state == SOUND and s.out[2] == VALID_SOUND


def test_vision_has_priority_in_search():
    s = Sim(sound_requires_loss=False)
    s.vis = (1.0, 0.0)
    s.snd = (50.0, 0.0)
    s.run(0.1)
    assert s.state == VISUAL


def test_full_scenario_sequence():
    s = Sim()
    s.vis = (0.0, 0.0); s.run(5)            # in view
    s.vis = None; s.snd = (-30.0, 2.0); s.run(5)   # walks out left, phone on
    s.vis = (-20.0, 2.0); s.snd = None; s.run(5)   # back in view, phone off
    s.vis = None; s.run(10)                 # gone, silent
    seq = [e[2] for e in s.sm.events]
    assert seq == [SEARCH, VISUAL, SEARCH, SOUND, VISUAL, SEARCH], seq
    # leaving with the phone still ON: the last sound keeps it in SOUND_TRACK, then times out to SEARCH
    s2 = Sim()
    s2.vis = (0.0, 0.0); s2.snd = (0.0, 0.0); s2.run(3)
    s2.vis = None; s2.snd = None; s2.run(10)
    seq2 = [e[2] for e in s2.sm.events]
    assert seq2 == [SEARCH, VISUAL, SEARCH, SOUND, SEARCH], seq2


def test_parse_and_frame():
    d, xyz = parse_drone_text("[DRONE] id 3 p 0.91 x 0.50 y -0.50 z 0.70")
    assert d and xyz == (0.5, -0.5, 0.7)
    d, xyz = parse_drone_text("[CLEAR] no drone")
    assert not d and xyz is None

    class M:
        pass
    m = M(); m.vector = M(); m.vector.x, m.vector.y, m.vector.z = 1.0, 1.0, 0.0
    az, el = extract_azel(m)
    assert abs(az - 45.0) < 1e-9 and abs(el) < 1e-9
    j = M(); j.data = json.dumps({"az": 12.5, "el": 30.0})
    assert extract_azel(j) == (12.5, 30.0)
    assert array_to_camera(45.0, 10.0) == (-45.0, 10.0)
    az, el = array_to_camera(45.0, 10.0, az_sign=-1, yaw_offset_deg=3.5, pitch_offset_deg=-1.0)
    assert abs(az + 41.5) < 1e-9 and el == 9.0



def test_parse_real_dalbd_json():
    d, azel = parse_warning('{"level": 0, "level_name": "CLEAR", "stamp": 1792431641.8, "input_ok": true}')
    assert not d and azel is None
    msg = ('{"level": 1, "level_name": "DRONE", "stamp": 1.0, "input_ok": true, "id": 37, "prob": 0.91, '
           '"x": 0.7071, "y": 0.7071, "z": 0.0, "azimuth_deg": 45.0, "elevation_deg": 0.0}')
    d, azel = parse_warning(msg)
    assert d and abs(azel[0] - 45.0) < 0.01 and abs(azel[1]) < 0.01
    d, azel = parse_warning('{"level": 1, "level_name": "DRONE", "input_ok": true, "x": null, "y": null, "z": null, '
                            '"azimuth_deg": -30.0, "elevation_deg": 12.0}')
    assert d and azel == (-30.0, 12.0)
    d, azel = parse_warning('{"level": 1, "level_name": "DRONE", "input_ok": false, "x": 1, "y": 0, "z": 0}')
    assert not d

    class M:
        pass
    m = M()
    m.data = json.dumps({"stamp": 1.0, "drone": True, "targets": [
        {"id": 3, "drone_prob": 0.6, "x": 1.0, "y": 0.0, "z": 0.0, "azimuth_deg": 0.0, "elevation_deg": 0.0},
        {"id": 4, "drone_prob": 0.9, "x": 0.0, "y": 1.0, "z": 0.0, "azimuth_deg": 90.0, "elevation_deg": 0.0}],
        "known_source_dropped": {}})
    az, el = extract_azel(m)
    assert abs(az - 90.0) < 1e-6, "best drone_prob target"
    m.data = json.dumps({"stamp": 1.0, "drone": False, "targets": []})
    assert extract_azel(m) is None



def test_vision_without_status_messages():
    """Tracker that publishes bearings only (status only on change, missed at start-up) -> still vision == 1."""
    sm = StateMachine()
    t = 0.0
    sm.step(t)
    for _ in range(20):                 # bearings at ~3.3 Hz (YOLO rate), no status at all
        t += 0.3
        sm.on_vision_bearing(t, 5.0, 1.0)
        sm.step(t)
    assert sm.state == VISUAL and sm.vision(t) == 1
    for _ in range(30):                 # bearings stop: 0.6 s timeout + 0.5 s lost_s
        t += 0.05
        sm.step(t)
    assert sm.state == SEARCH and sm.target_lost == 1


def test_status_searching_overrides_old_bearing():
    sm = StateMachine()
    sm.step(0.0)
    sm.on_vision_bearing(0.1, 5.0, 1.0)
    sm.on_vision_status(0.2, False)     # "searching" after the last bearing
    assert sm.vision(0.25) == 0



def test_option_ignore_status_flicker():
    sm = StateMachine(use_vision_status=False)
    sm.step(0.0)
    sm.on_vision_bearing(0.1, 5.0, 1.0)
    sm.on_vision_status(0.2, False)     # flicker "searching" right after a bearing
    assert sm.vision(0.25) == 1


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  ok    {name}")
            except AssertionError as e:
                fails += 1
                print(f"  FAIL  {name}  {e}")
    print("\nALL PASSED" if not fails else f"\n{fails} FAILED")
    sys.exit(1 if fails else 0)


# ---------------------------------------------------------------- axis remap (vertical UMA-8)
from dalbd_fusion.sound_parsing import parse_axes, warning_xyz, xyz_to_camera  # noqa: E402

AX = "forward=+z,left=-y,up=+x"


def test_axes_forward_is_centre():
    az, el = xyz_to_camera((0, 0, 1), parse_axes(AX))
    assert abs(az) < 1e-6 and abs(el) < 1e-6


def test_axes_mic1_is_up_and_minus_y_is_left():
    M = parse_axes(AX)
    assert abs(xyz_to_camera((1, 0, 0), M)[1] - 90) < 1e-6          # MIC1 side = UP
    assert abs(xyz_to_camera((0, -1, 0), M)[0] + 90) < 1e-6         # -y = LEFT (az negative)


def test_axes_generator_reading_is_left_and_down():
    az, el = xyz_to_camera((-0.45, -0.32, 0.83), parse_axes(AX))
    assert -25 < az < -17 and -31 < el < -23                          # ~21 deg LEFT, ~27 deg DOWN


def test_axes_reject_mirror_and_empty():
    assert parse_axes("") is None
    try:
        parse_axes("forward=+z,left=+y,up=+x")
        assert False, "mirror spec must be rejected"
    except ValueError:
        pass


def test_warning_xyz():
    assert warning_xyz('{"level_name":"CLEAR","input_ok":true}') is None
    assert warning_xyz('{"level_name":"DRONE","input_ok":true,"x":0.1,"y":0.2,"z":0.9}') == (0.1, 0.2, 0.9)

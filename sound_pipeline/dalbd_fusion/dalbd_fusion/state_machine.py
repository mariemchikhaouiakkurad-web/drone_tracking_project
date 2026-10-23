"""state_machine.py - DALBD fusion state machine (pure Python, no ROS).

Implements State_machine_drone_pipeline (corrected chart):

    default ──{lost_s = 0.5; sound_timeout = 2; search_timeout = 5; target_lost = 0}──> SEARCH

    SEARCH        entry: valid = 0                     during: publish_direction(NaN, NaN, 0)
    VISUAL_TRACK  entry: target_lost = 0; valid = 1    during: az, el = vision; publish (sound ignored)
    SOUND_TRACK   entry: valid = 2                     during: az, el = sound; publish; vision searches there

    SEARCH       ─1─ [vision == 1]                                        ─> VISUAL_TRACK
    SEARCH       ─2─ [sound == 1 && target_lost == 1]                     ─> SOUND_TRACK
    VISUAL_TRACK ─1─ [duration(vision == 0) > lost_s] {target_lost = 1}   ─> SEARCH
    SOUND_TRACK  ─1─ [vision == 1]                                        ─> VISUAL_TRACK
    SOUND_TRACK  ─2─ [duration(sound == 0) > search_timeout] {target_lost = 0} ─> SEARCH

Inputs (computed from the pipelines' messages):
    vision = 1  the last tracker bearing is younger than vision_timeout_s, unless a newer /target/status says the
                tracker is no longer tracking (the status is optional: some trackers publish it only on change)
    sound  = 1  a sound direction younger than sound_timeout_s AND (if require_drone_label) a [DRONE] label
                younger than sound_timeout_s

Stateflow semantics kept: one transition per step at most, checked in priority order; when a transition fires,
exit -> transition action -> entry run (not the during action); duration() restarts when the state is entered.
"""
import math

SEARCH, VISUAL, SOUND = "SEARCH", "VISUAL_TRACK", "SOUND_TRACK"
VALID_NONE, VALID_VISION, VALID_SOUND = 0, 1, 2
VALID_NAME = {VALID_NONE: "none", VALID_VISION: "vision", VALID_SOUND: "sound"}
NAN = float("nan")


class StateMachine:
    def __init__(self, lost_s=0.5, sound_timeout_s=2.0, search_timeout_s=5.0, vision_timeout_s=0.6,
                 require_drone_label=True, sound_requires_loss=True, use_vision_status=True):
        # ---- default transition {lost_s = 0.5; sound_timeout = 2; search_timeout = 5; target_lost = 0;}
        self.lost_s = float(lost_s)
        self.sound_timeout_s = float(sound_timeout_s)
        self.search_timeout_s = float(search_timeout_s)
        self.vision_timeout_s = float(vision_timeout_s)
        self.require_drone_label = bool(require_drone_label)
        self.sound_requires_loss = bool(sound_requires_loss)   # False: drop "&& target_lost == 1"
        self.use_vision_status = bool(use_vision_status)       # False: vision from bearing freshness only
        self.target_lost = 0

        # ---- raw inputs (latest values + arrival time)
        self.vis_tracking = False
        self.status_t = -1e18      # time of the last /target/status
        self.vis_t = -1e18
        self.vis_az = self.vis_el = NAN
        self.snd_t = -1e18
        self.snd_az = self.snd_el = NAN
        self.drone_t = -1e18

        # ---- outputs
        self.az = self.el = NAN
        self.valid = VALID_NONE

        # ---- bookkeeping for duration()
        self.state = None
        self.state_t = 0.0
        self._v0_since = None      # time vision became 0
        self._s0_since = None      # time sound became 0
        self.events = []           # (t, from, to, reason)
        self._started = False

    # ================================================================== inputs
    def on_vision_status(self, t, tracking):
        self.vis_tracking = bool(tracking)
        self.status_t = t

    def on_vision_bearing(self, t, az_deg, el_deg):
        self.vis_t, self.vis_az, self.vis_el = t, float(az_deg), float(el_deg)

    def on_sound(self, t, az_deg, el_deg):
        self.snd_t, self.snd_az, self.snd_el = t, wrap_deg(float(az_deg)), float(el_deg)

    def on_drone_label(self, t, is_drone):
        if is_drone:
            self.drone_t = t

    # ================================================================== chart inputs
    def vision(self, t):
        if t - self.vis_t > self.vision_timeout_s:
            return 0                                     # no recent bearing
        if self.use_vision_status and not self.vis_tracking and self.status_t >= self.vis_t:
            return 0                                     # tracker said "searching" after its last bearing
        return 1

    def sound(self, t):
        if t - self.snd_t > self.sound_timeout_s:
            return 0
        if self.require_drone_label and t - self.drone_t > self.sound_timeout_s:
            return 0
        return 1

    def duration(self, since, t):
        """Stateflow duration(): time the condition has been true, restarted when the state was entered."""
        if since is None:
            return 0.0
        return t - max(since, self.state_t)

    # ================================================================== state actions
    def _entry(self, state, t):
        self.state, self.state_t = state, t
        if state == SEARCH:
            self.valid = VALID_NONE
            self.az = self.el = NAN
        elif state == VISUAL:
            self.target_lost = 0
            self.valid = VALID_VISION
            self.az, self.el = self.vis_az, self.vis_el
        elif state == SOUND:
            self.valid = VALID_SOUND
            self.az, self.el = self.snd_az, self.snd_el

    def _during(self, t, v, s):
        if self.state == SEARCH:
            self.az = self.el = NAN
        elif self.state == VISUAL:
            if v:                      # while vision == 0 (< lost_s) the last vision bearing is held
                self.az, self.el = self.vis_az, self.vis_el
        elif self.state == SOUND:
            if s:                      # sound briefly missing: hold the last sound direction
                self.az, self.el = self.snd_az, self.snd_el

    def _go(self, t, new, reason, action=None):
        old = self.state
        # exit actions: none in this chart
        if action:
            action()                   # transition action {...}
        self._entry(new, t)
        self.events.append((t, old, new, reason))

    # ================================================================== one step (fixed sample time)
    def step(self, t):
        """Run one chart step at time t (seconds). Returns (az_deg, el_deg, valid)."""
        if not self._started:
            self._started = True
            self._entry(SEARCH, t)
            self.events.append((t, None, SEARCH, "default transition"))

        v, s = self.vision(t), self.sound(t)
        if v:
            self._v0_since = None
        elif self._v0_since is None:
            self._v0_since = t
        if s:
            self._s0_since = None
        elif self._s0_since is None:
            self._s0_since = t

        fired = False
        if self.state == SEARCH:
            if v == 1:
                self._go(t, VISUAL, "[vision == 1]")
                fired = True
            elif s == 1 and (self.target_lost == 1 or not self.sound_requires_loss):
                self._go(t, SOUND, "[sound == 1 && target_lost == 1]")
                fired = True
        elif self.state == VISUAL:
            if v == 0 and self.duration(self._v0_since, t) > self.lost_s:
                self._go(t, SEARCH, "[duration(vision == 0) > lost_s] {target_lost = 1}",
                         action=lambda: setattr(self, "target_lost", 1))
                fired = True
        elif self.state == SOUND:
            if v == 1:
                self._go(t, VISUAL, "[vision == 1] drone found again")
                fired = True
            elif s == 0 and self.duration(self._s0_since, t) > self.search_timeout_s:
                self._go(t, SEARCH, "[duration(sound == 0) > search_timeout] {target_lost = 0}",
                         action=lambda: setattr(self, "target_lost", 0))
                fired = True

        if not fired:
            self._during(t, v, s)
        return self.az, self.el, self.valid

    # ================================================================== helpers
    def snapshot(self, t):
        return {
            "state": self.state,
            "valid": self.valid,
            "source": VALID_NAME[self.valid],
            "az_deg": None if math.isnan(self.az) else round(self.az, 3),
            "el_deg": None if math.isnan(self.el) else round(self.el, 3),
            "vision": self.vision(t),
            "sound": self.sound(t),
            "target_lost": self.target_lost,
            "time_in_state_s": round(t - self.state_t, 2),
        }


def wrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0

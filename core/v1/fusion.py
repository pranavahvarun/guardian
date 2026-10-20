"""
core/fusion.py -- turns the three module outputs into one emergency score
(Stage 7.1). Design: core/FUSION_DESIGN.md and the doc "Guardian Fusion: How
the Emergency Score Works".

Every window (about every 1.24 s) fusion scores five emergency types:

    S_type = clip[0,1]( T * sum(w_i * q_i * e_i) - sum(v_j * q_j * c_j) )
    emergency score = max over types; the alert names the leading type

    T   gate, 0 or 1: has this type been triggered at all
    e   evidence, 0 to 1          w  its weight
    c   counter-evidence, 0 to 1  v  its weight
    q   quality of the module the value comes from

Types and STARTING weights (WEIGHTS below; tuned on the 7 scenarios, then
frozen before the held-out test set runs):

    hard fall     gate always            impact x1.0
                  counter: moved again x0.7, walking x0.2 (walking only counts
                  once the whole 10 s window is after the impact; Stage 7.3)
    long lie      gate impact >= 0.5     min(1, still_s/60) x1.0
                  counter: moved again x1.0
    faint         gate impact >= 0.3, or HR below 60 for 10 s
                  min(1, still_s/30) x0.5, min(1, s HR low/30) x0.4, impact x0.3
                  counter: moved again x1.0
    heart rate    gate HR outside 60-100 now
                  min(1, s out of range/60) x1.0
                  counter: walking x1.0 (high HR only)
    unresponsive  gate daytime           min(1, still_s/1800) x0.4
                  counter: night x1.0
                  (Stage 7.3: weight 1.0 -> 0.4, so 30 min still in the daytime
                  gives a check-in, as designed, never an alert by itself.
                  With 1.0 it was a check-in at 12 min and an alert at 21 min.
                  An unanswered check-in goes urgent in core/alerts.py.)

"Moved again" (Stage 7.3c). The motion module says recovered after 8 s of
movement since the impact, and takes it back after 10 s still. Fusion only
believes it when:
    - it belongs to the current impact (seconds moved <= seconds since the
      impact; a new impact starts the count again), and
    - it has held for CONFIRM_S = 12 s. Someone who struggles for 8 s and then
      lies still loses "recovered" within 10 s plus one window (about 11.2 s),
      so 12 s cannot be fooled by that pattern. Found in hard_fall_long_lie:
      movement right after the fall cleared the alert, and the alert only came
      back once the person had been still for 10 s.

Inputs, as stored by core/pipeline.py (one record per stream per window):
    motion         score = impact; extras still_s, recovered
    activity       extras state; used only while motion says the person is
                   moving (majority of the last 10 s), and smoothed the same way
    physiological  extras hr (estimated bpm, not the band); only windows with
                   quality >= 0.5 count. An untrusted window neither adds to
                   nor resets the low/high counters.
    context        clock time; night is 22:00 to 07:00. Scenarios without a
                   clock are daytime.

60 and 100 bpm are PLACEHOLDERS until Mithuna confirms thresholds with
sources. None of this is a validated risk measure; it is a rule set tuned on
simulated scenarios.
"""

from __future__ import annotations

from collections import deque

# ---- starting values (tuned in 7.3, then frozen) ---------------------------
WEIGHTS = {
    "hard_fall": {"impact": 1.0, "moved": 0.7, "walking": 0.2},
    "long_lie": {"gate_impact": 0.5, "still": 1.0, "still_full_s": 60.0, "moved": 1.0},
    "faint": {"gate_impact": 0.3, "gate_low_s": 10.0, "still": 0.5, "still_full_s": 30.0,
              "low": 0.4, "low_full_s": 30.0, "impact": 0.3, "moved": 1.0},
    "heart_rate": {"out": 1.0, "out_full_s": 60.0, "walking": 1.0},
    "unresponsive": {"still": 0.4, "still_full_s": 1800.0, "night": 1.0},
}
HR_LOW, HR_HIGH = 60.0, 100.0        # bpm, placeholders (Mithuna, with sources)
HR_MIN_QUALITY = 0.5
SMOOTH_S = 10.0                      # majority window for moving / activity
CONFIRM_S = 12.0                     # "moved again" must hold this long (Stage 7.3c)
NIGHT = (22 * 60, 7 * 60)            # minutes after midnight: 22:00 to 07:00
WALKING = "LOCOMOTION"
TYPES = tuple(WEIGHTS)


def _c01(x):
    return max(0.0, min(1.0, float(x)))


def _minutes(clock):
    if not clock:
        return None
    h, m = clock.split(":")
    return int(h) * 60 + int(m)


class Fusion:
    """Call step(records) once per window with that window's records."""

    def __init__(self, clock: str | None = None, weights: dict | None = None):
        self.w = weights or WEIGHTS
        self.start_min = _minutes(clock)
        self.reset()

    def reset(self):
        self.t_prev = None
        self.hist = deque()              # (t, moving, activity state)
        self.low_s = self.high_s = 0.0
        self.hr = None
        self.tsi_prev = None
        self.rec_since = None            # when "moved again" started to hold

    # -- helpers ------------------------------------------------------------
    def _night(self, t):
        if self.start_min is None:
            return False
        m = (self.start_min + t / 60.0) % 1440
        a, b = NIGHT
        return m >= a or m < b

    def _walking(self, t, moving, act):
        self.hist.append((t, moving, act))
        while self.hist and self.hist[0][0] < t - SMOOTH_S:
            self.hist.popleft()
        n = len(self.hist)
        moving_maj = sum(h[1] for h in self.hist) > n / 2
        walk_maj = sum(h[2] == WALKING for h in self.hist) > n / 2
        return moving_maj and walk_maj, moving_maj

    def _moved(self, t, rec, moved_s, tsi):
        """1.0 once "moved again" is confirmed for the current impact."""
        if tsi is None or (self.tsi_prev is not None and tsi < self.tsi_prev - 0.5):
            self.rec_since = None                         # no impact, or a new one
        self.tsi_prev = tsi
        if rec is True and tsi is not None and moved_s <= tsi + 1.3:
            if self.rec_since is None:
                self.rec_since = t
        else:
            self.rec_since = None
        return 1.0 if self.rec_since is not None and t - self.rec_since >= CONFIRM_S else 0.0

    def _heart(self, dt, hr, q):
        if hr is None or q < HR_MIN_QUALITY:
            return False                                  # hold counters
        self.hr = hr
        self.low_s = self.low_s + dt if hr < HR_LOW else 0.0
        self.high_s = self.high_s + dt if hr > HR_HIGH else 0.0
        return True

    # -- one window ---------------------------------------------------------
    def step(self, records) -> dict:
        r = {x["stream"]: x for x in records}
        t = float(next(iter(r.values()))["t"])
        dt = 0.0 if self.t_prev is None else max(0.0, t - self.t_prev)
        self.t_prev = t

        m = r.get("motion", {})
        mx = m.get("extras", {}) or {}
        qm = float(m.get("quality", 0.0))
        impact = float(m.get("score", 0.0))
        still_s = float(mx.get("still_s") or 0.0)
        moving = still_s == 0.0
        tsi = mx.get("time_since_impact")
        moved_s = float(mx.get("moved_after_impact_s") or 0.0)
        moved = self._moved(t, mx.get("recovered"), moved_s,
                            None if tsi is None else float(tsi))

        a = r.get("activity", {})
        qa = float(a.get("quality", 0.0))
        walking, moving_maj = self._walking(t, moving, (a.get("extras") or {}).get("state"))
        walk = 1.0 if walking else 0.0
        walk_after = 1.0 if walking and tsi is not None and tsi >= SMOOTH_S else 0.0

        p = r.get("physiological", {})
        qp = float(p.get("quality", 0.0))
        trusted = self._heart(dt, (p.get("extras") or {}).get("hr"), qp)
        night = self._night(t)

        W, S = self.w, {}
        w = W["hard_fall"]
        S["hard_fall"] = _c01(w["impact"] * qm * impact
                              - w["moved"] * qm * moved - w["walking"] * qa * walk_after)
        w = W["long_lie"]
        gate = impact >= w["gate_impact"]
        S["long_lie"] = _c01(gate * w["still"] * qm * min(1.0, still_s / w["still_full_s"])
                             - w["moved"] * qm * moved)
        w = W["faint"]
        gate = impact >= w["gate_impact"] or self.low_s >= w["gate_low_s"]
        ev = (w["still"] * qm * min(1.0, still_s / w["still_full_s"])
              + w["low"] * qp * min(1.0, self.low_s / w["low_full_s"])
              + w["impact"] * qm * impact)
        S["faint"] = _c01(gate * ev - w["moved"] * qm * moved)
        w = W["heart_rate"]
        out_s = max(self.low_s, self.high_s)
        gate = out_s > 0
        S["heart_rate"] = _c01(gate * w["out"] * qp * min(1.0, out_s / w["out_full_s"])
                               - (w["walking"] * qa * walk if self.high_s > 0 else 0.0))
        w = W["unresponsive"]
        S["unresponsive"] = _c01((not night) * w["still"] * qm * min(1.0, still_s / w["still_full_s"])
                                 - w["night"] * night)

        lead = max(TYPES, key=lambda k: S[k])
        return dict(t=t, score=S[lead], type=lead, scores=S, evidence=dict(
            impact=round(impact, 3), still_s=round(still_s, 1), moved_again=bool(moved),
            moved_s=round(moved_s, 1), recovered_raw=mx.get("recovered"),
            since_impact=None if tsi is None else round(float(tsi), 1),
            activity=(a.get("extras") or {}).get("state"),
            moving=moving_maj, walking=walking, hr=self.hr, hr_trusted=trusted,
            hr_low_s=round(self.low_s, 1), hr_high_s=round(self.high_s, 1), night=night,
            q_motion=round(qm, 2), q_activity=round(qa, 2), q_physio=round(qp, 2)))


def fuse(records, fusion: Fusion) -> dict:
    """Kept for the old placeholder signature."""
    return fusion.step(records)

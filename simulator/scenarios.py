"""
simulator/scenarios.py -- SIMULATED timelines for demos and integration tests
(Stage 5.2).

How a scenario is made:
- Motion is REAL wrist data: WEDA-FALL recordings from the test people
  (U04 U07 U08 U10 young, U21 U22 U29 elderly), none of whom the motion model
  was trained on. Recordings are joined with a 0.5 s cross-fade.
- "Lying still" stretches after a fall are SYNTHETIC: the last wrist
  orientation held with small noise. WEDA recordings end a few seconds
  after the fall, so a long lie has to be made up.
- Heart rate and skin temperature are SYNTHETIC traces with illustrative
  values. They are not clinical thresholds and not measured data.
- PPG (Stage 6.4) is SYNTHETIC too: a textbook pulse shape driven by the
  heart-rate trace, at 100 Hz, with artefact wherever the wrist moves
  (simulator/ppg_synth.py). It lets the physiological stream run; it says
  nothing about how well that stream works on real PPG.
- Lead-in (Stage 7.3): every scenario starts with 30 s of SYNTHETIC resting
  (band already worn), so the physiological stream has finished its 30 s
  warm-up before anything happens. A real band is worn all day; without
  this, events in the first 30 s were invisible to the heart-rate stream.

Every scenario carries simulated=True and a ground-truth timeline. The truth
never goes to the streams; it is only used to draw and check results.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data import loader_weda as weda  # noqa: E402
from data.loader import HOP_N, TARGET_FS, WINDOW_N  # noqa: E402
from simulator import ppg_synth  # noqa: E402

FS = TARGET_FS
FADE = int(0.5 * FS)
IMPACT_G = 2.25
LEAD_IN_S = 30.0                        # Stage 7.3: physiological warm-up before events


@dataclass
class Scenario:
    name: str
    title: str
    description: str
    acc: np.ndarray                     # (N, 3) g
    gyro: np.ndarray                    # (N, 3) deg/s
    hr: np.ndarray                      # (N,) bpm, synthetic
    temp: np.ndarray                    # (N,) deg C, synthetic
    truth: list = field(default_factory=list)    # segments
    events: list = field(default_factory=list)   # {"t", "what"}
    simulated: bool = True
    ppg: np.ndarray | None = None       # (M,) synthetic, at ppg_fs
    ppg_fs: int = ppg_synth.PPG_FS
    clock: str | None = None            # time of day at the start, "HH:MM" (for fusion)

    @property
    def seconds(self) -> float:
        return len(self.acc) / FS

    def windows(self):
        """contract.py windows, 2.5 s every 1.24 s. HR/temp = value at window end.
        ppg = the same 2.5 s of synthetic PPG at ppg_fs."""
        pw = int(round(WINDOW_N / FS * self.ppg_fs))
        for start in range(0, len(self.acc) - WINDOW_N + 1, HOP_N):
            stop = start + WINDOW_N
            p0 = int(round(start / FS * self.ppg_fs))
            ppg = None
            if self.ppg is not None and p0 + pw <= len(self.ppg):
                ppg = self.ppg[p0:p0 + pw]
            yield {"t": start / FS, "acc": self.acc[start:stop],
                   "gyro": self.gyro[start:stop],
                   "hr": float(self.hr[stop - 1]), "temp": float(self.temp[stop - 1]),
                   "ppg": ppg, "ppg_fs": self.ppg_fs if ppg is not None else None,
                   "fs": FS}


class Library:
    """WEDA-FALL recordings by person and activity code."""

    def __init__(self, root=weda.DEFAULT_ROOT):
        self.root = Path(root)

    def get(self, subject, code, rep=1):
        recs = sorted(weda.scan(self.root, subjects=[subject], codes=[code]),
                      key=lambda r: r.path.name)
        if len(recs) < rep:
            raise LookupError(f"WEDA-FALL {code} for {subject} (repetition {rep}) not found "
                              f"under {self.root}")
        return recs[rep - 1]


class Builder:
    def __init__(self, lib: Library, seed=0, lead_in=LEAD_IN_S):
        self.lib, self.rng = lib, np.random.default_rng(seed)
        self.lead_in = lead_in
        self.acc = np.zeros((0, 3))
        self.gyro = np.zeros((0, 3))
        self.truth, self.events = [], []

    @property
    def now(self):
        return len(self.acc) / FS

    def _append(self, a, g, label, kind):
        a, g = np.asarray(a, float), np.asarray(g, float)
        if len(self.acc) >= FADE and len(a) > FADE:
            w = np.linspace(0, 1, FADE)[:, None]
            self.acc[-FADE:] = (1 - w) * self.acc[-FADE:] + w * a[:FADE]
            self.gyro[-FADE:] = (1 - w) * self.gyro[-FADE:] + w * g[:FADE]
            a, g = a[FADE:], g[FADE:]
        start = self.now
        self.acc = np.vstack([self.acc, a])
        self.gyro = np.vstack([self.gyro, g])
        self.truth.append(dict(start=round(start, 2), end=round(self.now, 2),
                               label=label, kind=kind))
        return start

    def _rest(self, seconds, o, label, kind="synthetic"):
        """Wrist held at orientation o with small sensor noise (synthetic)."""
        n = int(seconds * FS)
        o = np.asarray(o, float)
        o = o / max(np.linalg.norm(o), 1e-6)
        start = self.now
        self.acc = np.vstack([self.acc, o + self.rng.normal(0, 0.008, (n, 3))])
        self.gyro = np.vstack([self.gyro, self.rng.normal(0, 0.8, (n, 3))])
        self.truth.append(dict(start=round(start, 2), end=round(self.now, 2),
                               label=label, kind=kind))

    def _lead(self, o):
        if len(self.acc) == 0 and self.lead_in > 0:
            self._rest(self.lead_in, o, "lead-in: resting, band already worn (synthetic)")

    def record(self, subject, code, rep=1):
        rec = self.lib.get(subject, code, rep)
        self._lead(rec.acc[:FADE].mean(axis=0))
        g = rec.gyro if rec.gyro is not None else np.zeros_like(rec.acc)
        name, _state, is_fall, *_ = weda.ACTIVITIES[code]
        start = self._append(rec.acc, g, f"{code} {name} ({subject})",
                             "fall" if is_fall else "daily activity")
        if is_fall:
            svm = np.linalg.norm(rec.acc, axis=1)
            hit = np.flatnonzero(svm > IMPACT_G)
            i = int(hit[0]) if len(hit) else int(np.argmax(svm))
            offset = FADE if start > 0 else 0          # samples lost to the cross-fade
            self.events.append(dict(t=round(start + max(i - offset, 0) / FS, 2),
                                    what=f"fall impact ({code})"))
        return self

    def still(self, seconds, label="lying still (synthetic)"):
        self._lead([0, 0, 1.0])
        o = self.acc[-FADE:].mean(axis=0)
        self._rest(seconds, o, label)
        return self

    def trace(self, points, noise):
        """Piecewise-linear trace through (t, value) points, plus noise."""
        t = np.arange(len(self.acc)) / FS
        pt, pv = zip(*sorted(points))
        ts = np.arange(0, t[-1] + 2)                    # slow noise, one value per second
        wobble = np.interp(t, ts, self.rng.normal(0, noise, len(ts)))
        return np.interp(t, pt, pv) + wobble

    def impact_t(self):
        return self.events[0]["t"] if self.events else None

    def build(self, name, title, description, hr_points, temp_points=None, clock=None):
        end = self.now
        hr = self.trace([(0, hr_points[0][1])] + hr_points + [(end, hr_points[-1][1])], 1.5)
        temp_points = temp_points or [(0, 33.2), (end, 33.4)]
        temp = self.trace(temp_points, 0.05)
        pfs = ppg_synth.PPG_FS
        tp = np.arange(int(len(self.acc) / FS * pfs)) / pfs
        t50 = np.arange(len(self.acc)) / FS
        ppg = ppg_synth.from_hr(np.interp(tp, t50, hr), pfs,
                                acc_mag=np.interp(tp, t50, np.linalg.norm(self.acc, axis=1)),
                                seed=int(self.rng.integers(1 << 30)))
        return Scenario(name, title, description, self.acc, self.gyro, hr, temp,
                        self.truth, self.events, ppg=ppg, ppg_fs=pfs, clock=clock)


# --------------------------------------------------------------------------
# the six scenarios
# --------------------------------------------------------------------------

def normal_day(lib):
    b = Builder(lib, 1)
    for code in ("D01", "D04", "D11", "D10", "D01"):
        b.record("U21", code)
    return b.build("normal_day", "Normal day (elderly)",
                   "Walking, sitting down and getting up, a door, clapping, walking. "
                   "Nothing should be flagged.",
                   [(0, 74), (b.now * 0.3, 82), (b.now * 0.6, 76), (b.now, 78)])


def hard_fall_long_lie(lib):
    b = Builder(lib, 2).record("U04", "D01").record("U04", "F01").still(60)
    t = b.impact_t()
    return b.build("hard_fall_long_lie", "Hard fall, then a long lie",
                   "Walking, a forward slip fall, then 60 s without moving. "
                   "The motion stream should flag the impact; stillness should stay high.",
                   [(0, 76), (t, 80), (t + 20, 104), (t + 60, 100)])


def fall_then_get_up(lib):
    b = Builder(lib, 3).record("U07", "D01").record("U07", "F04").still(8, "on the floor (synthetic)")
    b.record("U07", "D04").record("U07", "D01")
    t = b.impact_t()
    return b.build("fall_then_get_up", "Fall, then the person gets up",
                   "Walking, a trip fall, 8 s on the floor, then sitting, getting up and "
                   "walking. The impact is flagged but stillness should not stay high.",
                   [(0, 75), (t, 80), (t + 15, 96), (t + 45, 86)])


def faint_collapse(lib):
    b = Builder(lib, 4).record("U08", "D04").record("U08", "F07").still(60)
    t = b.impact_t()
    return b.build("faint_collapse", "Faint-type collapse from sitting",
                   "Sitting, then a backward fall from the chair as in a faint (acted in "
                   "WEDA-FALL), then 60 s still. Heart rate dips before the fall in this "
                   "synthetic trace.",
                   [(0, 72), (max(t - 8, 0), 70), (t, 52), (t + 30, 62), (t + 60, 66)])


def false_alarm(lib):
    b = Builder(lib, 5)
    for code in ("D01", "D09", "D07", "D01"):
        b.record("U10", code)
    return b.build("false_alarm", "Hard knocks that are not falls",
                   "Walking, hitting a table with the hand, a stumble, walking. The gate "
                   "may open; the motion score should stay low.",
                   [(0, 76), (b.now, 80)])


def hr_spike_no_fall(lib):
    b = Builder(lib, 6).record("U22", "D04").record("U22", "D11").still(60, "sitting quietly (synthetic)")
    t0 = b.truth[-1]["start"]
    return b.build("hr_spike_no_fall", "Heart-rate spike while sitting, no fall",
                   "Sitting, opening a door, then sitting quietly while heart rate climbs "
                   "to 150 bpm. Motion should stay quiet; this is for the physiological "
                   "stream and fusion.",
                   [(0, 74), (t0, 78), (t0 + 20, 150), (t0 + 60, 148)])


def night_sleep(lib):
    b = Builder(lib, 7).record("U29", "D01").record("U29", "D04")
    b.still(120, "lying in bed, asleep (synthetic)")
    t0 = b.truth[-1]["start"]
    return b.build("night_sleep", "Going to bed at night (elderly)",
                   "Late evening (starts 23:30): walking, sitting on the bed, then two "
                   "minutes lying still with a calm heart rate. Long stillness alone is "
                   "not an emergency: nothing should be raised.",
                   [(0, 74), (t0, 68), (t0 + 60, 63), (t0 + 120, 62)], clock="23:30")


SCENARIOS = {f.__name__: f for f in (normal_day, hard_fall_long_lie, fall_then_get_up,
                                     faint_collapse, false_alarm, hr_spike_no_fall,
                                     night_sleep)}


# --------------------------------------------------------------------------
# live demo only (9 Oct 2026). NOT a tuning or test scenario: fusion was
# frozen and tested before this existed, and simulator.run all skips it.
# --------------------------------------------------------------------------

def demo_long_lie(lib):
    """hard_fall_long_lie with a longer lie (340 s instead of 60 s), to show
    on the live screen that motion's impact score wears off (held 60 s, then
    decays, gone about 3 min after the fall) while fusion keeps the impact
    and long lie stays high, up to URGENT at 300 s without movement."""
    b = Builder(lib, 2).record("U04", "D01").record("U04", "F01").still(340)
    t = b.impact_t()
    return b.build("demo_long_lie", "Hard fall, then a long lie (5+ min)",
                   "Walking, a forward slip fall, then 340 s without moving. The motion "
                   "impact score wears off after about a minute; the emergency score "
                   "should stay high and go urgent at 300 s still.",
                   [(0, 76), (t, 80), (t + 20, 104), (t + 60, 100), (t + 120, 92)])


DEMOS = {"demo_long_lie": demo_long_lie}


def build(name, root=weda.DEFAULT_ROOT) -> Scenario:
    return {**SCENARIOS, **DEMOS}[name](Library(root))

"""
simulator/test_scenarios.py -- HELD-OUT fusion test scenarios T1-T7 (Stage 7.4).

Agreed and locked on 8 Oct 2026, before any of them was built or run. Fusion
was tuned on the 7 scenarios in simulator/scenarios.py only; these are run
once with the frozen settings (py -m testing.fusion_eval test).

Same recipe as the tuning scenarios: real WEDA-FALL wrist motion from test
people the motion model never trained on, synthetic still periods, synthetic
heart rate and PPG, and the 30 s resting lead-in.

Build details fixed now, before the first run:
- "short still" in T2 is 6 s; "~10 s still" in T5 is 10 s.
- T2 and T5 end with the recorded walk (D01) played twice, so the person
  keeps moving long enough for the outcome to be decided inside the scenario.
- T7: heart rate 80 at rest, rising to 115 once walking starts.
- T6: heart rate 72, falling to 42 over 10 s, then 42 for 2 min, sitting.
- Missing recordings (added 8 Oct 2026 after the first attempt stopped while
  BUILDING T3: U22 has no D03; no test outcome had been computed or shown).
  A daily activity a person never recorded is replaced by the first one they
  do have from SUBSTITUTE below, fixed before any outcome was seen, and the
  swap is written into the scenario description and printed. Falls are never
  substituted: a missing fall stops the test before anything runs.

  ID  scenario                                                    expected
  T1  U10 walk, backward slip (F03), 90 s still                   caregiver
  T2  U07 walk, lateral slip (F02), 6 s still, walks              alert, cleared, no caregiver
  T3  U22 busy day: stairs, hits table, claps, walks              nothing
  T4  U04 sitting, faint forward (F06), HR dips to 48, 90 s still caregiver
  T5  U08 fall trying to sit (F05), 10 s still, sit/stand, walks  check-in or alert, cleared
  T6  U21 sitting, then sits still, HR sinks to 42 for 2 min      caregiver (provisional)
  T7  U29 walking and stairs, HR 115, no fall                     nothing
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data import loader_weda as weda  # noqa: E402
from simulator.scenarios import Builder, Library  # noqa: E402


# same family of activity first, then the nearest one
SUBSTITUTE = {
    "D03": ["D01"],                 # stairs -> walking
    "D09": ["D10", "D11"],          # hit table -> clapping, door
    "D10": ["D09", "D11"],          # clapping -> hit table, door
    "D11": ["D10", "D09"],
    "D04": ["D05", "D06"],          # sit / get up -> collapse into chair, crouch
}


class TBuilder(Builder):
    def __init__(self, lib, seed):
        super().__init__(lib, seed)
        self.swaps = []

    def record(self, subject, code, rep=1):
        try:
            return super().record(subject, code, rep)
        except LookupError:
            if code.startswith("F"):
                raise
            for alt in SUBSTITUTE.get(code, []):
                try:
                    super().record(subject, alt, rep)
                except LookupError:
                    continue
                self.swaps.append(f"{code}->{alt} ({subject} has no {code})")
                return self
            raise

    def build(self, name, title, description, *a, **k):
        if self.swaps:
            description += " Substituted: " + "; ".join(self.swaps) + "."
        sc = super().build(name, title, description, *a, **k)
        sc.swaps = list(self.swaps)
        return sc


def T1(lib):
    b = TBuilder(lib, 11).record("U10", "D01").record("U10", "F03").still(90)
    t = b.impact_t()
    return b.build("T1", "T1: backward slip, then a long lie",
                   "Walking, a backward slip fall, then 90 s without moving.",
                   [(0, 75), (t, 80), (t + 30, 92), (t + 90, 95)])


def T2(lib):
    b = TBuilder(lib, 12).record("U07", "D01").record("U07", "F02").still(6, "on the floor (synthetic)")
    b.record("U07", "D01").record("U07", "D01")
    t = b.impact_t()
    return b.build("T2", "T2: lateral slip, gets up and walks",
                   "Walking, a lateral slip fall, 6 s on the floor, then walks away "
                   "(recorded walk played twice).",
                   [(0, 76), (t, 82), (t + 15, 95), (t + 40, 88)])


def T3(lib):
    b = TBuilder(lib, 13)
    for code in ("D03", "D09", "D10", "D01"):
        b.record("U22", code)
    return b.build("T3", "T3: busy day (elderly)",
                   "Stairs, hitting a table with the hand, clapping, walking. Nothing "
                   "should be raised.",
                   [(0, 74), (b.now * 0.4, 88), (b.now * 0.7, 80), (b.now, 78)])


def T4(lib):
    b = TBuilder(lib, 14).record("U04", "D04").record("U04", "F06").still(90)
    t = b.impact_t()
    return b.build("T4", "T4: faint forward from sitting, slow heart rate",
                   "Sitting, a forward fall from the chair as in a faint, heart rate "
                   "dips to 48, then 90 s still.",
                   [(0, 72), (max(t - 8, 0), 68), (t, 48), (t + 30, 50), (t + 90, 52)])


def T5(lib):
    b = TBuilder(lib, 15).record("U08", "F05").still(10, "on the floor (synthetic)")
    b.record("U08", "D04").record("U08", "D01").record("U08", "D01")
    t = b.impact_t()
    return b.build("T5", "T5: falls trying to sit, gets up",
                   "A backward fall while trying to sit down, 10 s on the floor, "
                   "then sits, gets up and walks (recorded walk played twice).",
                   [(0, 74), (t, 80), (t + 20, 92), (t + 50, 86)])


def T6(lib):
    b = TBuilder(lib, 16).record("U21", "D04").still(130, "sitting quietly (synthetic)")
    t0 = b.truth[-1]["start"]
    return b.build("T6", "T6: heart rate sinks to 42 while sitting, no fall",
                   "Sitting down, then sitting still while heart rate falls to 42 bpm "
                   "and stays there for 2 min. No impact.",
                   [(0, 72), (t0, 72), (t0 + 10, 42), (t0 + 130, 42)])


def T7(lib):
    b = TBuilder(lib, 17)
    for code in ("D01", "D03", "D01", "D03"):
        b.record("U29", code)
    t0 = b.truth[1]["start"]                     # first recording, after the lead-in
    return b.build("T7", "T7: walking with a high heart rate (elderly)",
                   "Walking and stairs with heart rate 115 bpm. Walking explains it: "
                   "nothing should be raised.",
                   [(0, 80), (t0, 80), (t0 + 10, 115), (b.now, 115)])


TESTS = {
    "T1": dict(f=T1, short="backward slip, long lie"),
    "T2": dict(f=T2, short="lateral slip, walks"),
    "T3": dict(f=T3, short="busy day"),
    "T4": dict(f=T4, short="faint, HR 48"),
    "T5": dict(f=T5, short="fall sitting, gets up"),
    "T6": dict(f=T6, short="HR 42 sitting"),
    "T7": dict(f=T7, short="walking, HR 115"),
}


def build_test(tid, root=weda.DEFAULT_ROOT):
    return TESTS[tid]["f"](Library(root))

"""
simulator/failure_scenarios.py -- cases built to find where fusion FAILS
(Stage 7.6). Written 9 Oct 2026, after the held-out test, with the settings
frozen (0a3011d7b96be571). Nothing here is used for tuning; any fix that
comes out of it is reported as made after the test.

Same recipe as the other scenarios (real WEDA-FALL wrist motion, synthetic
stillness, heart rate and PPG, 30 s lead-in). For each case the file says
what a good system should do and what we PREDICT ours will do, written
before running it.

  F1 soft collapse, no impact: sits, slumps, stays still 2 min, normal HR
     good: alert within minutes      predict: nothing (looks like resting)
  F2 fall, struggles 20 s on the floor, then still 90 s
     good: caregiver                 predict: alert, wrongly cleared during
                                     the struggle, alert again, caregiver late
  F3 fall at night (02:30), then still 90 s
     good: caregiver                 predict: caregiver (night only lowers
                                     the unresponsive type)
  F4 fall, then keeps moving the arms on the floor (cannot get up)
     good: caregiver                 predict: cleared, caregiver never told
  F5 daytime nap, 35 min still, normal HR, nobody answers the check-in
     good: nothing, or a check-in    predict: check-in at 30 min, urgent
                                     2 min later (false caregiver alert)

Added 9 Oct 2026 together with the v2 fixes (so NOT independent of them):
  F6 fall, then still for 6 min
     good: caregiver, urgent at 5 min still
     predict v1: caregiver, never urgent (the impact fades, long lie closes)
  F7 fall, gets up, walks, then rests in a chair for 60 s
     good: alert, cleared, nothing after
     predict v1: may raise a second check-in while resting (impact still held)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data import loader_weda as weda  # noqa: E402
from simulator.scenarios import Library  # noqa: E402
from simulator.test_scenarios import TBuilder  # noqa: E402


def F1(lib):
    b = TBuilder(lib, 21).record("U21", "D04").still(120, "slumped, not moving (synthetic)")
    t0 = b.truth[-1]["start"]
    return b.build("F1", "F1: soft collapse, no impact",
                   "Sitting, then slumps and does not move for 2 min. No impact, heart "
                   "rate stays normal.", [(0, 72), (t0, 70), (t0 + 120, 68)])


def F2(lib):
    b = TBuilder(lib, 22).record("U10", "D01").record("U10", "F01")
    b.record("U10", "D10").record("U10", "D09")            # arm movement on the floor
    b.still(90)
    t = b.impact_t()
    return b.build("F2", "F2: fall, struggles, then lies still",
                   "Walking, a forward slip, about 20 s of arm movement on the floor "
                   "(recorded clapping and table hits), then 90 s still.",
                   [(0, 76), (t, 82), (t + 30, 96), (t + 110, 98)])


def F3(lib):
    b = TBuilder(lib, 23).record("U07", "D01").record("U07", "F01").still(90)
    t = b.impact_t()
    return b.build("F3", "F3: fall at night",
                   "Night (starts 02:30): walking to the bathroom, a forward slip, then "
                   "90 s still.", [(0, 66), (t, 72), (t + 30, 88), (t + 90, 90)],
                   clock="02:30")


def F4(lib):
    b = TBuilder(lib, 24).record("U08", "D01").record("U08", "F04")
    for code in ("D10", "D09", "D10", "D11"):              # cannot get up, keeps moving
        b.record("U08", code)
    t = b.impact_t()
    return b.build("F4", "F4: fall, cannot get up, keeps moving the arms",
                   "Walking, a trip fall, then the arms keep moving on the floor "
                   "(recorded clapping, table hits, door) without getting up.",
                   [(0, 76), (t, 84), (t + 20, 98), (t + 60, 100)])


def F5(lib):
    b = TBuilder(lib, 25).record("U22", "D04").still(2100, "daytime nap (synthetic)")
    t0 = b.truth[-1]["start"]
    return b.build("F5", "F5: daytime nap, 35 min",
                   "Sits down and naps for 35 min in the afternoon, heart rate calm. "
                   "In the simulation nobody answers the band.",
                   [(0, 72), (t0, 66), (t0 + 600, 62), (t0 + 2100, 63)], clock="14:30")


def F6(lib):
    b = TBuilder(lib, 26).record("U04", "D01").record("U04", "F02").still(360)
    t = b.impact_t()
    return b.build("F6", "F6: fall, then 6 min on the floor",
                   "Walking, a lateral slip, then 6 min without moving.",
                   [(0, 75), (t, 80), (t + 30, 92), (t + 360, 90)])


def F7(lib):
    b = TBuilder(lib, 27).record("U10", "D01").record("U10", "F04").still(6, "on the floor (synthetic)")
    b.record("U10", "D04").record("U10", "D01").record("U10", "D01")
    b.still(60, "resting in a chair (synthetic)")
    t = b.impact_t()
    return b.build("F7", "F7: fall, gets up, then rests",
                   "Walking, a trip fall, 6 s on the floor, gets up and walks, then rests "
                   "in a chair for 60 s.", [(0, 76), (t, 84), (t + 30, 96), (t + 90, 80)])


FAILURES = {
    "F1": dict(f=F1, short="soft collapse, no impact", good="alert", predict="nothing"),
    "F2": dict(f=F2, short="struggles, then still", good="caregiver", predict="caregiver (late)"),
    "F3": dict(f=F3, short="fall at night", good="caregiver", predict="caregiver"),
    "F4": dict(f=F4, short="cannot get up, moves arms", good="caregiver", predict="alert, cleared"),
    "F5": dict(f=F5, short="daytime nap 35 min", good="nothing / check-in",
               predict="check-in, urgent"),
    "F6": dict(f=F6, short="fall, 6 min on floor", good="caregiver, urgent",
               predict="v1: caregiver, no urgent"),
    "F7": dict(f=F7, short="gets up, then rests", good="alert, cleared",
               predict="v1: 2nd check-in?"),
}


def build_failure(fid, root=weda.DEFAULT_ROOT):
    return FAILURES[fid]["f"](Library(root))

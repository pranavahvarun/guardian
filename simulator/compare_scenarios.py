"""
simulator/compare_scenarios.py -- cases for the comparison of the Emergency
Risk Estimation Module (fusion v3 + Alert Escalation Module) against an
IMPACT-ONLY detector (core/baseline.py). Written 20 Oct 2026.

Locked BEFORE any of them was built or run: the situations, the recordings
and the expected outcome of each. testing/compare_impact_only.py prints a
fingerprint of this table with every result. Do not edit after the first run.

Fusion is NOT tuned on these. Its weights and rules are the frozen v3 settings
(core/fusion.py, core/alerts.py), unchanged.

Same recipe as every other scenario: real WEDA-FALL wrist motion from the
test people the impact model never trained on (U04 U07 U08 U10 young,
U21 U22 U29 aged 77 to 95), synthetic still periods, synthetic heart rate and
PPG, and the 30 s resting lead-in. To keep these independent of the tuning
scenarios, the held-out test and the failure study (which all used
repetition 1), every recording here is repetition 2 or 3 (R02, R03).

Expected outcome (what a good system should do):
    caregiver      caregiver told (urgent also counts)
    no caregiver   caregiver never told (a check-in or a cleared alert is allowed)
    no alert       at most a check-in; no alert, caregiver never told
    urgent         caregiver told as URGENT

  ID   situation                                                     expected
  C1   U07 walks, backward slip (F03), lies still 2 min              caregiver
  C2   U08 walks, forward slip (F01), 6 s on the floor, gets up,     no caregiver
       sits, walks away
  C3   U04 falls trying to sit down (F05), lies still 2 min          caregiver
  C4   U10 sitting, faints sideways (F08), HR drops to 45, still     caregiver
       2 min
  C5   U21 (elderly) walks, hits a table, claps, opens a door,       no caregiver
       walks
  C6   U08 walks, stumbles, jumps to reach something, stumbles,      no caregiver
       walks
  C7   U22 (elderly) sits, HR falls to 40 and stays there 2.5 min,   caregiver (provisional:
       no impact                                                      HR thresholds)
  C8   U29 (elderly) walking with HR 118                             no caregiver
  C9   U21 (elderly) afternoon nap, 35 min still, calm HR            no alert
  C10  U22 (elderly) slumps in a chair and does not respond for      urgent
       45 min, daytime, no impact, HR normal
  C11  U10 walks, lateral slip (F02), lies still 2 min, NO PPG       caregiver
       (heart rate sensor off)
  C12  U29 (elderly) sits, slumps with no impact, still 2.5 min,     caregiver (known gap:
       HR normal                                                      fusion is expected to miss it)

Event time ("onset") for the timing columns: the fall impact for falls; for
the others, when the change starts (heart rate starts falling, stillness
starts). Kept in events[0] like the other scenario files.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data import loader_weda as weda  # noqa: E402
from simulator.scenarios import Library  # noqa: E402
from simulator.test_scenarios import TBuilder  # noqa: E402

LOCKED = "20 Oct 2026, before any case was built or run"


def _onset(b, what):
    b.events.append(dict(t=round(b.truth[-1]["start"], 2), what=what))


def C1(lib):
    b = TBuilder(lib, 101).record("U07", "D01", 2).record("U07", "F03", 2).still(120)
    t = b.impact_t()
    return b.build("C1", "C1: backward slip, lies still",
                   "Walking, a backward slip fall, then 2 min without moving.",
                   [(0, 76), (t, 80), (t + 30, 95), (t + 120, 96)])


def C2(lib):
    b = TBuilder(lib, 102).record("U08", "D01", 2).record("U08", "F01", 2)
    b.still(6, "on the floor (synthetic)")
    b.record("U08", "D04", 2).record("U08", "D01", 2).record("U08", "D01", 3)
    t = b.impact_t()
    return b.build("C2", "C2: forward slip, gets up and walks away",
                   "Walking, a forward slip fall, 6 s on the floor, then sits, gets up "
                   "and walks away.", [(0, 76), (t, 82), (t + 15, 96), (t + 45, 88)])


def C3(lib):
    b = TBuilder(lib, 103).record("U04", "F05", 2).still(120)
    t = b.impact_t()
    return b.build("C3", "C3: falls trying to sit down, lies still",
                   "A backward fall while trying to sit down, then 2 min without moving.",
                   [(0, 74), (t, 80), (t + 30, 92), (t + 120, 94)])


def C4(lib):
    b = TBuilder(lib, 104).record("U10", "D04", 2).record("U10", "F08", 2).still(120)
    t = b.impact_t()
    return b.build("C4", "C4: faints sideways from a chair, slow heart rate",
                   "Sitting, a sideways fall from the chair as in a faint, heart rate "
                   "drops to 45, then 2 min still.",
                   [(0, 72), (max(t - 8, 0), 66), (t, 46), (t + 30, 45), (t + 120, 48)])


def C5(lib):
    b = TBuilder(lib, 105)
    for code in ("D01", "D09", "D10", "D11", "D01"):
        b.record("U21", code, 2)
    return b.build("C5", "C5: busy day with hard knocks (elderly)",
                   "Walking, hitting a table with the hand, clapping, opening a door, "
                   "walking.", [(0, 74), (b.now * 0.5, 84), (b.now, 78)])


def C6(lib):
    b = TBuilder(lib, 106)
    for code, rep in (("D01", 2), ("D07", 2), ("D08", 2), ("D07", 3), ("D01", 3)):
        b.record("U08", code, rep)
    return b.build("C6", "C6: stumbles and a jump, no fall",
                   "Walking, a stumble, a gentle jump to reach something, another "
                   "stumble, walking.", [(0, 78), (b.now * 0.5, 96), (b.now, 90)])


def C7(lib):
    b = TBuilder(lib, 107).record("U22", "D04", 2).still(150, "sitting quietly (synthetic)")
    t0 = b.truth[-1]["start"]
    _onset(b, "heart rate starts falling")
    return b.build("C7", "C7: heart rate falls to 40 while sitting, no impact",
                   "Sitting down, then sitting still while heart rate falls to 40 bpm and "
                   "stays there for 2.5 min.",
                   [(0, 72), (t0, 72), (t0 + 10, 40), (t0 + 150, 40)])


def C8(lib):
    b = TBuilder(lib, 108)
    for rep in (2, 3, 2, 3):
        b.record("U29", "D01", rep)
    t0 = b.truth[1]["start"]
    return b.build("C8", "C8: walking with a high heart rate (elderly)",
                   "Walking with heart rate 118 bpm. Walking explains it.",
                   [(0, 80), (t0, 80), (t0 + 10, 118), (b.now, 118)])


def C9(lib):
    b = TBuilder(lib, 109).record("U21", "D04", 2).still(2100, "afternoon nap (synthetic)")
    t0 = b.truth[-1]["start"]
    _onset(b, "nap starts")
    return b.build("C9", "C9: afternoon nap, 35 min (elderly)",
                   "Sits down and naps for 35 min, heart rate calm. Nobody answers the "
                   "band in the simulation.",
                   [(0, 72), (t0, 66), (t0 + 600, 62), (t0 + 2100, 63)], clock="14:30")


def C10(lib):
    b = TBuilder(lib, 110).record("U22", "D04", 2).still(2700, "slumped, not responding (synthetic)")
    t0 = b.truth[-1]["start"]
    _onset(b, "stops responding")
    return b.build("C10", "C10: does not respond for 45 min, daytime (elderly)",
                   "Sitting, then slumps and does not move or answer for 45 min. No "
                   "impact, heart rate normal.",
                   [(0, 72), (t0, 70), (t0 + 2700, 66)], clock="11:00")


def C11(lib):
    b = TBuilder(lib, 111).record("U10", "D01", 2).record("U10", "F02", 2).still(120)
    t = b.impact_t()
    sc = b.build("C11", "C11: lateral slip, lies still, heart-rate sensor off",
                 "Walking, a lateral slip fall, then 2 min without moving. No PPG at all "
                 "(sensor off or not touching the skin).",
                 [(0, 76), (t, 80), (t + 30, 92), (t + 120, 94)])
    sc.ppg = None                         # the Heart Rate Monitoring Module gets nothing
    return sc


def C12(lib):
    b = TBuilder(lib, 112).record("U29", "D04", 2).still(150, "slumped, not moving (synthetic)")
    t0 = b.truth[-1]["start"]
    _onset(b, "slumps")
    return b.build("C12", "C12: soft collapse, no impact (elderly)",
                   "Sitting, then slumps and does not move for 2.5 min. No impact, heart "
                   "rate stays normal.", [(0, 72), (t0, 70), (t0 + 150, 68)])


CASES = {
    "C1": dict(f=C1, short="backward slip, lies still", expected="caregiver"),
    "C2": dict(f=C2, short="forward slip, gets up", expected="no caregiver"),
    "C3": dict(f=C3, short="falls sitting down, still", expected="caregiver"),
    "C4": dict(f=C4, short="faint sideways, HR 45", expected="caregiver"),
    "C5": dict(f=C5, short="hard knocks, daily (elderly)", expected="no caregiver"),
    "C6": dict(f=C6, short="stumbles and a jump", expected="no caregiver"),
    "C7": dict(f=C7, short="HR 40 sitting, no impact", expected="caregiver", provisional=True),
    "C8": dict(f=C8, short="walking, HR 118 (elderly)", expected="no caregiver"),
    "C9": dict(f=C9, short="35 min nap (elderly)", expected="no alert"),
    "C10": dict(f=C10, short="no response 45 min (elderly)", expected="urgent"),
    "C11": dict(f=C11, short="lateral slip, still, no PPG", expected="caregiver"),
    "C12": dict(f=C12, short="soft collapse, no impact", expected="caregiver", known_gap=True),
}


def build_case(cid, root=weda.DEFAULT_ROOT):
    return CASES[cid]["f"](Library(root))

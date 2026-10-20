"""
demo/motion.py -- review output 1: the motion module, SAVED results.
Reads testing/outputs/stage35/step6_test.csv (one-time locked test).
To run the module on the data instead: py -m demo.motion_eval

    py -m demo.motion
"""

from __future__ import annotations

import pandas as pd

from demo.common import OUT, need, rows, section, title


def show(t, note_same, note_cross):
    """t: one row per test set (index WEDA-FALL / UMAFall / FallAllD) with
    falls, adls, auc_gated, falls_caught, false_alarms_per_hour,
    faint_caught, elder_adl_false_alarms and optionally false_alarm_count."""
    title("1. IMPACT AND IMMOBILITY DETECTION MODULE",
          "Did a hard impact happen, and has the person moved since?")
    w = t.loc["WEDA-FALL"]
    caught = int(round(w.falls_caught * w.falls))
    fa = w.get("false_alarm_count")
    fa = int(fa) if fa is not None and pd.notna(fa) else (0 if w.false_alarms_per_hour == 0 else None)
    section(f"SAME DATASET, NEW PEOPLE   WEDA-FALL, {note_same}")
    rows([
        f"{'falls caught':34s}{caught} of {int(w.falls)}  ({w.falls_caught:.0%})",
        f"{'false alarms':34s}" + (f"{fa} of {int(w.adls)} daily activities" if fa is not None
                                   else f"{int(w.adls)} daily activities")
        + f"  ({w.false_alarms_per_hour:.1f} per hour)",
        f"{'gated AUC':34s}{w.auc_gated:.3f}",
        f"{'faint-type falls caught':34s}{w.faint_caught:.1%}",
        f"{'older adults (77-95), daily life':34s}{w.elder_adl_false_alarms} false alarms",
    ])
    section(f"DATASETS NEVER TRAINED ON   {note_cross}")
    rows([f"{'dataset':22s}{'falls':>7s}{'daily acts':>12s}{'gated AUC':>11s}"
          f"{'falls caught':>14s}{'false alarms/h':>16s}"])
    for name, label in (("UMAFall", "UMAFall"), ("FallAllD", "FallAllD"),
                        ("WEDA-FALL", "WEDA-FALL (same)")):
        r = t.loc[name]
        rows([f"{label:22s}{int(r.falls):7d}{int(r.adls):12d}{r.auc_gated:11.3f}"
              f"{r.falls_caught:14.0%}{r.false_alarms_per_hour:16.1f}"])
    print()


def main():
    t = pd.read_csv(need(OUT / "stage35" / "step6_test.csv", "py -m testing.stage35"))
    t = t[t.model.str.startswith("WEDA")].set_index("test")
    show(t, "7 locked people, one-time test", "other labs and devices, locked people")


if __name__ == "__main__":
    main()

"""
demo/activity.py -- review output 2: the activity module, SAVED results.
Reads testing/outputs/stage6/activity_same.txt and activity_cross.txt.
To run the module on the data instead: py -m demo.activity_eval

    py -m demo.activity
"""

from __future__ import annotations

import re

from demo.common import OUT, need, rows, section, title

CLASSES = ("LOCOMOTION", "SITTING_STANDING", "UPPER_BODY", "OTHER")
NICE = {"LOCOMOTION": "walking, stairs", "SITTING_STANDING": "sit down / stand up",
        "UPPER_BODY": "arm and hand tasks", "OTHER": "crouch, stumble, jump"}


def parse(path):
    """(metrics in %, per-class (recall %, windows), people) from a saved report."""
    m, per, people = {}, {}, ""
    for line in path.read_text().splitlines():
        s = line.strip()
        g = re.match(r"(accuracy|balanced accuracy|macro F1)\s+.*?([\d.]+)%\s*$", s)
        if g:
            m[g.group(1)] = float(g.group(2))
            continue
        g = re.match(r"(" + "|".join(CLASSES) + r")\s+(\d+)%\s+\((\d+) windows\)", s)
        if g and g.group(1) not in per:
            per[g.group(1)] = (int(g.group(2)), int(g.group(3)))
        g = re.search(r"(\d+) recordings, (\d+) people", s) or re.search(r"each of the (\d+) people", s)
        if g:
            people = f"{g.group(g.lastindex)} people"
    return m, per, people


def show(same, cross):
    """same / cross: (metrics in %, {class: (recall %, windows)}, 'N people')."""
    title("2. ACTIVITY RECOGNITION MODULE", "What is the person doing?")
    section("RESULTS   same dataset (leave-one-person-out) and a dataset never trained on")
    rows([f"{'':30s}{'same dataset':>16s}{'never trained on':>20s}",
          f"{'':30s}{'WEDA-FALL':>16s}{'UMAFall':>20s}",
          f"{'':30s}{same[2]:>16s}{cross[2]:>20s}"])
    for k in ("accuracy", "balanced accuracy", "macro F1"):
        rows([f"{k:30s}{same[0].get(k, float('nan')):15.1f}%{cross[0].get(k, float('nan')):19.1f}%"])
    print()
    rows(["caught, per state:"])
    for c in CLASSES:
        a = f"{same[1][c][0]}%" if c in same[1] else "-"
        b = f"{cross[1][c][0]}%" if c in cross[1] else "-"
        rows([f"  {c:17s}{'(' + NICE[c] + ')':24s}{a:>5s}{b:>20s}"])
    print()


def main():
    cmd = "py -m testing.activity_report_tables same   (and: cross)"
    show(parse(need(OUT / "stage6" / "activity_same.txt", cmd)),
         parse(need(OUT / "stage6" / "activity_cross.txt", cmd)))


if __name__ == "__main__":
    main()

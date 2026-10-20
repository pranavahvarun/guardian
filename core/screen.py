"""
core/screen.py -- the table both live screens print, so the band side
(simulator/live.py) and the caregiver side (core/watch.py) look the same.

Columns, in pipeline order:
    time         seconds since the run started (end of the window)
    impact       Impact and Immobility Detection Module: impact score, 0 to 1
    still (s)    Impact and Immobility Detection Module: seconds without movement
    activity     Activity Recognition Module: current state
    heart (bpm)  Heart Rate Monitoring Module: heart rate ("--" until it has one)
    emergency    Emergency Risk Estimation Module: emergency score, 0 to 1
(module names: core/names.py; the stored stream keys did not change)
Caregiver side only: the five type scores before emergency (hard fall,
long lie, faint, heart rate, unresponsive); emergency is the highest of them.
Alert events are added at the end of the row they happen on.
"""

from __future__ import annotations

from core.names import LEGEND

COLS = (("time", 8, ">"), ("impact", 6, ">"), ("still (s)", 9, ">"),
        ("activity", 16, "<"), ("heart (bpm)", 11, ">"))
LAST = ("emergency", 9, ">")
GAP = "   "
# caregiver side only: the five type scores fusion computes; emergency = the highest
TYPES = (("hard_fall", "fall"), ("long_lie", "long lie"), ("faint", "faint"),
         ("heart_rate", "HR"), ("unresponsive", "unresp."))
TYPE_W = 8

EVENT = {"check-in": "CHECK-IN: {k}", "alert": "ALERT: {k} (30 s for the wearer to cancel)",
         "caregiver alerted": "CAREGIVER ALERTED: {k}", "urgent: long lie": "URGENT: long lie",
         "urgent: check-in not answered": "URGENT: check-in not answered",
         "check-in unanswered": "no answer, asking again in 5 min",
         "second check-in": "SECOND CHECK-IN: {k}", "cleared": "cleared",
         "cancelled by wearer": "cancelled by wearer", "wearer answered OK": "wearer answered OK",
         "recovered": "moving again (alert stays open)"}


def _fmt(vals, types=None, last=""):
    line = "  " + GAP.join(f"{v:{al}{w}}" for v, (_, w, al) in zip(vals, COLS))
    if types is not None:
        line += GAP + "  ".join(f"{v:>{TYPE_W}}" for v in types)
    return line + GAP + f"{last:{LAST[2]}{LAST[1]}}"


def legend():
    """Which module each column comes from (printed once, above the headings)."""
    return ["  " + x for x in LEGEND] + [""]


def header(types=False):
    """Column headings (printed once). types=True adds the five type scores."""
    names = [n for n, _, _ in COLS]
    dash = ["-" * w for _, w, _ in COLS]
    tn = [n for _, n in TYPES] if types else None
    td = ["-" * TYPE_W for _ in TYPES] if types else None
    lines = [_fmt(names, tn, LAST[0]), _fmt(dash, td, "-" * LAST[1])]
    if types:
        base = len(_fmt(names)) - LAST[1]
        span = TYPE_W * len(TYPES) + 2 * (len(TYPES) - 1)
        title = "type scores, 0 to 1 (emergency = highest)"
        lines.insert(0, " " * base + f"{' ' + title + ' ':-^{span}}")
    return lines


def marker(events, label):
    """events: alert manager events (dicts with 'event' and 'type')."""
    out = []
    for e in events:
        out.append(EVENT.get(e["event"], e["event"]).format(k=label.get(e["type"], e["type"])))
    return ("   << " + "; ".join(out)) if out else ""


def row(t, motion=None, still=None, activity=None, hr=None, score=None, types=None):
    """types: dict of the five type scores (caregiver side), or None."""
    def num(x, f):
        return "--" if x is None else format(x, f)
    tv = None if types is None else [num(types.get(k), ".2f") for k, _ in TYPES]
    return _fmt([f"{t:.1f} s", num(motion, ".2f"), num(still, ".1f"), activity or "--",
                 num(hr, ".0f")], tv, num(score, ".2f"))


def from_records(rows, score):
    """One table row from a window's pipeline records (stream -> extras)."""
    by = {r["stream"]: r for r in rows}
    m, a, p = by.get("motion"), by.get("activity"), by.get("physiological")
    return row(rows[0]["t"],
               m["score"] if m else None,
               (m["extras"].get("still_s") or 0.0) if m else None,
               a["extras"].get("state") if a else None,
               p["extras"].get("hr") if p else None,
               score)

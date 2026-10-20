"""
core/baseline.py -- an IMPACT-ONLY detector, used only as a comparison for the
Emergency Risk Estimation Module (20 Oct 2026). It is not part of Guardian.

It stands for the usual wrist fall detector: one signal, one threshold.

    IDLE       -> ALERT       impact score (Impact and Immobility Detection
                              Module, raw) reaches IMPACT_ALERT = 0.7, the same
                              line our alert uses
    ALERT      -> CAREGIVER   CANCEL_S = 30 s later, the same cancel window
                              as core/alerts.py

No recovery, no stillness, no heart rate, no activity, no time of day. Nobody
presses cancel in the simulations, for either system, so every impact alert
reaches the caregiver.

It reads the same pipeline records as fusion (one list per window), so both
see exactly the same module outputs in the comparison.
"""

from __future__ import annotations

IMPACT_ALERT = 0.7
CANCEL_S = 30.0


class ImpactOnly:
    def __init__(self, threshold: float = IMPACT_ALERT, cancel_s: float = CANCEL_S):
        self.threshold, self.cancel_s = threshold, cancel_s
        self.state, self.since = "IDLE", None
        self.events = []

    def step(self, records) -> str:
        m = next((r for r in records if r["stream"] == "motion"), None)
        t = float(records[0]["t"])
        s = float(m["score"]) if m else 0.0
        if self.state == "IDLE" and s >= self.threshold:
            self.state, self.since = "ALERT", t
            self.events.append(dict(t=round(t, 2), event="alert", score=round(s, 3)))
        elif self.state == "ALERT" and t - self.since >= self.cancel_s:
            self.state = "CAREGIVER"
            self.events.append(dict(t=round(t, 2), event="caregiver alerted", score=round(s, 3)))
        return self.state

    def outcome(self) -> str:
        ev = [e["event"] for e in self.events]
        return " -> ".join("caregiver" if e.startswith("caregiver") else e for e in ev) or "nothing"


def replay(records_by_window, **kw):
    b = ImpactOnly(**kw)
    for recs in records_by_window:
        b.step(recs)
    return b

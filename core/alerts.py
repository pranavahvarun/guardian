"""
core/alerts.py -- alert manager (Stage 7.2). Turns the fusion score into what
the wearer and the caregiver see.

States:
    IDLE        nothing going on
    CHECK_IN    score 0.4 to 0.7: band asks "Are you OK?"
    ALERT       score 0.7 or more: band says "press to cancel", 30 s window
    CAREGIVER   nobody cancelled: caregiver alerted
    URGENT      caregiver alert escalated: long lie past 5 min, or a
                check-in nobody answered

Rules (starting values, frozen with the fusion weights):
    - CHECK_IN or ALERT clear by themselves if the score falls below 0.4
      (the person moved again, heart rate came back). Event "cleared".
    - The wearer can answer a check-in or cancel an alert (answer_ok / cancel).
    - A check-in not answered within CHECKIN_ANSWER_S -> URGENT.
      v3 (team decision after the F5 nap case): for the UNRESPONSIVE type only,
      an unanswered check-in first waits SECOND_CHECKIN_GAP_S (5 min), then
      the band asks again; urgent only if that is unanswered too. A nap no
      longer pages the caregiver; a truly unresponsive person is reported
      about 7 min later than before. Heart-rate and faint check-ins still go
      urgent after 2 min.
    - ALERT not cancelled within CANCEL_S -> CAREGIVER.
    - CAREGIVER -> URGENT when the person has not moved for URGENT_STILL_S
      (v2: checked on the long-lie score, which v1 could lose to a tie).
    - Once the caregiver is told, the alert stays open until a caregiver or
      the admin resolves it (Stage 8); a recovery is logged, not auto-closed.
    - Every event stores the type, score and evidence at that moment.

    py -m core.alerts hard_fall_long_lie       replay a saved scenario
    py -m core.alerts all                      all 7 saved scenarios
    py -m core.alerts hard_fall_long_lie --trace   also every window's evidence
    py -m core.alerts T1 --trace               a held-out test, from its saved outputs
                                               (reading them changes nothing)
(reads testing/outputs/stage5/<name>_outputs.csv and _truth.json)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.fusion import Fusion  # noqa: E402

CHECK_IN, ALERT = 0.4, 0.7
CANCEL_S = 30.0
CHECKIN_ANSWER_S = 120.0
URGENT_STILL_S = 300.0
SECOND_CHECKIN_GAP_S = 300.0     # v3 (team decision, 9 Oct): unresponsive only
TEST_IDS = {f"T{i}" for i in range(1, 8)}      # held-out, saved by fusion_eval test
LABEL = {"hard_fall": "Hard fall", "long_lie": "Long lie", "faint": "Faint",
         "heart_rate": "Heart rate", "unresponsive": "Unresponsive"}


class AlertManager:
    def __init__(self):
        self.state, self.since, self.type = "IDLE", None, None
        self.events = []
        self.second_due, self.second_sent = None, False

    def _go(self, state, f, what):
        if state != "CHECK_IN":
            self.second_due, self.second_sent = None, False
        self.state, self.since = state, f["t"]
        self.events.append(dict(t=round(f["t"], 2), event=what, state=state,
                                type=f["type"], score=round(f["score"], 3),
                                evidence=f["evidence"]))

    # wearer actions from the band
    def cancel(self, f):
        if self.state == "ALERT":
            self._go("IDLE", f, "cancelled by wearer")

    def answer_ok(self, f):
        if self.state == "CHECK_IN":
            self._go("IDLE", f, "wearer answered OK")

    def step(self, f: dict) -> str:
        s, t, st = f["score"], f["t"], self.state
        if st == "IDLE":
            if s >= ALERT:
                self.type = f["type"]
                self._go("ALERT", f, "alert")
            elif s >= CHECK_IN:
                self.type = f["type"]
                self._go("CHECK_IN", f, "check-in")
        elif st == "CHECK_IN":
            if s >= ALERT:
                self._go("ALERT", f, "alert")
            elif s < CHECK_IN:
                self._go("IDLE", f, "cleared")
            elif t - self.since >= CHECKIN_ANSWER_S:
                if self.type == "unresponsive" and not self.second_sent:
                    if self.second_due is None:           # v3: wait, then ask again
                        self.second_due = t + SECOND_CHECKIN_GAP_S
                        self.events.append(dict(t=round(t, 2), event="check-in unanswered",
                                                state="CHECK_IN", type=f["type"],
                                                score=round(s, 3), evidence=f["evidence"]))
                    elif t >= self.second_due:
                        self._go("CHECK_IN", f, "second check-in")
                        self.second_sent = True
                else:
                    self._go("URGENT", f, "urgent: check-in not answered")
        elif st == "ALERT":
            if s < CHECK_IN:
                self._go("IDLE", f, "cleared")
            elif t - self.since >= CANCEL_S:
                self._go("CAREGIVER", f, "caregiver alerted")
        elif st == "CAREGIVER":
            # v2: by the long-lie score, not the leading type (hard fall can tie it)
            if f["scores"]["long_lie"] >= ALERT and f["evidence"]["still_s"] >= URGENT_STILL_S:
                self._go("URGENT", f, "urgent: long lie")
            elif s < CHECK_IN and not any(e["event"] == "recovered" for e in self.events
                                          if e["t"] >= self.since):
                self.events.append(dict(t=round(t, 2), event="recovered", state=st,
                                        type=f["type"], score=round(s, 3),
                                        evidence=f["evidence"]))
        if self.state != "IDLE" and f["type"] != self.type and s >= CHECK_IN:
            self.type = f["type"]                     # reason updates (e.g. now long lie)
        return self.state

    def outcome(self) -> str:
        ev = [e["event"] for e in self.events if e["event"] != "recovered"]
        if not ev:
            return "nothing"
        out = []
        for e in ev:
            short = {"check-in": "check-in", "alert": "alert", "cleared": "cleared",
                     "caregiver alerted": "caregiver"}.get(e, e.split(":")[0])
            if not out or out[-1] != short:
                out.append(short)
        return " -> ".join(out)


STAGE = {"CHECK_IN": "check_in", "ALERT": "alert", "CAREGIVER": "caregiver", "URGENT": "urgent"}


def fusion_record(f, state):
    """The fusion result as a pipeline-style record (stream 'fusion'), so it is
    stored next to the module outputs and the live screen can read it."""
    return dict(t=f["t"], stream="fusion", score=float(f["score"]), quality=1.0,
                extras=dict(type=f["type"], label=LABEL[f["type"]], state=state,
                            scores={k: round(float(v), 3) for k, v in f["scores"].items()},
                            evidence=f["evidence"]))


def replay(records_by_window, clock=None):
    """records_by_window: list of record lists (one per window).
    Returns (fusion outputs, AlertManager)."""
    fu, am, rows = Fusion(clock=clock), AlertManager(), []
    for recs in records_by_window:
        f = fu.step(recs)
        f["state"] = am.step(f)
        rows.append(f)
    return rows, am


def load_saved(name, folder=None):
    import pandas as pd
    folder = Path(folder or ROOT / "testing" / "outputs" / "stage5")
    d = pd.read_csv(folder / f"{name}_outputs.csv")
    d["extras"] = d.extras.apply(json.loads)
    truth = json.loads((folder / f"{name}_truth.json").read_text())
    wins = [g.to_dict("records") for _, g in d.groupby("t", sort=True)]
    return wins, truth


def _trace(rows):
    print(f"  {'t':>6s} {'state':9s} {'lead':12s} {'score':>5s} {'fall':>5s} {'lie':>5s} "
          f"{'faint':>5s} {'hr':>5s} | {'imp':>4s} {'qm':>4s} {'still':>5s} {'mvd_s':>5s} "
          f"{'rec':3s} {'since':>5s} {'activity':16s} {'walk':4s} {'bpm':>4s} {'qp':>4s} "
          f"{'low':>4s} {'high':>4s}")
    for r in rows:
        e, S = r["evidence"], r["scores"]
        print(f"  {r['t']:6.1f} {r['state']:9s} {LABEL[r['type']]:12s} {r['score']:5.2f} "
              f"{S['hard_fall']:5.2f} {S['long_lie']:5.2f} {S['faint']:5.2f} {S['heart_rate']:5.2f} | "
              f"{e['impact']:4.2f} {e['q_motion']:4.2f} {e['still_s']:5.0f} {e['moved_s']:5.1f} "
              f"{'Y' if e['moved_again'] else '-':3s} "
              f"{'-' if e['since_impact'] is None else round(e['since_impact']):>5} "
              f"{str(e['activity'])[:16]:16s} {'Y' if e['walking'] else '-':4s} "
              f"{'-' if e['hr'] is None else round(e['hr']):>4} {e['q_physio']:4.2f} "
              f"{e['hr_low_s']:4.0f} {e['hr_high_s']:4.0f}")


def _print(name, rows, am, truth):
    print(f"\n=== {name} (SIMULATED, clock {truth.get('clock') or 'daytime'}) ===")
    for e in truth.get("events", []):
        print(f"truth: {e['what']} at {e['t']:.1f} s")
    top = max(rows, key=lambda r: r["score"])
    print(f"peak score {top['score']:.2f} ({LABEL[top['type']]}) at {top['t']:.1f} s")
    for e in am.events:
        ev = e["evidence"]
        print(f"  {e['t']:6.1f} s  {e['event']:30s} {LABEL[e['type']]:12s} score {e['score']:.2f}"
              f"  impact {ev['impact']:.2f} still {ev['still_s']:.0f}s moved {ev['moved_again']}"
              f" hr {ev['hr'] if ev['hr'] is None else round(ev['hr'])}"
              f" low {ev['hr_low_s']:.0f}s high {ev['hr_high_s']:.0f}s walking {ev['walking']}")
    print(f"OUTCOME {name}: {am.outcome()}")


def main(argv=None):
    from simulator.scenarios import SCENARIOS
    args = list(argv if argv is not None else sys.argv[1:])
    trace = "--trace" in args
    args = [a for a in args if a != "--trace"]
    names = list(SCENARIOS) if not args or args[0] == "all" else args
    for n in names:
        folder = ROOT / "testing" / "outputs" / "stage7" / "tests" if n in TEST_IDS else None
        wins, truth = load_saved(n, folder)
        rows, am = replay(wins, truth.get("clock"))
        _print(n, rows, am, truth)
        if trace:
            _trace(rows)


if __name__ == "__main__":
    main()

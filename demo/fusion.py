"""
demo/fusion.py -- review output 4: how fusion turns the three modules into
one emergency score. Replays SAVED module outputs of four simulated cases
through the final fusion (v3) and prints the key moments, with each score
worked out. Then the test results and one failure found by testing.

    py -m demo.fusion            everything
    py -m demo.fusion 1          one case (1-4)
    py -m demo.fusion results    test results only
    py -m demo.fusion found      the failure found by testing only

Reads testing/outputs/stage5/<case>_outputs.csv (py -m simulator.run all),
testing/outputs/stage7/fusion_tune.txt, fusion_test.txt and failures/F4_*.
"""

from __future__ import annotations

import sys

from core import fusion as F
from core.alerts import LABEL, load_saved, replay
from demo.common import OUT, W, need, rows, section, title

CASES = [
    ("hard_fall_long_lie", "Hard fall, then lying still",
     "walks, slips forward, then lies still for 60 s (daytime)"),
    ("fall_then_get_up", "Fall, then the person gets up",
     "walks, trips, 8 s on the floor, then sits, stands and walks away"),
    ("faint_collapse", "Faint from a chair",
     "sitting, heart rate drops to 52, falls backward, 60 s still"),
    ("night_sleep", "Going to bed at night",
     "23:30: walks, sits on the bed, lies still for 2 min, calm heart rate"),
]
STEP = {"IDLE": "-", "CHECK_IN": "CHECK-IN", "ALERT": "ALERT",
        "CAREGIVER": "CAREGIVER", "URGENT": "URGENT"}
EVENT = {"check-in": "CHECK-IN on the band: \"Are you OK?\"",
         "alert": "ALERT: 30 s for the wearer to cancel",
         "caregiver alerted": "CAREGIVER TOLD (nobody cancelled)",
         "cleared": "CLEARED: the person got up",
         "urgent: long lie": "URGENT: 5 min on the floor"}


def _q(x):
    return "" if x >= 0.995 else f" x q{x:.2f}"


def formula(k, f):
    """The score of type k, written out with this window's numbers."""
    e, S, Wt = f["evidence"], f["scores"], F.WEIGHTS
    imp, still, qm, qp = e["impact"], e["still_s"], e["q_motion"], e["q_physio"]
    got_up = e["moved_again"]
    if k == "hard_fall":
        if got_up and imp == 0:
            return "hard fall = 0 (got up: event closed)"
        walk_after = e["walking"] and (e["since_impact"] or 0) >= F.SMOOTH_S
        return (f"hard fall = impact {imp:.2f}{_q(qm)}"
                + (" - 0.2 walking" if walk_after else "") + f" = {S[k]:.2f}")
    if k == "long_lie":
        return f"long lie = still {min(still, 60):.0f}/60{_q(qm)} = {S[k]:.2f}"
    if k == "faint":
        a, b = min(1, still / 30), min(1, e["hr_low_s"] / 30)
        txt = f"faint = 0.5 x still {a:.2f}"
        if b > 0:
            txt += f" + 0.4 x HR-low {b:.2f}"
        if imp > 0:
            txt += f" + 0.3 x impact {imp:.2f}"
        return txt + (" (x quality)" if min(qm, qp) < 0.995 else "") + f" = {S[k]:.2f}"
    if k == "heart_rate":
        out = max(e["hr_low_s"], e["hr_high_s"])
        txt = f"heart rate = {out:.0f} s out of range/60{_q(qp)}"
        return txt + (" - walking" if e["walking"] and e["hr_high_s"] > 0 else "") + f" = {S[k]:.2f}"
    w = Wt["unresponsive"]["still"]
    txt = f"unresponsive = {w} x still {still:.0f}/1800"
    return txt + (" - night 1.0" if e["night"] else "") + f" = {S[k]:.2f}"


def modules(f):
    e = f["evidence"]
    imp = max(e["impact"], e["impact_now"])
    if imp < 0.2:
        mot = "no impact"
    elif e["impact"] > e["impact_now"] + 0.05:
        mot = f"impact {e['impact']:.2f} (held)"
    else:
        mot = f"impact {imp:.2f}"
    mot += ", moving" if e["still_s"] == 0 else f", still {e['still_s']:.0f} s"
    if e["moved_again"] and imp >= 0.2:
        mot += ", got up"
    act = str(e["activity"] or "-").replace("SITTING_STANDING", "SIT/STAND")
    hr = "warming up" if e["hr"] is None else f"{e['hr']:.0f} bpm"
    if e["hr_low_s"] > 0:
        hr += f" (low {e['hr_low_s']:.0f} s)"
    elif e["hr_high_s"] > 0:
        hr += f" (high {e['hr_high_s']:.0f} s)"
    return f"impact: {mot:30s} activity: {act:11s} heart: {hr}"


def moments(rows_, am):
    """Times worth showing: before anything, each alert step, when the lead
    type changes, and the end."""
    ev = [e for e in am.events if e["event"] != "recovered"]
    pick = set()
    first = ev[0]["t"] if ev else None
    before = [r for r in rows_ if r["evidence"]["hr"] is not None and (first is None or r["t"] < first)]
    if before:
        pick.add(before[-1]["t"])
    for e in ev:
        pick.add(e["t"])
    lead = None
    for r in rows_:
        if r["score"] >= 0.4 and r["type"] != lead and lead is not None:
            pick.add(r["t"])
        if r["score"] >= 0.4:
            lead = r["type"]
    best_faint = max(rows_, key=lambda r: r["scores"]["faint"])
    if best_faint["scores"]["faint"] >= 0.4 and best_faint["evidence"]["hr_low_s"] > 0:
        pick.add(best_faint["t"])
    pick.add(rows_[-1]["t"])
    if not ev:                                     # a quiet case: start, long still part, end
        known = [r for r in rows_ if r["evidence"]["hr"] is not None]
        if known:
            pick.add(known[0]["t"])
        still = [r for r in rows_ if r["evidence"]["still_s"] >= 60]
        if still:
            pick.add(still[0]["t"])
    return [r for r in rows_ if r["t"] in pick]


def show_case(i):
    name, head_, story = CASES[i - 1]
    need(OUT / "stage5" / f"{name}_outputs.csv", "py -m simulator.run all")
    wins, truth = load_saved(name)
    rows_, am = replay(wins, truth.get("clock"))
    events = {e["t"]: e for e in am.events if e["event"] != "recovered"}
    t0 = truth["events"][0]["t"] if truth.get("events") else None

    section(f"CASE {i}. {head_.upper()}   ({name}, simulated)")
    rows([f"What happens: {story}" + (f"; impact at {t0:.1f} s" if t0 else "")])
    print()
    for r in moments(rows_, am):
        rows([f"{r['t']:6.1f} s  {modules(r)}"])
        others = [k for k in F.TYPES if k != r["type"] and r["scores"][k] >= 0.3]
        line = formula(r["type"], r)
        if r["score"] < 0.005 and not others:
            if r["evidence"]["moved_again"] and r["evidence"]["impact_now"] >= 0.2:
                line = ("got up confirmed (moving 12 s, mostly walking or sit/stand): "
                        "fall closed, all types 0")
            elif r["evidence"]["night"]:
                line = formula("unresponsive", r) + "   (no impact, heart rate normal)"
            else:
                line = "all five types 0: no impact, heart rate normal"
        rows([f"{'':10s}{line}"])
        for k in others[:2]:
            rows([f"{'':10s}{formula(k, r)}"])
        if r["t"] in events:
            ev = events[r["t"]]["event"]
            step = EVENT.get(ev, ev)
            if ev == "cleared" and not r["evidence"]["moved_again"]:
                step = "CLEARED: score back below 0.4"
        else:
            step = STEP[r["state"]]
        rows([f"{'':10s}emergency score {r['score']:.2f}  ({LABEL[r['type']] if r['score'] > 0 else '-'})"
              f"  ->  {step}"])
        print()
    a = next((e for e in am.events if e["event"] == "alert"), None)
    c = next((e for e in am.events if e["event"].startswith("caregiver")), None)
    cl = next((e for e in am.events if e["event"] == "cleared"), None)
    res = am.outcome()
    if a and t0 is not None:
        res += f"   (alert {a['t'] - t0:.1f} s after the impact"
        res += f", caregiver {c['t'] - t0:.1f} s after)" if c else (
            f", cleared {cl['t'] - t0:.1f} s after)" if cl else ")")
    rows([f"Result: {res}"])


def show_results():
    section("RESULTS   expected outcomes locked before any run")
    tune = need(OUT / "stage7" / "fusion_tune.txt", "py -m testing.fusion_eval tune").read_text()
    test = need(OUT / "stage7" / "fusion_test.txt", "py -m testing.fusion_eval test").read_text()

    def res(txt):
        line = next((x for x in txt.splitlines() if x.startswith("RESULT")), "")
        return line.split(":", 1)[-1].strip()
    rows([f"{'Tuning: 7 scenarios, weights set here, then frozen':58s}{res(tune)}",
          f"{'Held-out test: 7 new scenarios, new people, run once':58s}{res(test)}", ""])
    rows([f"  {'test':30s}{'expected':18s}{'got':20s}"])
    import re
    pat = re.compile(r"^\s{2}(T\d .*?)\s+(caregiver|alert|nothing|check-in/alert|check-in)"
                     r"\s+(.+?)\s+(PASS|FAIL)")
    for x in test.splitlines():
        m = pat.match(x)
        if m:
            tid, exp, got, ok = m.groups()
            rows([f"  {tid[:30]:30s}{exp:18s}{got:20s}{'ok' if ok == 'PASS' else 'MISSED'}"])


def show_found():
    from core.v1.alerts import replay as replay_v1
    folder = OUT / "stage7" / "failures"
    need(folder / "F4_outputs.csv", "py -m testing.fusion_failures --only F4")
    wins, truth = load_saved("F4", folder)
    t0 = truth["events"][0]["t"]
    _, a1 = replay_v1(wins, truth.get("clock"))
    _, a3 = replay(wins, truth.get("clock"))

    def told(am):
        c = next((e for e in am.events if e["event"].startswith("caregiver")), None)
        return f"caregiver told {c['t'] - t0:.0f} s after the fall" if c else "caregiver never told"
    section("FOUND BY TESTING   a fall where the person cannot get up but keeps moving the arms")
    rows([f"{'first version (tested):':26s}{a1.outcome():34s}{told(a1)}",
          f"{'final version:':26s}{a3.outcome():34s}{told(a3)}"])


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    title("4. EMERGENCY RISK ESTIMATION MODULE",
          "Emergency score from impact, activity and heart rate, every 1.24 s")
    if not args or args[0] in "1234":
        for i in ([int(args[0])] if args else range(1, 5)):
            show_case(i)
    if not args or args[0] == "results":
        show_results()
    if not args or args[0] == "found":
        show_found()
    print()
    print("=" * W)


if __name__ == "__main__":
    main()

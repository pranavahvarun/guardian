"""
testing/fusion_failures.py -- where does fusion fail? (Stage 7.6)

Runs on the FROZEN settings (0a3011d7b96be571); changes nothing. Three parts:

Every case is run through v1 (frozen, tested) and v2 (after-test fixes).

A  Slow alerts in the held-out test: for T1 and T4, what the motion impact
   score was and which type finally raised the alert.
B  New failure scenarios F1-F5 (simulator/failure_scenarios.py), built and
   run through all streams.
C  Broken or missing streams, made by editing SAVED outputs before fusion:
     I1  T7 (walking, HR 115) with the activity stream missing
     I2  hard_fall_long_lie with no heart rate (no PPG)
     I3  T4 (faint, HR 48) with no heart rate
     I4  hr_spike_no_fall with heart-rate quality 0.45 (just below the 0.5 gate)

Every case states what a good system should do and our prediction, both
written before the first run.

    py -m testing.fusion_failures            all three parts
    py -m testing.fusion_failures --skip-f   A and C only (F5 alone takes a few minutes)
    py -m testing.fusion_failures --only F6 F7

Writes testing/outputs/stage7/fusion_failures.txt, outputs in .../failures/
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.alerts import LABEL, load_saved, replay  # noqa: E402
from core.v1.alerts import replay as replay_v1  # noqa: E402
from core.fusion import VERSION as VER  # noqa: E402
from testing.fusion_eval import level, settings  # noqa: E402

OUT = ROOT / "testing" / "outputs" / "stage7"
STAGE5 = ROOT / "testing" / "outputs" / "stage5"
TESTS = OUT / "tests"

INJECT = {
    "I1": dict(src="T7", folder=TESTS, what="T7 walking HR 115, activity missing",
               good="nothing", predict="heart-rate alert (false)"),
    "I2": dict(src="hard_fall_long_lie", folder=STAGE5, what="hard fall + long lie, no PPG",
               good="caregiver", predict="caregiver"),
    "I3": dict(src="T4", folder=TESTS, what="T4 faint HR 48, no PPG",
               good="caregiver", predict="caregiver, later"),
    "I4": dict(src="hr_spike_no_fall", folder=STAGE5, what="HR spike, HR quality 0.45",
               good="check-in", predict="nothing (HR never trusted)"),
}


def _inject(wins, iid):
    wins = copy.deepcopy(wins)
    for w in wins:
        for r in w:
            if iid == "I1" and r["stream"] == "activity":
                r["quality"], r["extras"] = 0.0, {"state": "NO_DATA"}
            if iid in ("I2", "I3") and r["stream"] == "physiological":
                r["score"], r["quality"], r["extras"] = 0.0, 0.0, {"state": "NO_PPG", "hr": None}
            if iid == "I4" and r["stream"] == "physiological" and r["quality"] > 0:
                r["quality"] = 0.45
    return wins


def _first(am, name):
    for e in am.events:
        if e["event"].startswith(name):
            return e
    return None


def _brief(am, t0):
    a, c = _first(am, "alert"), _first(am, "caregiver")
    u = _first(am, "urgent")

    def at(e):
        if e is None:
            return "-"
        return f"{e['t']:.0f}s" + (f"(+{e['t'] - t0:.0f})" if t0 is not None else "")
    return f"{level(am) + (' URGENT' if u else ''):17s} {at(a):11s} {at(c):11s}", am.outcome()


def _row(cid, what, good, predict, wins, clock, t0):
    """v1 = frozen and tested; vN = current version (after-test fixes, Stage 7.6)."""
    _, am1 = replay_v1(wins, clock)
    _, am2 = replay(wins, clock)
    b1, p1 = _brief(am1, t0)
    b2, p2 = _brief(am2, t0)
    return [f"  {cid:3s} {what:36s} {good:18s} {predict:26s} v1 {b1} {p1}",
            f"  {'':3s} {'':36s} {'':18s} {'':26s} v{VER} {b2} {p2}"]


HEAD = (f"  {'id':3s} {'case':36s} {'good system':18s} {'we predicted':26s}    "
        f"{'got':17s} {'alert':11s} {'caregiver':11s} path")


def part_a(L):
    L += ["A  SLOW ALERTS IN THE HELD-OUT TEST (frozen settings, saved outputs)"]
    for tid in ("T1", "T4"):
        try:
            wins, truth = load_saved(tid, TESTS)
        except FileNotFoundError:
            L.append(f"  {tid}: no saved outputs (run py -m testing.fusion_eval test first)")
            continue
        rows, am = replay(wins, truth.get("clock"))
        t0 = truth["events"][0]["t"] if truth.get("events") else None
        peak = max((r for w in wins for r in w if r["stream"] == "motion"),
                   key=lambda r: r["score"])
        a = _first(am, "alert")
        L.append(f"  {tid}: impact {t0:.1f} s; motion peak impact {peak['score']:.2f} at "
                 f"{peak['t']:.1f} s; alert at {a['t']:.1f} s raised by {LABEL[a['type']]}"
                 f" (still {a['evidence']['still_s']:.0f} s, HR {a['evidence']['hr']})"
                 if a else f"  {tid}: no alert")
    L.append("")


def part_b(L, only=None):
    from simulator.failure_scenarios import FAILURES, build_failure
    from simulator.run import run_scenario
    import pandas as pd
    d = OUT / "failures"
    d.mkdir(parents=True, exist_ok=True)
    L += ["B  FAILURE SCENARIOS (SIMULATED)", HEAD]
    for fid, spec in FAILURES.items():
        if only and fid not in only:
            continue
        print(f"  running {fid} ...", flush=True)
        sc, wins = run_scenario(build_failure(fid))
        flat = pd.DataFrame([r for w in wins for r in w])
        flat.assign(extras=flat.extras.apply(json.dumps)).to_csv(d / f"{fid}_outputs.csv",
                                                                 index=False)
        (d / f"{fid}_truth.json").write_text(json.dumps(dict(
            name=fid, title=sc.title, description=sc.description, simulated=True,
            seconds=round(sc.seconds, 1), clock=sc.clock, truth=sc.truth,
            events=sc.events), indent=2))
        t0 = sc.events[0]["t"] if sc.events else None
        L += _row(fid, spec["short"], spec["good"], spec["predict"], wins, sc.clock, t0)
        for sw in getattr(sc, "swaps", []):
            L.append(f"      substituted {sw}")
    L.append("")


def part_c(L):
    L += ["C  MISSING OR POOR STREAMS (saved outputs edited before fusion)", HEAD]
    for iid, spec in INJECT.items():
        try:
            wins, truth = load_saved(spec["src"], spec["folder"])
        except FileNotFoundError:
            L.append(f"  {iid}: no saved outputs for {spec['src']}")
            continue
        t0 = truth["events"][0]["t"] if truth.get("events") else None
        L += _row(iid, spec["what"], spec["good"], spec["predict"], _inject(wins, iid),
                  truth.get("clock"), t0)
    L.append("")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-f", action="store_true")
    ap.add_argument("--only", nargs="*", help="only these F cases, e.g. --only F6 F7")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    s = settings()
    L = [f"Fusion failure study (Stage 7.6). v1 = frozen 0a3011d7b96be571 (held-out tested); "
         f"v{VER} = after-test fixes, settings {s['fingerprint']}",
         "Simulated cases. 'good system' and 'we predicted' were written before running "
         "(F6, F7 together with the v2 fixes).", ""]
    part_a(L)
    if not a.skip_f:
        part_b(L, a.only)
    part_c(L)
    (OUT / "fusion_failures.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()

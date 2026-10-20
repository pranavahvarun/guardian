"""
testing/compare_impact_only.py -- Emergency Risk Estimation Module (fusion v3
+ Alert Escalation Module, frozen settings, unchanged) against an IMPACT-ONLY
detector (core/baseline.py) on the same simulated cases (20 Oct 2026).

    py -m testing.compare_impact_only            all 12 cases (C9, C10 take a few minutes)
    py -m testing.compare_impact_only --only C1 C2    some cases (labelled partial)
    py -m testing.compare_impact_only --again    re-run after a first full run (labelled)

Cases and expected outcomes: simulator/compare_scenarios.py, locked before
the first run. Both systems read the SAME module outputs, window by window
(one pipeline run per case). Nobody presses cancel or answers a check-in in
the simulations, for either system.

Writes
    testing/outputs/compare/compare_first_run.txt   (or compare_rerun.txt / _partial.txt)
    testing/outputs/compare/C*_outputs.csv, C*_truth.json
Shown for the review by: py -m demo.compare

Every case is reported, including any where impact-only does better. These
are SIMULATED cases (real wrist motion, synthetic stillness and heart rate):
they show how the two rule sets behave, not a detection rate.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core import baseline  # noqa: E402
from core.alerts import replay  # noqa: E402
from testing.fusion_eval import settings  # noqa: E402

OUT = ROOT / "testing" / "outputs" / "compare"


def fingerprint():
    from simulator import compare_scenarios as cs
    src = inspect.getsource(cs)
    return hashlib.sha256(src.encode()).hexdigest()[:16]


def fusion_level(am):
    ev = [e["event"] for e in am.events]
    if any(e.startswith("urgent") for e in ev):
        return "urgent"
    if any(e.startswith("caregiver") for e in ev):
        return "caregiver"
    if "alert" in ev:
        return "alert"
    if "check-in" in ev or "second check-in" in ev:
        return "check-in"
    return "nothing"


def base_level(b):
    ev = [e["event"] for e in b.events]
    if any(e.startswith("caregiver") for e in ev):
        return "caregiver"
    return "alert" if ev else "nothing"


def passes(expected, got):
    told = got in ("caregiver", "urgent")
    return {"caregiver": told, "urgent": got == "urgent", "no caregiver": not told,
            "no alert": got in ("nothing", "check-in")}[expected]


def first(events, prefix):
    return next((e["t"] for e in events if e["event"].startswith(prefix)), None)


def told_at(events):
    ts = [e["t"] for e in events if e["event"].startswith(("caregiver", "urgent"))]
    return min(ts) if ts else None


def rel(t, t0):
    if t is None:
        return "-"
    return f"+{t - t0:.0f} s" if t0 is not None else f"{t:.0f} s"


def main(argv=None):
    from simulator.compare_scenarios import CASES, LOCKED, build_case
    from simulator.run import run_scenario
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--again", action="store_true")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    done = OUT / "compare_first_run.txt"
    ids = [c for c in CASES if not a.only or c in a.only]
    full = len(ids) == len(CASES)
    if full and done.exists() and not a.again:
        print(done.read_text())
        sys.exit("Already run once (above). Use --again to re-run; it is labelled a re-run.")

    built = {}
    for cid in ids:                       # build all first: a missing recording stops everything
        try:
            built[cid] = build_case(cid)
        except LookupError as e:
            sys.exit(f"{cid} cannot be built: {e}\nNothing was run.")

    s, fp = settings(), fingerprint()
    res = []
    for cid in ids:
        print(f"  running {cid} ({CASES[cid]['short']}) ...", flush=True)
        sc, wins = run_scenario(built[cid])
        flat = pd.DataFrame([r for w in wins for r in w])
        flat.assign(extras=flat.extras.apply(json.dumps)).to_csv(OUT / f"{cid}_outputs.csv",
                                                                 index=False)
        (OUT / f"{cid}_truth.json").write_text(json.dumps(dict(
            name=cid, title=sc.title, description=sc.description, simulated=True,
            seconds=round(sc.seconds, 1), clock=sc.clock, truth=sc.truth,
            events=sc.events, ppg=sc.ppg is not None), indent=2))
        t0 = sc.events[0]["t"] if sc.events else None
        _, am = replay(wins, sc.clock)
        bo = baseline.replay(wins)
        exp = CASES[cid]["expected"]
        fg, bg = fusion_level(am), base_level(bo)
        motion_peak = max((r["score"] for w in wins for r in w if r["stream"] == "motion"),
                          default=0.0)
        res.append(dict(
            id=cid, short=CASES[cid]["short"], expected=exp,
            provisional=CASES[cid].get("provisional", False),
            known_gap=CASES[cid].get("known_gap", False), peak=motion_peak,
            b_got=bg, b_ok=passes(exp, bg), b_alert=rel(first(bo.events, "alert"), t0),
            b_told=rel(told_at(bo.events), t0),
            f_got=fg, f_ok=passes(exp, fg), f_alert=rel(first(am.events, "alert"), t0),
            f_told=rel(told_at(am.events), t0), f_path=am.outcome(),
            swaps=getattr(sc, "swaps", [])))

    label = ("first run" if full and not done.exists() else
             "RE-RUN" if full else f"PARTIAL run ({', '.join(ids)})")
    L = ["COMPARISON: Emergency Risk Estimation Module vs an impact-only detector  "
         f"({label}, SIMULATED cases)",
         f"cases locked {LOCKED}; case-file fingerprint {fp}",
         f"fusion v{s.get('version', 1)} + Alert Escalation Module, frozen settings "
         f"{s['fingerprint']} (unchanged)",
         "impact-only: alert when the impact score reaches 0.7, caregiver 30 s later "
         "(no cancel pressed, for either system)",
         "times are after the event (fall impact, or when the change starts)", "",
         f"  {'id':4s} {'situation':29s} {'expected':13s}| {'impact-only':12s} {'caregiver':10s} "
         f"{'ok':3s} | {'Guardian':10s} {'alert':9s} {'caregiver':10s} {'ok':3s}"]
    L.append("  " + "-" * (len(L[-1]) - 2))
    for r in res:
        mark = "*" if r["provisional"] else ("+" if r["known_gap"] else "")
        L.append(f"  {r['id']:4s} {r['short'][:29]:29s} {(r['expected'] + mark)[:13]:13s}| "
                 f"{r['b_got']:12s} {r['b_told']:10s} {'yes' if r['b_ok'] else 'NO':3s} | "
                 f"{r['f_got']:10s} {r['f_alert']:9s} {r['f_told']:10s} {'yes' if r['f_ok'] else 'NO':3s}")
    nb, nf = sum(r["b_ok"] for r in res), sum(r["f_ok"] for r in res)
    both = [r for r in res if r["b_told"] != "-" and r["f_told"] != "-"
            and r["expected"] in ("caregiver", "urgent")]
    L += ["",
          f"  as expected: impact-only {nb} of {len(res)}, Guardian {nf} of {len(res)}",
          "  * provisional: heart-rate thresholds 60/100 bpm not yet confirmed with sources",
          "  + known gap, written down before the run: no impact and normal heart rate",
          "",
          "  where impact-only did better:"]
    better = [r for r in res if r["b_ok"] and not r["f_ok"]]
    for r in better:
        L.append(f"    {r['id']} {r['short']}: impact-only {r['b_got']}, Guardian {r['f_got']}")
    faster = [r for r in both if float(r["b_told"].split()[0]) < float(r["f_told"].split()[0])]
    for r in faster:
        L.append(f"    {r['id']} caregiver told sooner by impact-only ({r['b_told']} vs {r['f_told']})")
    if not better and not faster:
        L.append("    none")
    L += ["", "  Guardian alert path per case:"]
    for r in res:
        L.append(f"    {r['id']:4s} {r['f_path']}   (peak impact score {r['peak']:.2f})")
    for r in res:
        for sw in r["swaps"]:
            L.append(f"  {r['id']} substituted {sw}")
    name = ("compare_first_run.txt" if label == "first run" else
            "compare_rerun.txt" if full else "compare_partial.txt")
    (OUT / name).write_text("\n".join(L) + "\n")
    (OUT / name.replace(".txt", ".json")).write_text(json.dumps(res, indent=2))
    print("\n".join(L))
    print(f"\n  saved {(OUT / name).relative_to(ROOT)}")


if __name__ == "__main__":
    main()

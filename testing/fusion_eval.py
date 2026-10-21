"""
testing/fusion_eval.py -- checks fusion + alerts against the expected results
locked on 8 Oct 2026 (Stage 7.3 / 7.4).

    py -m testing.fusion_eval tune        the 7 tuning scenarios (saved outputs)
    py -m testing.fusion_eval freeze      tune, and if all 7 pass, freeze the
                                          fusion and alert settings
    py -m testing.fusion_eval freeze --after-test "reason"
                                          freeze a later version (v2+), labelled
                                          as made after the held-out test
    py -m testing.fusion_eval test        held-out T1-T7, once (Stage 7.4);
                                          outputs in testing/outputs/stage7/tests

tune reads testing/outputs/stage5/<name>_outputs.csv and _truth.json, so run
py -m simulator.run all first whenever a module or a scenario changes.

freeze writes testing/outputs/stage7/fusion_frozen.json: every weight and
threshold plus a fingerprint. test refuses to run if the settings no longer
match the frozen ones, and runs only once (a second run needs --again and is
labelled as a re-run).

Outcome levels, from the alert manager's events:
    nothing      no check-in, no alert
    check-in     check-in only, never an alert, caregiver not told
    alert        an alert, caregiver not told
For fall_then_get_up, T2 and T5 the last event must also be "cleared"
(locked: "alert, then cleared, no caregiver").
    caregiver    caregiver told (or urgent)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core import alerts, fusion  # noqa: E402
from core.alerts import load_saved, replay  # noqa: E402

OUT = ROOT / "testing" / "outputs" / "stage7"
FROZEN = OUT / "fusion_frozen.json"          # v1, frozen 8 Oct 2026, never overwritten


def frozen_path(v):
    return FROZEN if v == 1 else OUT / f"fusion_frozen_v{v}.json"

# locked 8 Oct 2026 ("lock"); never edit after seeing results
TUNE = {
    "normal_day": ("nothing",),
    "hard_fall_long_lie": ("caregiver",),
    "fall_then_get_up": ("alert",),
    "faint_collapse": ("caregiver",),
    "false_alarm": ("nothing",),
    "hr_spike_no_fall": ("check-in",),           # provisional: HR thresholds
    "night_sleep": ("nothing",),
}
TEST = {
    "T1": ("caregiver",),
    "T2": ("alert",),
    "T3": ("nothing",),
    "T4": ("caregiver",),
    "T5": ("check-in", "alert"),                 # check-in or alert, cleared
    "T6": ("caregiver",),                        # provisional: HR thresholds
    "T7": ("nothing",),
}
PROVISIONAL = {"hr_spike_no_fall", "T6"}
NEED_CLEAR = {"fall_then_get_up", "T2", "T5"}    # "... cleared, no caregiver"
CLEARS = ("cleared", "cancelled by wearer", "wearer answered OK")


def level(am) -> str:
    ev = [e["event"] for e in am.events]
    if any(e.startswith(("caregiver", "urgent")) for e in ev):
        return "caregiver"
    if "alert" in ev:
        return "alert"
    if "check-in" in ev:
        return "check-in"
    return "nothing"


def settings() -> dict:
    s = dict(weights=fusion.WEIGHTS, hr_low=fusion.HR_LOW, hr_high=fusion.HR_HIGH,
             hr_min_quality=fusion.HR_MIN_QUALITY, smooth_s=fusion.SMOOTH_S,
             confirm_s=fusion.CONFIRM_S, night=fusion.NIGHT, walking=fusion.WALKING,
             check_in=alerts.CHECK_IN, alert=alerts.ALERT, cancel_s=alerts.CANCEL_S,
             checkin_answer_s=alerts.CHECKIN_ANSWER_S, urgent_still_s=alerts.URGENT_STILL_S)
    v = getattr(fusion, "VERSION", 1)
    if v >= 2:                    # v1's fingerprint stays what was frozen on 8 Oct
        s.update(version=v, upright=fusion.UPRIGHT, upright_share=fusion.UPRIGHT_SHARE,
                 recover_s=fusion.RECOVER_S)
    if v >= 3:
        s.update(second_checkin_gap_s=alerts.SECOND_CHECKIN_GAP_S)
    s["fingerprint"] = hashlib.sha256(json.dumps(s, sort_keys=True).encode()).hexdigest()[:16]
    return s


def judge(name, rows, am, truth, expected, key=None):
    got = level(am)
    ok = got in expected
    ev = [e["event"] for e in am.events if e["event"] != "recovered"]
    if (key or name) in NEED_CLEAR:
        ok = ok and bool(ev) and ev[-1] in CLEARS     # must end cleared
        got += ", cleared" if ev and ev[-1] in CLEARS else ", NOT cleared"
    t0 = truth["events"][0]["t"] if truth.get("events") else None
    first = {}
    for e in am.events:
        k = e["event"].split(":")[0]
        first.setdefault(k, e["t"])

    def rel(k):
        if k not in first:
            return "-"
        return f"{first[k]:.1f}s" + (f" (+{first[k] - t0:.1f})" if t0 is not None else "")
    top = max(rows, key=lambda r: r["score"]) if rows else None
    lead = alerts.LABEL[am.events[0]["type"]] if am.events else "-"
    return dict(name=name, expected="/".join(expected), got=got, ok=ok, path=am.outcome(),
                lead=lead, peak=round(top["score"], 2) if top else 0.0,
                checkin=rel("check-in"), alert=rel("alert"), cleared=rel("cleared"),
                caregiver=rel("caregiver alerted"), provisional=(key or name) in PROVISIONAL)


def table(res, title):
    L = [title, "",
         f"  {'scenario':26s} {'expected':16s} {'got':18s} {'ok':4s} {'first type':12s} "
         f"{'alert at (+s after event)':26s} {'caregiver at':16s} path"]
    for r in res:
        ok = ("PASS" if r["ok"] else "FAIL") + ("*" if r["provisional"] else "")
        L.append(f"  {r['name']:26s} {r['expected']:16s} {r['got']:18s} {ok:4s} {r['lead']:12s} "
                 f"{r['alert']:26s} {r['caregiver']:16s} {r['path']}")
    n = sum(r["ok"] for r in res)
    L += ["  * provisional: heart-rate thresholds 60/100 bpm not yet confirmed with sources",
          f"RESULT {title.split(':')[0]}: {n} of {len(res)} as expected"]
    return L, n


def tune():
    res = []
    for name, exp in TUNE.items():
        wins, truth = load_saved(name)
        rows, am = replay(wins, truth.get("clock"))
        res.append(judge(name, rows, am, truth, exp))
    s = settings()
    L, n = table(res, f"tune: 7 tuning scenarios, settings {s['fingerprint']}")
    return L, n, res, s


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["tune", "freeze", "test"])
    ap.add_argument("--again", action="store_true", help="re-run test (labelled as a re-run)")
    ap.add_argument("--after-test", default=None, metavar="REASON",
                    help="freeze a version made after the held-out test (v2+)")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)

    if a.mode in ("tune", "freeze"):
        L, n, res, s = tune()
        if a.mode == "freeze":
            v = s.get("version", 1)
            path = frozen_path(v)
            if n < len(TUNE):
                L.append("NOT FROZEN: not all tuning scenarios pass")
            elif path.exists():
                old = json.loads(path.read_text())
                L.append(f"ALREADY FROZEN on {old['date']} as {old['fingerprint']}"
                         + (" (same settings)" if old["fingerprint"] == s["fingerprint"]
                            else " (settings have changed since: do not change after freezing)"))
            elif v >= 2 and not a.after_test:
                L.append('NOT FROZEN: a version made after the held-out test needs '
                         '--after-test "reason"')
            else:
                extra = dict(after_test=v >= 2, reason=a.after_test,
                             based_on=json.loads(FROZEN.read_text())["fingerprint"]
                             if v >= 2 and FROZEN.exists() else None)
                path.write_text(json.dumps(dict(s, date=str(date.today()), tune_results=res,
                                                **extra), indent=2))
                L.append(f"FROZEN v{v} {s['fingerprint']} -> {path.relative_to(ROOT)}"
                         + (" (made AFTER the held-out test)" if v >= 2 else ""))
        (OUT / "fusion_tune.txt").write_text("\n".join(L) + "\n")
        print("\n".join(L))
        return

    # test (Stage 7.4)
    s = settings()
    v = s.get("version", 1)
    if not frozen_path(v).exists():
        sys.exit(f"Freeze v{v} first: py -m testing.fusion_eval freeze"
                 + (' --after-test "reason"' if v >= 2 else ""))
    fz = json.loads(frozen_path(v).read_text())
    if fz["fingerprint"] != s["fingerprint"]:
        sys.exit(f"Settings changed since freezing ({fz['fingerprint']} -> {s['fingerprint']}). "
                 "The test must run on the frozen settings.")
    try:
        from simulator.test_scenarios import TESTS, build_test
    except ImportError:
        sys.exit("T1-T7 not built yet (Stage 7.4).")
    done = OUT / "fusion_test.txt"
    if v >= 2:
        a.again = True                # v2+ is always a check on cases already seen
    if done.exists() and not a.again:
        print(done.read_text())
        sys.exit("Test already run once (above). Use --again only to re-check; it is labelled a re-run.")
    import pandas as pd
    from simulator.run import run_scenario
    tests_dir = OUT / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    # build all seven first: a missing recording stops the test before any runs
    built = {}
    for tid in TEST:
        try:
            built[tid] = build_test(tid)
        except LookupError as e:
            sys.exit(f"{tid} cannot be built: {e}\nNothing was run.")
        for sw in getattr(built[tid], "swaps", []):
            print(f"  {tid} substituted {sw}")
    res = []
    for tid in TEST:
        print(f"  running {tid} ...", flush=True)
        sc, wins = run_scenario(built[tid])
        flat = pd.DataFrame([r for w in wins for r in w])
        flat.assign(extras=flat.extras.apply(json.dumps)).to_csv(
            tests_dir / f"{tid}_outputs.csv", index=False)
        (tests_dir / f"{tid}_truth.json").write_text(json.dumps(dict(
            name=tid, title=sc.title, description=sc.description, simulated=True,
            seconds=round(sc.seconds, 1), clock=sc.clock, truth=sc.truth,
            events=sc.events), indent=2))
        rows, am = replay(wins, sc.clock)
        truth = dict(events=sc.events)
        res.append(judge(f"{tid} {TESTS[tid]['short']}", rows, am, truth, TEST[tid], key=tid))
    label = ("RE-RUN of the held-out test" if done.exists() else "held-out test, first run")
    if v >= 2:
        label = (f"v{v} AFTER-TEST CHECK (these cases were already seen; v1 first run is "
                 "the held-out result)")
    L, n = table(res, f"test: T1-T7 {label}, frozen settings {s['fingerprint']}")
    sw = [f"{t}: {x}" for t in TEST for x in getattr(built[t], "swaps", [])]
    if sw:
        L.insert(-1, "  substituted recordings: " + "; ".join(sw))
    (OUT / (f"fusion_test_v{v}.txt" if v >= 2 else
            "fusion_test_rerun.txt" if done.exists() else "fusion_test.txt")).write_text(
        "\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()

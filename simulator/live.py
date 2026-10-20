"""
simulator/live.py -- play a SIMULATED scenario in real time, as if the band
were sending it now, and save each window to the database the moment it is
scored (Stage 5.9). This is what the website will show live in Stage 8.

    py -m simulator.live hard_fall_long_lie              real time, saved under its demo wearer
    py -m simulator.live demo_long_lie --speed 4         review demo: impact wears off,
                                                         long lie stays high, then urgent
    py -m simulator.live faint_collapse --speed 4        four times faster
    py -m simulator.live normal_day --no-db              print only
    py -m simulator.live fall_then_get_up --wearer 3     save under another demo wearer

While it runs, watch it from another terminal as a given user:
    py -m core.watch --user 5        (Caregiver 1)

Each saved window also sends a 'guardian_live' notification (ids only),
which a backend can use to push updates to the browser.
Stage 7.5: every window also goes through fusion and the alert manager
(frozen settings). The emergency score is saved as stream 'fusion', and each
alert episode as a row in the alerts table (needs 05_alerts.sql once).
The held-out tests can be replayed too: py -m simulator.live T1 --speed 2

Output (review demo): one table row per window, headings printed once:
time | motion | still (s) | activity | heart (bpm) | emergency, with alert
events at the end of their row (core/screen.py, same as core.watch).
The heart module's 30 s warm-up (the scenario lead-in) runs first without
being shown or saved, so the table starts with a heart rate; --show-warmup
plays it as before.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.pipeline import Pipeline  # noqa: E402
from data.loader import WINDOW_SEC  # noqa: E402
from simulator.run import WEARER  # noqa: E402
from core.alerts import LABEL, AlertManager, fusion_record  # noqa: E402
from core.fusion import Fusion  # noqa: E402
from core.screen import from_records, header, legend, marker  # noqa: E402
from simulator.scenarios import DEMOS, SCENARIOS, build  # noqa: E402
from simulator.test_scenarios import TESTS, build_test  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("name", choices=list(SCENARIOS) + list(DEMOS) + list(TESTS))
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--wearer", type=int, default=None)
    ap.add_argument("--no-db", action="store_true")
    ap.add_argument("--show-warmup", action="store_true",
                    help="also play and show the heart module's 30 s warm-up")
    a = ap.parse_args(argv)

    sc = build_test(a.name) if a.name in TESTS else build(a.name)
    pipe = Pipeline()
    fu, am = Fusion(clock=sc.clock), AlertManager()
    wid = a.wearer or WEARER.get(a.name, 2)
    store = run_id = None
    if not a.no_db:
        from core.store import Store
        store = Store()
        truth = dict(name=sc.name, title=sc.title, simulated=True, truth=sc.truth,
                     events=sc.events, seconds=round(sc.seconds, 1))
        run_id = store.start_run(wid, sc.name, truth=truth, status="live",
                                 notes=f"live replay x{a.speed:g}")
    print(f"\nLIVE (SIMULATED)   {sc.title}   |   wearer {wid}"
          + (f"   |   run {run_id}" if run_id else "   |   not saved")
          + f"   |   speed x{a.speed:g}\n")
    if not a.show_warmup:
        print("  band already worn: the heart module's 30 s warm-up runs first, not shown\n")
    print("\n".join(legend() + header()), flush=True)

    # warm-up: windows go through every module, fusion and the alert manager
    # straight away, but are not played in real time, printed or saved
    start = time.monotonic() if a.show_warmup else None
    warm, status = not a.show_warmup, "done"
    try:
        for w in sc.windows():
            t_end = w["t"] + WINDOW_SEC
            if start is not None:
                wait = start + t_end / a.speed - time.monotonic()
                if wait > 0:
                    time.sleep(wait)                   # the window "arrives" now
            rows = pipe.step(w)
            for r in rows:
                r["t"] = round(r["t"] + WINDOW_SEC, 2)
            f = fu.step(rows)
            n = len(am.events)
            state = am.step(f)
            if warm:
                p = next((r for r in rows if r["stream"] == "physiological"), None)
                if p is not None and p["extras"].get("state") == "WARMING_UP":
                    continue
                warm = False
                start = time.monotonic() - t_end / a.speed
            if store is not None:
                store.sync_alerts(run_id, wid, am)     # alerts first: a watcher that
                store.write(run_id, wid, rows + [fusion_record(f, state)])  # sees the row
                store.notify(run_id, wid, rows[0]["t"])                     # sees its alert
            print(from_records(rows, f["score"]) + marker(am.events[n:], LABEL), flush=True)
    except KeyboardInterrupt:
        status = "stopped"
        print("\n  stopped")
    finally:
        if store is not None:
            store.end_run(run_id, status)
            store.close()
            print(f"\n  run {run_id} saved ({status})")
        print(f"  alerts: {am.outcome()}")


if __name__ == "__main__":
    main()

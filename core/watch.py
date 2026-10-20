"""
core/watch.py -- follow live runs from the database AS a given user
(Stage 5.9). A terminal stand-in for the caregiver / admin live screen:
it only shows what that user is allowed to see.

    py -m core.watch --user 5            Caregiver 1 (sees Wearer 1 only)
    py -m core.watch --user 1            Admin (sees everyone)
    py -m core.watch --user 4            Wearer 3 (sees only themselves)

Stage 7.5: also shows the emergency score and the alert events of each run
(only for wearers this user may see).

Output (review demo): the same table as the band side (core/screen.py),
headings printed once: time | motion | still (s) | activity | heart (bpm) |
five type scores | emergency (the highest type score). Every window is shown, not only the latest, and alert events
(from the alerts table) are added at the end of their row.

Stop with Ctrl+C. Start a live replay in another terminal:
    py -m simulator.live hard_fall_long_lie --speed 2
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.alerts import LABEL  # noqa: E402
from core.screen import header, legend, marker, row  # noqa: E402
from core.store import as_user, connect  # noqa: E402

NEW_ROWS = """
SELECT t_s, stream, score, extras FROM stream_outputs
WHERE run_id = %s AND t_s > %s ORDER BY t_s
"""


def _events(c, run_id):
    """Every alert event of the run, from the alerts table history."""
    ev = []
    for (hist,) in c.execute("SELECT history FROM alerts WHERE run_id = %s ORDER BY id",
                             (run_id,)).fetchall():
        ev += hist or []
    return ev


def poll(uid, seen):
    """New table lines for this user since the last poll. seen: run_id -> last t."""
    out = []
    with as_user(uid) as c:
        runs = c.execute(
            "SELECT r.id, u.full_name FROM runs r JOIN users u ON u.id = r.wearer_id "
            "WHERE r.status = 'live' ORDER BY r.id").fetchall()
        live = {r[0] for r in runs}
        for run_id in [r for r in seen if r not in live and seen[r] is not None]:
            out.append(f"  ---- run {run_id} ended ----")
            seen[run_id] = None
        for run_id, who in runs:
            if seen.get(run_id, 0) is None:
                continue
            if run_id not in seen:
                seen[run_id] = -1.0
                out.append(f"  ---- {who}: live run {run_id} started ----")
            by_t = {}
            # t_s is stored as a float, so 54.6 can come back as 54.600002:
            # skip anything within 0.1 s of the last printed window (windows are 1.24 s apart)
            for t, stream, score, ex in c.execute(NEW_ROWS, (run_id, seen[run_id] + 0.1)).fetchall():
                by_t.setdefault(round(float(t), 2), {})[stream] = (float(score), ex or {})
            if not by_t:
                continue
            events = _events(c, run_id)
            for t in sorted(by_t):
                s = by_t[t]
                if "fusion" not in s:          # fusion is written with the window;
                    break                      # wait for the complete window
                m, a, p = s.get("motion"), s.get("activity"), s.get("physiological")
                line = row(t,
                           m[0] if m else None,
                           (m[1].get("still_s") or 0.0) if m else None,
                           a[1].get("state") if a else None,
                           p[1].get("hr") if p else None,
                           s["fusion"][0],
                           types=s["fusion"][1].get("scores") or {})
                here = [e for e in events if abs(float(e["t"]) - t) < 0.01]
                out.append(line + marker(here, LABEL))
                seen[run_id] = t
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", type=int, required=True, help="users.id (see 03_seed.sql)")
    ap.add_argument("--every", type=float, default=0.5)
    a = ap.parse_args(argv)
    with connect() as c:
        u = c.execute("SELECT full_name, role FROM users WHERE id = %s", (a.user,)).fetchone()
    if u is None:
        sys.exit(f"no user {a.user}")
    print(f"\nCAREGIVER VIEW   watching as {u[0]} ({u[1]})   |   started "
          f"{time.strftime('%H:%M:%S')}   |   Ctrl+C to stop\n")
    print("\n".join(legend() + header(types=True)), flush=True)
    seen, waiting = {}, False
    try:
        while True:
            lines = poll(a.user, seen)
            if lines:
                print("\n".join(lines), flush=True)
                waiting = False
            elif not seen and not waiting:
                print("  (waiting for a live run this user may see)", flush=True)
                waiting = True
            time.sleep(a.every)
    except KeyboardInterrupt:
        print("\n  stopped")


if __name__ == "__main__":
    main()

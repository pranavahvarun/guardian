"""
demo/show_alerts.py -- the alerts table AS a given user sees it (Alert
Escalation Module output, stored in Postgres). The database itself limits
what each user may see (row-level security, backend/db/init/02_access.sql).

    py -m demo.show_alerts --user 5          Caregiver 1 (sees Wearer 1 only)
    py -m demo.show_alerts --user 1          Admin (sees everyone)
    py -m demo.show_alerts --user 5 --last 5

Needs Docker running and at least one run saved (simulator.live or
simulator.run all --db).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.store import as_user, connect  # noqa: E402

Q = """
SELECT a.id, a.run_id, r.scenario, u.full_name, a.t_s, a.kind, a.stage, a.status,
       a.emergency_score, jsonb_array_length(a.history), a.history
FROM alerts a JOIN runs r ON r.id = a.run_id JOIN users u ON u.id = a.wearer_id
ORDER BY a.id DESC LIMIT %s
"""


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", type=int, required=True)
    ap.add_argument("--last", type=int, default=8)
    a = ap.parse_args(argv)
    with connect() as c:
        u = c.execute("SELECT full_name, role FROM users WHERE id = %s", (a.user,)).fetchone()
        total = c.execute("SELECT count(*) FROM alerts").fetchone()[0]
    if u is None:
        sys.exit(f"no user {a.user}")
    with as_user(a.user) as c:
        rows = c.execute(Q, (a.last,)).fetchall()
        mine = c.execute("SELECT count(*) FROM alerts").fetchone()[0]
    with as_user(0) as c:
        nobody = c.execute("SELECT count(*) FROM alerts").fetchone()[0]

    print(f"\nALERTS (Alert Escalation Module)   as {u[0]} ({u[1]})")
    print(f"  database row-level security: this user sees {mine} of {total} alerts; "
          f"with no user logged in: {nobody}\n")
    print(f"  {'alert':>5s}  {'run':>4s}  {'scenario':20s} {'wearer':24s} {'at':>7s}  "
          f"{'type':12s} {'stage':22s} {'status':9s} {'score':>5s}")
    print("  " + "-" * 118)
    for i, run, scen, who, t, kind, stage, status, sc, n, _ in rows:
        print(f"  {i:5d}  {run:4d}  {(scen or '-')[:20]:20s} {who[:24]:24s} {t:6.1f}s  "
              f"{kind:12s} {stage[:22]:22s} {status:9s} {sc:5.2f}")
    if rows:
        i, *_, hist = rows[0]
        print(f"\n  history of alert {i} (every step stored with its evidence):")
        for e in hist:
            ev = e.get("evidence", {})
            hr = ev.get("hr")
            print(f"    {e['t']:7.1f} s  {e['event']:30s} score {e['score']:.2f}   impact "
                  f"{ev.get('impact', 0):.2f}  still {ev.get('still_s', 0):.0f} s  heart "
                  f"{'--' if hr is None else round(hr)} bpm  moved again {ev.get('moved_again')}")
    print()


if __name__ == "__main__":
    main()

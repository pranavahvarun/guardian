"""
testing/check_activity_inputs.py -- does Sudharshan's activity model see the
same input inside our pipeline as it saw in training? (Stage 6.1)

Three checks, all through the real wrapper (streams/activity/stream.py):

A  Timing. WEDA-FALL timestamps arrive in bursts. His training loader trusted
   the timestamps (sort, drop duplicates, interpolate). Our loader spreads the
   samples evenly ("even"). Same recordings, both ways, his model on top.
   IN-SAMPLE: his final model was trained on all 25 WEDA people, so the
   accuracy is too high in both columns. Only the GAP between them matters.

B  UMAFall with the right units. His UMAFall run converted acc to m/s^2 but
   left gyro in deg/s (model wants rad/s). Here the wrapper converts both.
   Run for wrist sensor ID 2 (our loader) and ID 3 (his loader).

C  Which UMAFall sensor ID is the wrist: prints the file's own header lines.

    py -m testing.check_activity_inputs
    py -m testing.check_activity_inputs --skip-uma

Writes testing/outputs/stage6/activity_inputs.txt (+ per-activity CSV).
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data import loader as uma  # noqa: E402
from data import loader_weda as weda  # noqa: E402
from data.loader import iter_windows  # noqa: E402
from streams.activity import ActivityStream  # noqa: E402

OUT = ROOT / "testing" / "outputs" / "stage6"
WEDA_ROOT = ROOT / "data" / "raw" / "WEDA-FALL" / "dataset" / "50Hz"
UMA_ROOT = ROOT / "data" / "raw" / "UMAFall"

# WEDA code -> his class (his activity_config.json activity_map)
# UMAFall activity -> his class (his experiments/evaluate_umafall.py)
UMA_MAP = {
    "walking": "LOCOMOTION", "jogging": "LOCOMOTION",
    "goupstairs": "LOCOMOTION", "godownstairs": "LOCOMOTION",
    "sitting_gettinguponachair": "SITTING_STANDING",
    "aplausing": "UPPER_BODY", "clapping": "UPPER_BODY", "handsup": "UPPER_BODY",
    "makingacall": "UPPER_BODY", "openingdoor": "UPPER_BODY",
    "bending": "OTHER", "hopping": "OTHER", "lyingdown_onabed": "OTHER",
}


def _run(stream, rec):
    """Per-window predicted state for one recording (fresh history)."""
    stream.reset()
    out = []
    for w, _ in iter_windows(rec):
        stream.score(w)
        out.append(stream.last_state)
    return out


def _scores(y, p):
    y, p = np.asarray(y), np.asarray(p)
    if not len(y):
        return float("nan"), float("nan")
    acc = float((y == p).mean())
    bal = float(np.mean([(p[y == c] == c).mean() for c in np.unique(y)]))
    return acc, bal


def check_timing(stream, L):
    amap = stream.engine.config["activity_map"]
    files = sorted(f for f in WEDA_ROOT.glob("D*/U*_R*_accel.csv")
                   if f.parent.name.upper() in amap and weda.NAME_RE.match(f.name))
    # (the glob also matches *_vertical_accel.csv; NAME_RE drops those)
    if not files:
        L.append(f"A  WEDA: no files under {WEDA_ROOT}")
        return
    res = {}
    for timing in ("timestamps", "even"):
        y, p, n_skip = [], [], 0
        for f in files:
            try:
                rec = weda.read_record(f, timing=timing)
            except Exception:                                    # noqa: BLE001
                n_skip += 1
                continue
            pred = _run(stream, rec)
            y += [amap[f.parent.name.upper()]] * len(pred)
            p += pred
        res[timing] = (*_scores(y, p), len(y), n_skip)
    L += ["A  TIMING on WEDA-FALL ADLs (D01-D11, all people; IN-SAMPLE)",
          f"   {'timing':12s} {'accuracy':>9s} {'balanced':>9s} {'windows':>8s} {'bad files':>9s}"]
    for k, (a, b, n, s) in res.items():
        who = "his loader" if k == "timestamps" else "our loader"
        L.append(f"   {k:12s} {a:9.3f} {b:9.3f} {n:8d} {s:9d}   ({who})")
    gap = res["timestamps"][1] - res["even"][1]
    L.append(f"RESULT timing: balanced accuracy his timing {res['timestamps'][1]:.3f}, "
             f"ours {res['even'][1]:.3f} (gap {gap:+.3f}, in-sample)")
    L.append("")


def _uma_label(name):
    low = name.lower()
    if "_adl_" not in low:
        return None, None
    rest = low.split("_adl_", 1)[1]
    for key in sorted(UMA_MAP, key=len, reverse=True):
        if rest.startswith(key):
            return key, UMA_MAP[key]
    return None, None


def check_umafall(stream, L, rows):
    files = sorted(UMA_ROOT.glob("*.csv"))
    files = [f for f in files if _uma_label(f.name)[1]]
    if not files:
        L.append(f"B  UMAFall: no ADL files under {UMA_ROOT}")
        return
    for sid in (2, 3):
        y, p, per, bad = [], [], defaultdict(lambda: [0, 0]), 0
        for f in files:
            act, lab = _uma_label(f.name)
            try:
                rec = uma.read_umafall(f, sensor_id=sid)
            except Exception:                                    # noqa: BLE001
                bad += 1
                continue
            if rec.gyro is None:
                bad += 1
                continue
            pred = _run(stream, rec)
            y += [lab] * len(pred)
            p += pred
            per[act][0] += sum(q == lab for q in pred)
            per[act][1] += len(pred)
        a, b = _scores(y, p)
        L.append(f"B  UMAFall ADLs, sensor ID {sid}: accuracy {a:.3f}, balanced {b:.3f}, "
                 f"{len(y)} windows, {bad} files with no rows or no gyro for this ID")
        for act, (c, n) in sorted(per.items()):
            L.append(f"     {act:28s} {c / n if n else float('nan'):6.3f}  ({n} windows)")
            rows.append(dict(sensor_id=sid, activity=act, correct=c, windows=n))
        L.append(f"RESULT UMAFall (units fixed) sensor {sid}: accuracy {a:.3f}, "
                 f"balanced {b:.3f} (his reported 0.286 / 0.284)")
    L.append("")


def check_header(L):
    files = sorted(UMA_ROOT.glob("*.csv"))
    if not files:
        return
    L.append(f"C  UMAFall header of {files[0].name} (lines that name a sensor position):")
    with open(files[0], "r", encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh):
            if i > 60 or not line.startswith("%"):
                break
            if any(k in line.upper() for k in ("WRIST", "WAIST", "ANKLE", "CHEST",
                                               "POCKET", "SENSOR")):
                L.append("   " + line.rstrip()[:110])
    L.append("")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-uma", action="store_true")
    a = ap.parse_args(argv)
    stream = ActivityStream.load()
    L = ["Activity module input check (Stage 6.1)",
         f"model note: {stream.version_note or 'none'}", ""]
    check_timing(stream, L)
    rows = []
    if not a.skip_uma:
        check_header(L)
        check_umafall(stream, L, rows)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "activity_inputs.txt").write_text("\n".join(L) + "\n")
    if rows:
        pd.DataFrame(rows).to_csv(OUT / "activity_umafall_per_activity.csv", index=False)
    print("\n".join(L))
    print(f"saved {OUT / 'activity_inputs.txt'}")


if __name__ == "__main__":
    main()

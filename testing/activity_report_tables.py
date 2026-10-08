"""
testing/activity_report_tables.py -- activity module results for screenshots
(Stage 6). Two commands:

    py -m testing.activity_report_tables same    WEDA-FALL, people never trained on
    py -m testing.activity_report_tables cross   UMAFall, a dataset never trained on

SAME: leave-one-person-out on WEDA-FALL daily activities (25 people), the
model in use (his features and settings, our even timing). Uses the feature
cache from testing/activity_loso_timing.py (built if missing). About 1-2 min.

CROSS:

Both activity models are trained on WEDA-FALL only. Here they are run,
frozen, through the real wrapper (streams/activity/stream.py) on UMAFall's
wrist sensor (ID 2), with the units the models expect (the wrapper converts
g -> m/s^2 and deg/s -> rad/s). UMAFall activities are mapped to the four
classes as in Sudharshan's own evaluation (experiments/evaluate_umafall.py).


Model: the one in use (models/activity_rf_even.joblib).
Writes testing/outputs/stage6/activity_same.txt and activity_cross.txt
"""

from __future__ import annotations

import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data import loader as uma  # noqa: E402
from streams.activity import ActivityStream  # noqa: E402
from testing.check_activity_inputs import UMA_ROOT, _run, _uma_label  # noqa: E402

OUT = ROOT / "testing" / "outputs" / "stage6"
W = 78
MODEL = ROOT / "models" / "activity_rf_even.joblib"     # the model in use
CLASSES = ["LOCOMOTION", "SITTING_STANDING", "UPPER_BODY", "OTHER"]
WRIST_ID = 2


def metrics(y, p):
    y, p = np.asarray(y), np.asarray(p)
    acc = float((y == p).mean())
    rec = {c: float((p[y == c] == c).mean()) for c in CLASSES if (y == c).any()}
    bal = float(np.mean(list(rec.values())))
    f1s = []
    for c in rec:
        tp = float(((p == c) & (y == c)).sum())
        prec = tp / max((p == c).sum(), 1)
        f1s.append(0.0 if tp == 0 else 2 * prec * rec[c] / (prec + rec[c]))
    return acc, bal, float(np.mean(f1s)), rec


def cross():
    files = [f for f in sorted(UMA_ROOT.glob("*.csv")) if _uma_label(f.name)[1]]
    if not files:
        print(f"No UMAFall ADL files under {UMA_ROOT}")
        sys.exit(1)
    recs, skipped = [], 0
    for f in files:
        act, lab = _uma_label(f.name)
        try:
            r = uma.read_umafall(f, sensor_id=WRIST_ID)
        except Exception:                                        # noqa: BLE001
            skipped += 1
            continue
        if r.gyro is None:
            skipped += 1
            continue
        recs.append((act, lab, r))
    people = {r.subject for _, _, r in recs}

    s = ActivityStream.load(MODEL)
    y, p, per = [], [], defaultdict(lambda: [0, 0])
    for act, lab, r in recs:
        pred = _run(s, r)
        y += [lab] * len(pred)
        p += pred
        per[act][0] += sum(q == lab for q in pred)
        per[act][1] += len(pred)
    acc, bal, f1, rec = metrics(y, p)
    y = np.array(y)

    L = ["=" * W, "ACTIVITY MODULE - DATASET NEVER TRAINED ON (UMAFall)",
         "Trained on WEDA-FALL only, run frozen on UMAFall daily activities",
         f"(wrist sensor). {len(recs)} recordings, {len(people)} people, "
         f"{len(y)} windows of 2.5 s.", "=" * W,
         f"  {'accuracy':30s}{acc:10.1%}",
         f"  {'balanced accuracy':30s}{bal:10.1%}",
         f"  {'macro F1':30s}{f1:10.1%}",
         "-" * W, "  caught, per class:"]
    for c in CLASSES:
        if c in rec:
            L.append(f"    {c:28s}{rec[c]:10.0%}   ({int((y == c).sum())} windows)")
    L += ["-" * W, "  caught, per UMAFall activity:"]
    for act in sorted(per, key=lambda a: (_uma_label("x_ADL_" + a)[1], a)):
        cls = _uma_label("x_ADL_" + act)[1]
        c, n = per[act]
        L.append(f"    {act[:26]:26s} {cls:18s}{c / max(n, 1):8.0%}   ({n} windows)")
    L.append("-" * W)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "activity_cross.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def same():
    from testing import activity_loso_timing as lt
    print("building or loading WEDA features ...", flush=True)
    ev, _ = lt.build("even")
    X, ycol, g, rec = ev[lt.NAMES], ev.label.values, ev.subject.values, ev.rec.values
    y_all, raw_all, sm_all = [], [], []
    subs = sorted(set(g))
    for k, s in enumerate(subs, 1):
        print(f"  fold {k}/{len(subs)}", end="\r", flush=True)
        tr, te = g != s, g == s
        m = lt._rf().fit(X[tr], ycol[tr])
        P = m.predict_proba(X[te])
        y_all += list(ycol[te])
        raw_all += list(m.classes_[P.argmax(1)])
        sm_all += list(m.classes_[lt._smooth(P, rec[te]).argmax(1)])
    raw, sm = metrics(y_all, raw_all), metrics(y_all, sm_all)
    y, p = np.array(y_all), np.array(sm_all)
    L = ["=" * W, "ACTIVITY MODULE - SAME DATASET, NEW PEOPLE (WEDA-FALL)",
         f"Leave-one-person-out: each of the {len(subs)} people is tested by a model that",
         f"never saw them. Daily activities D01-D11, {len(y)} windows of 2.5 s.", "=" * W,
         f"  {'':30s}{'raw forest':>16s}{'smoothed (in use)':>20s}", "-" * W]
    for i, lab in enumerate(["accuracy", "balanced accuracy", "macro F1"]):
        L.append(f"  {lab:30s}{raw[i]:16.1%}{sm[i]:20.1%}")
    L += ["-" * W, "  caught, per class (smoothed):"]
    for c in CLASSES:
        L.append(f"    {c:28s}{sm[3].get(c, float('nan')):16.0%}   ({int((y == c).sum())} windows)")
    L += ["-" * W, "  confusion (smoothed): rows = true class, columns = predicted",
          "    " + " " * 18 + "".join(f"{c[:10]:>12s}" for c in CLASSES)]
    for c in CLASSES:
        L.append(f"    {c[:18]:18s}" + "".join(f"{int(((y == c) & (p == q)).sum()):12d}" for q in CLASSES))
    L += ["-" * W,
          "  Smoothed = mean of the last 3 windows' probabilities (causal).",
          "  His report (his timing): 80.6% accuracy, 74.0% balanced, 75.6% F1.",
          "  WEDA-FALL: young volunteers and adults aged 77-95, wrist smartwatch."]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "activity_same.txt").write_text("\n".join(L) + "\n")
    print()
    print("\n".join(L))


def main():
    warnings.filterwarnings("ignore")
    cmds = {"same": same, "cross": cross}
    if len(sys.argv) != 2 or sys.argv[1] not in cmds:
        print(__doc__)
        sys.exit(1)
    cmds[sys.argv[1]]()


if __name__ == "__main__":
    main()

"""
testing/activity_loso_timing.py -- does the WEDA timing choice matter for
Sudharshan's activity model OUT OF SAMPLE? (Stage 6.1, follow-up)

check_activity_inputs.py (part A) showed his final model reads our evenly
spaced windows differently from his timestamp-based ones (balanced 1.000 vs
0.914), but that was in-sample. This runs leave-one-person-out with HIS
features and HIS random forest settings, three ways:

  A  train his timing,  test his timing   (his setup; should land near his
                                            raw 70.7% / smoothed 74.0% balanced)
  B  train his timing,  test our timing   (what the pipeline runs today)
  C  train our timing,  test our timing   (model retrained on our loader)

"his timing" = trust the WEDA timestamps (sort, drop duplicates, interpolate),
"our timing" = spread samples evenly over the file's span (data/loader_weda).
The real band samples at a fixed rate, which is closer to "our timing".

Decision rule, fixed before running: if C beats B by more than 0.02 balanced
accuracy (smoothed), retrain the deployed model on our timing (--save).

    py -m testing.activity_loso_timing
    py -m testing.activity_loso_timing --save    (also writes models/activity_rf_even.joblib)
    py -m testing.activity_loso_timing --save-only   (only writes the model, no LOSO)

Takes several minutes; progress is printed. Features are cached in
testing/outputs/stage6/activity_features_*.csv, so a re-run is faster
(delete those files if the loader or features change).

Writes testing/outputs/stage6/activity_loso_timing.txt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data import loader_weda as weda  # noqa: E402
from data.loader import iter_windows  # noqa: E402
from streams.activity.features import calculate_features  # noqa: E402

OUT = ROOT / "testing" / "outputs" / "stage6"
WEDA_ROOT = ROOT / "data" / "raw" / "WEDA-FALL" / "dataset" / "50Hz"
CONFIG = json.loads((ROOT / "models" / "activity_config.json").read_text())
AMAP, NAMES = CONFIG["activity_map"], CONFIG["features"]["names"]
RF = CONFIG["random_forest"]
G_MS2, DEG_TO_RAD = 9.80665, np.pi / 180.0


def build(timing):
    """Feature table (one row per window) for every WEDA ADL recording.
    Cached in testing/outputs/stage6/ so a second run skips this step."""
    cache = OUT / f"activity_features_{timing}.csv"
    if cache.exists():
        df = pd.read_csv(cache)
        print(f"  {timing}: {len(df)} windows from cache ({cache.name})", flush=True)
        return df, 0
    files = sorted(f for f in WEDA_ROOT.glob("D*/U*_R*_accel.csv")
                   if f.parent.name.upper() in AMAP and weda.NAME_RE.match(f.name))
    rows, bad = [], 0
    for rid, f in enumerate(files):
        if rid % 100 == 0:
            print(f"  {timing}: file {rid}/{len(files)}", flush=True)
        try:
            rec = weda.read_record(f, timing=timing)
        except Exception:                                        # noqa: BLE001
            bad += 1
            continue
        if rec.gyro is None:
            bad += 1
            continue
        for w, _ in iter_windows(rec):
            x = np.hstack([w["acc"] * G_MS2, w["gyro"] * DEG_TO_RAD])
            feat = calculate_features(x)
            feat.update(subject=rec.subject, rec=rid, label=AMAP[f.parent.name.upper()])
            rows.append(feat)
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)
    return df, bad


def _rf():
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(n_estimators=RF["n_estimators"],
                                  random_state=RF["random_state"],
                                  class_weight=RF["class_weight"], n_jobs=-1)


def _smooth(P, rec, k=3):
    """Causal mean of the last k windows' probabilities, per recording."""
    out = np.empty_like(P)
    for r in np.unique(rec):
        idx = np.where(rec == r)[0]
        for j, i in enumerate(idx):
            out[i] = P[idx[max(0, j - k + 1): j + 1]].mean(axis=0)
    return out


def _metrics(y, p):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
    return (accuracy_score(y, p), balanced_accuracy_score(y, p),
            f1_score(y, p, average="macro"))


def loso(train, test, tag=""):
    subs = sorted(set(train.subject) & set(test.subject))
    y_all, raw_all, sm_all = [], [], []
    for k, s in enumerate(subs, 1):
        print(f"  {tag} fold {k}/{len(subs)} (leave out {s})", flush=True)
        tr, te = train[train.subject != s], test[test.subject == s]
        if te.empty:
            continue
        m = _rf().fit(tr[NAMES], tr.label)
        P = m.predict_proba(te[NAMES])
        cls = m.classes_
        y_all += list(te.label)
        raw_all += list(cls[P.argmax(1)])
        sm_all += list(cls[_smooth(P, te.rec.values).argmax(1)])
    return _metrics(y_all, raw_all), _metrics(y_all, sm_all), len(subs), len(y_all)


def save(ev):
    """Train on every person with our timing and save the deployed model."""
    import joblib
    import sklearn
    m = _rf().fit(ev[NAMES], ev.label)
    m.n_jobs = 1
    out = ROOT / "models" / "activity_rf_even.joblib"
    joblib.dump(m, out, compress=3)
    msg = (f"saved {out.name}: {len(ev)} windows, {ev.subject.nunique()} people, our timing, "
           f"classes {list(m.classes_)}, scikit-learn {sklearn.__version__}")
    print(msg)
    return msg


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true",
                    help="train on all people with our timing, save models/activity_rf_even.joblib")
    ap.add_argument("--save-only", action="store_true", help="--save without the LOSO runs")
    a = ap.parse_args(argv)
    if a.save_only:
        ev, _ = build("even")
        save(ev)
        return
    t0 = time.time()
    print("step 1/2: features for both timings (slowest part, cached after)", flush=True)
    ts, bad_ts = build("timestamps")
    ev, bad_ev = build("even")
    L = ["Activity model, leave-one-person-out, his features and RF settings",
         f"windows: his timing {len(ts)}, our timing {len(ev)} "
         f"(files dropped: {bad_ts}, {bad_ev})", "",
         f"  {'':38s} {'raw RF':>22s}   {'smoothed (3 windows)':>22s}",
         f"  {'condition':38s} {'acc':>6s} {'bal':>6s} {'F1':>6s}   {'acc':>6s} {'bal':>6s} {'F1':>6s}"]
    res = {}
    print("step 2/2: 3 conditions x 25 folds of a 300-tree forest", flush=True)
    for key, name, tr, te in (("A", "A train his timing, test his timing", ts, ts),
                              ("B", "B train his timing, test our timing", ts, ev),
                              ("C", "C train our timing, test our timing", ev, ev)):
        raw, sm, n_sub, n_win = loso(tr, te, key)
        res[key] = sm[1]
        L.append(f"  {name:38s} {raw[0]:6.3f} {raw[1]:6.3f} {raw[2]:6.3f}   "
                 f"{sm[0]:6.3f} {sm[1]:6.3f} {sm[2]:6.3f}")
    L += ["", f"  his reported numbers: raw acc 0.769 bal 0.707 F1 0.716; "
              f"smoothed acc 0.806 bal 0.740 F1 0.756"]
    gain = res["C"] - res["B"]
    verdict = ("RETRAIN on our timing (C beats B by more than 0.02 balanced)" if gain > 0.02
               else "KEEP his model (C does not beat B by more than 0.02 balanced)")
    L += [f"RESULT loso: smoothed balanced A {res['A']:.3f}, B {res['B']:.3f}, "
          f"C {res['C']:.3f} (C - B = {gain:+.3f})", f"VERDICT {verdict}"]

    if a.save:
        L.append(save(ev))
    L.append(f"time: {time.time() - t0:.0f} s")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "activity_loso_timing.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"saved {OUT / 'activity_loso_timing.txt'}")


if __name__ == "__main__":
    main()

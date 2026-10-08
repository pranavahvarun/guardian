"""
testing/train_physio_v2.py -- retrain Mithuna's physiological model without
skin temperature (Stage 6.4).

Why: her code reads the 4 Hz temperature as if it were 1 Hz (wrong moment,
1.82 C off on average), and leave-one-person-out showed temperature adds
nothing (0.848 with it read correctly, 0.846 without). Without it the module
also keeps working when there is no temperature reading.

Input: testing/outputs/stage6/physio_dalia_windows.csv, written by
testing/check_physio_dalia.py (features recomputed from the real PPG, with
the reference heart rate). The raw PPG-DaLiA files are not needed here.

Same as hers: XGBoost, 200 trees, depth 4, learning rate 0.05, subsample and
column sample 0.8, random_state 42. Same bands from the reference HR, with
honest names: LOW (<60), NORMAL (60-100), HIGH (>100) bpm.

One open choice, decided by a rule fixed before running:
    A  no class weights (as she trained)
    B  balanced class weights (LOW is only about 6% of windows)
    Pick B if it catches more slow heart rates (LOW) for new people and its
    balanced accuracy is no more than 0.02 below A. Otherwise A.

    py -m testing.train_physio_v2            LOSO for A and B, then save the pick
    py -m testing.train_physio_v2 --no-save  LOSO only

Writes models/physiological_xgb_v2.joblib and
testing/outputs/stage6/physio_v2.txt
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

WIN = ROOT / "testing" / "outputs" / "stage6" / "physio_dalia_windows.csv"
OUT = ROOT / "testing" / "outputs" / "stage6"
MODEL = ROOT / "models" / "physiological_xgb_v2.joblib"
FEATURES = ["hr", "mean_rr", "rmssd", "sdnn", "pulse_amplitude", "quality_score"]
LABELS = {"LOW": 0, "NORMAL": 1, "HIGH": 2}
PARAMS = dict(objective="multi:softprob", num_class=3, n_estimators=200, max_depth=4,
              learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42,
              eval_metric="mlogloss", tree_method="hist", verbosity=0)


def band(hr):
    return np.where(hr < 60, 0, np.where(hr <= 100, 1, 2))


def load():
    d = pd.read_csv(WIN)
    X = d[[f"f_{k}" for k in FEATURES]].copy()
    X.columns = FEATURES
    ok = X.notna().all(axis=1).values
    return X[ok].reset_index(drop=True), band(d.ref_hr.values[ok]), d.subject.values[ok]


def fit(X, y, weighted):
    from sklearn.utils.class_weight import compute_sample_weight
    from xgboost import XGBClassifier
    m = XGBClassifier(**PARAMS)
    w = compute_sample_weight("balanced", y) if weighted else None
    m.fit(X, y, sample_weight=w)
    return m


def loso(X, y, g, weighted):
    from sklearn.metrics import balanced_accuracy_score as bal
    scores, yl, pl = [], [], []
    for s in sorted(set(g)):
        tr, te = g != s, g == s
        if len(set(y[tr])) < 3:
            continue
        p = fit(X[tr], y[tr], weighted).predict(X[te])
        scores.append(bal(y[te], p))
        yl += list(y[te])
        pl += list(p)
    yl, pl = np.array(yl), np.array(pl)
    rec = {k: float(np.mean(pl[yl == v] == v)) for k, v in LABELS.items()}
    return float(np.mean(scores)), float(np.std(scores)), rec


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-save", action="store_true")
    a = ap.parse_args(argv)
    warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
    if not WIN.exists():
        print(f"Missing {WIN}\nRun testing.check_physio_dalia first.")
        sys.exit(1)
    X, y, g = load()
    L = ["Physiological model v2: no skin temperature, leave-one-person-out",
         f"{len(X)} windows, {len(set(g))} people; bands: "
         + ", ".join(f"{k} {int((y == v).sum())}" for k, v in LABELS.items()), "",
         f"  {'variant':32s} {'balanced':>8s} {'sd':>6s} {'LOW':>6s} {'NORMAL':>7s} {'HIGH':>6s}"]
    res = {}
    for key, name, w in (("A", "A no class weights (hers)", False),
                         ("B", "B balanced class weights", True)):
        print(f"  running {name} ...", flush=True)
        res[key] = loso(X, y, g, w)
        b, sd, r = res[key]
        L.append(f"  {name:32s} {b:8.3f} {sd:6.3f} {r['LOW']:6.0%} {r['NORMAL']:7.0%} {r['HIGH']:6.0%}")
    L.append("  (LOW / NORMAL / HIGH = share of each band caught, people never trained on)")
    pick = "B" if (res["B"][2]["LOW"] > res["A"][2]["LOW"]
                   and res["B"][0] >= res["A"][0] - 0.02) else "A"
    L.append(f"RESULT v2 loso: A balanced {res['A'][0]:.3f} LOW {res['A'][2]['LOW']:.0%}; "
             f"B balanced {res['B'][0]:.3f} LOW {res['B'][2]['LOW']:.0%}")
    L.append(f"VERDICT use {pick} (rule: B if it catches more LOW and loses at most 0.02 balanced)")
    if not a.no_save:
        import joblib
        import xgboost
        m = fit(X, y, pick == "B")
        joblib.dump({"model": m, "features": FEATURES, "label_map": LABELS,
                     "note": f"v2: no skin temperature, variant {pick}, all {len(set(g))} "
                             f"PPG-DaLiA people, xgboost {xgboost.__version__}"}, MODEL)
        L.append(f"saved {MODEL.name}: variant {pick}, {len(X)} windows, all {len(set(g))} people")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "physio_v2.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()

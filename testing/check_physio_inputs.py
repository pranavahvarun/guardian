"""
testing/check_physio_inputs.py -- what does Mithuna's physiological model do
inside our pipeline, and what did it learn? (Stage 6.4)

A  Heart-rate sweep. Synthetic PPG (simulator/ppg_synth.py, 100 Hz, constant
   heart rate) through the real wrapper (streams/physiological/stream.py) at
   40 to 160 bpm. Shows the estimated HR, the band the model picks, and the
   wrapper's quality. Synthetic, so it checks the plumbing and the model's
   behaviour, not accuracy on real PPG.

B  Her training table (testing/data/dalia_features_mithuna.csv, from her repo
   data/dalia_features.csv: 8612 PPG-DaLiA windows, 15 people). Shows:
   - how often a LOW estimated HR came with a reference HR above 60
   - the range of her quality score
   - leave-one-person-out with her XGBoost settings, three feature sets
     (all 7, without skin temperature, HR only), plus plain 60/100 bpm
     thresholds on her estimated HR as a no-model baseline.
   Every fold uses balanced class weights, so these numbers are comparable
   with each other, not with her reported 77.2%.

    py -m testing.check_physio_inputs
    py -m testing.check_physio_inputs --skip-loso

Writes testing/outputs/stage6/physio_inputs.txt
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
from simulator.ppg_synth import PPG_FS, at_times  # noqa: E402
from streams.physiological import PhysiologicalStream  # noqa: E402

OUT = ROOT / "testing" / "outputs" / "stage6"
TABLE = ROOT / "testing" / "data" / "dalia_features_mithuna.csv"
FEATS = ["hr", "mean_rr", "rmssd", "sdnn", "pulse_amplitude", "skin_temperature",
         "quality_score"]
LABELS = {"Normal": 0, "Elevated": 1, "Abnormal": 2}      # her names
NAMES = {"Normal": "LOW <60", "Elevated": "NORMAL 60-100", "Abnormal": "HIGH >100"}


def sweep(L):
    s = PhysiologicalStream.load()
    L += ["A  HR SWEEP, synthetic PPG at 100 Hz, 30 s buffer, skin temp 33 C",
          f"   {'true':>5s} {'est':>6s} {'peak':>6s} {'spec':>6s}  {'band':7s} "
          f"{'P(LOW)':>7s} {'P(NORM)':>7s} {'P(HIGH)':>7s} {'quality':>7s}"]
    low_hits = 0
    for hr in (40, 45, 50, 55, 58, 62, 70, 80, 90, 98, 102, 110, 130, 160):
        s.reset()
        for i in range(30):
            t = i * 1.24
            s.score({"t": t, "ppg": at_times(t + np.arange(int(2.5 * PPG_FS)) / PPG_FS, hr),
                     "ppg_fs": PPG_FS, "temp": 33.0})
        p = s.last_probs

        def f(v):
            return "  -   " if v is None else f"{v:6.1f}"
        L.append(f"   {hr:5d} {f(s.last_hr)} {f(s.last_hr_peak)} {f(s.last_hr_spectral)}  "
                 f"{s.last_state:7s} {p.get('LOW', 0):7.3f} {p.get('NORMAL', 0):7.3f} "
                 f"{p.get('HIGH', 0):7.3f} {s.last_quality:7.2f}")
        low_hits += hr < 60 and s.last_state == "LOW"
    n_low = sum(h < 60 for h in (40, 45, 50, 55, 58))
    L.append(f"RESULT sweep: LOW band picked for {low_hits} of {n_low} heart rates below 60 bpm")
    L.append("")


def table(L, loso=True):
    if not TABLE.exists():
        L.append(f"B  no training table at {TABLE}")
        return
    d = pd.read_csv(TABLE)
    y = d.state.map(LABELS).values
    low = d[d.hr < 60]
    L += [f"B  HER TRAINING TABLE: {len(d)} windows, {d.subject.nunique()} people",
          "   band counts: " + ", ".join(f"{NAMES[k]} {int((d.state == k).sum())}"
                                         for k in LABELS),
          f"   estimated HR below 60: {len(low)} windows; reference HR above 60 in "
          f"{(low.reference_hr > 60).mean():.0%} of them, above 100 in "
          f"{(low.reference_hr > 100).mean():.0%}",
          f"   her HR error: mean {np.abs(d.hr - d.reference_hr).mean():.1f} bpm, median "
          f"{np.abs(d.hr - d.reference_hr).median():.1f} bpm",
          f"   her quality score: min {d.quality_score.min():.2f}, "
          f"{(d.quality_score < 0.7).mean():.0%} of windows below GOOD (0.70)"]
    L.append(f"RESULT table: low estimated HR came with reference above 60 in "
             f"{(low.reference_hr > 60).mean():.0%}; quality score never below "
             f"{d.quality_score.min():.2f}")
    if not loso:
        L.append("")
        return
    from sklearn.metrics import balanced_accuracy_score as bal
    from sklearn.utils.class_weight import compute_sample_weight
    from xgboost import XGBClassifier
    warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")

    def run(cols):
        out = []
        for s in sorted(d.subject.unique()):
            tr, te = (d.subject != s).values, (d.subject == s).values
            m = XGBClassifier(objective="multi:softprob", n_estimators=200, max_depth=4,
                              learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                              random_state=42, tree_method="hist", verbosity=0)
            m.fit(d.loc[tr, cols], y[tr], sample_weight=compute_sample_weight("balanced", y[tr]))
            out.append(bal(y[te], m.predict(d.loc[te, cols])))
        return np.mean(out), np.std(out)

    def thr(h):
        return np.where(h < 60, 0, np.where(h <= 100, 1, 2))
    base = [bal(y[(d.subject == s).values], thr(d.hr[d.subject == s].values))
            for s in sorted(d.subject.unique())]
    L += ["   leave-one-person-out, balanced accuracy (mean, sd over 15 people):",
          f"     60/100 bpm thresholds on her estimated HR (no model)  "
          f"{np.mean(base):.3f}  {np.std(base):.3f}"]
    res = {}
    for name, cols in (("all 7 features (as deployed)", FEATS),
                       ("without skin temperature", [c for c in FEATS if c != "skin_temperature"]),
                       ("HR only", ["hr"])):
        mu, sd = run(cols)
        res[name] = mu
        L.append(f"     {name:52s} {mu:.3f}  {sd:.3f}")
    L.append(f"RESULT loso: thresholds {np.mean(base):.3f}, all 7 "
             f"{res['all 7 features (as deployed)']:.3f}, no temperature "
             f"{res['without skin temperature']:.3f}, HR only {res['HR only']:.3f}")
    L.append("")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-loso", action="store_true")
    a = ap.parse_args(argv)
    L = ["Physiological module input check (Stage 6.4)", ""]
    sweep(L)
    table(L, loso=not a.skip_loso)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "physio_inputs.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"saved {OUT / 'physio_inputs.txt'}")


if __name__ == "__main__":
    main()

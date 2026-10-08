"""
testing/check_physio_dalia.py -- Mithuna's physiological module on REAL wrist
PPG (PPG-DaLiA), through our wrapper (Stage 6.4, real-data test).

Needs the raw files (not in git):
    data/raw/PPG-DaLiA/PPG_FieldStudy/S1/S1.pkl ... S15/S15.pkl

Windows are 30 s, step 15 s, as in her training. For each window:
reference HR (ECG-based, from the dataset), her estimated HR and her two
sub-estimates, our quality (agreement rule), her quality score, the band
her model picks, and how much the wrist moved (std of |acc| in g; below
0.03 g counts as still, the same limit the motion module uses).

1  HR error, still vs moving.
2  Quality rule. Decided before running: KEEP the agreement rule if the
   windows it trusts (quality >= 0.5) have at most half the HR error of the
   ones it rejects, and it trusts at least half of all windows.
3  Slow heart rate: windows with reference HR below 60, what the estimate
   and the model said.
4  Temperature: her training read the 4 Hz temperature as if 1 Hz. Shows how
   far off that was, then leave-one-person-out with her settings: features
   with the temperature read correctly, and without temperature.
   (Her deployed model was trained on all 15 people, so its band accuracy
   here is in-sample and only printed for reference.)

    py -m testing.check_physio_dalia
    py -m testing.check_physio_dalia --subjects S1 S2      (quick try)

Writes testing/outputs/stage6/physio_dalia.txt and physio_dalia_windows.csv
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from streams.physiological import PhysiologicalStream  # noqa: E402
from streams.physiological.features import extract_features  # noqa: E402
from streams.physiological.ppg import preprocess_ppg  # noqa: E402
from streams.physiological.quality import assess_quality  # noqa: E402

DALIA = ROOT / "data" / "raw" / "PPG-DaLiA" / "PPG_FieldStudy"
OUT = ROOT / "testing" / "outputs" / "stage6"
FS_BVP, FS_TEMP, FS_ACC = 64, 4, 32       # wrist rates in PPG-DaLiA
WIN_S, STEP_S = 30, 15                    # her training windows
LABEL_WIN_S, LABEL_STEP_S = 8, 2          # reference HR: 8 s windows every 2 s
STILL_G = 0.03                            # motion module's movement limit
FEATS = ["hr", "mean_rr", "rmssd", "sdnn", "pulse_amplitude", "skin_temperature",
         "quality_score"]
BANDS = {"LOW": 0, "NORMAL": 1, "HIGH": 2}


def band(hr):
    return np.where(hr < 60, "LOW", np.where(hr <= 100, "NORMAL", "HIGH"))


def load(path):
    with open(path, "rb") as fh:
        d = pickle.load(fh, encoding="latin1")
    w = d["signal"]["wrist"]
    bvp = np.asarray(w["BVP"], float).reshape(-1)
    temp = np.asarray(w["TEMP"], float).reshape(-1)
    acc = np.asarray(w["ACC"], float).reshape(-1, 3)
    ref = np.asarray(d["label"], float).reshape(-1)
    mag = np.linalg.norm(acc, axis=1)
    med = float(np.median(mag))
    scale = 64.0 if 30 < med < 100 else (9.80665 if 5 < med < 15 else 1.0)
    return bvp, temp, mag / scale, ref, med, scale


def windows(path, stream):
    bvp, temp, acc_g, ref, acc_med, acc_scale = load(path)
    n, step = WIN_S * FS_BVP, STEP_S * FS_BVP
    rows = []
    for start in range(0, len(bvp) - n + 1, step):
        t0 = start / FS_BVP
        t1 = t0 + WIN_S
        i0 = int(np.ceil(t0 / LABEL_STEP_S))
        i1 = int(np.floor((t1 - LABEL_WIN_S) / LABEL_STEP_S))
        r = ref[i0:i1 + 1] if i1 >= i0 else []
        if len(r) == 0:
            continue
        tw = temp[int(t0 * FS_TEMP):int(t1 * FS_TEMP)]
        tw_her = temp[int(t0):int(t1)]                       # her 1 Hz indexing
        aw = acc_g[int(t0 * FS_ACC):int(t1 * FS_ACC)]
        seg = bvp[start:start + n]
        temp_ok = float(np.mean(tw)) if len(tw) else None
        try:
            stream.assess(seg, temp_ok)
        except Exception:                                    # noqa: BLE001
            continue
        try:
            clean = preprocess_ppg(seg, FS_BVP)
            q = assess_quality(clean, FS_BVP)
            f = extract_features(clean, FS_BVP, skin_temperature=temp_ok,
                                 quality_score=q["quality_score"])
        except Exception:                                    # noqa: BLE001
            f = {}
        rows.append(dict(
            subject=path.stem, t=t0, ref_hr=float(np.median(r)),
            hr=stream.last_hr, hr_peak=stream.last_hr_peak, hr_spec=stream.last_hr_spectral,
            quality=stream.last_quality, her_quality=stream.last_ppg_quality,
            state=stream.last_state, p_low=stream.last_probs.get("LOW"),
            p_normal=stream.last_probs.get("NORMAL"), p_high=stream.last_probs.get("HIGH"),
            acc_std_g=float(np.std(aw)) if len(aw) else np.nan,
            temp_ok=temp_ok, temp_her=float(np.mean(tw_her)) if len(tw_her) else np.nan,
            **{f"f_{k}": f.get(k) for k in FEATS}))
    return rows, acc_med, acc_scale


def _mae(x):
    return float(np.nanmean(np.abs(x))) if len(x) else float("nan")


def report(D, L):
    D = D.copy()
    D["err"] = D.hr - D.ref_hr
    D["still"] = D.acc_std_g < STILL_G
    ok = D[D.hr.notna()]
    L += [f"1  HR ERROR ({len(ok)} windows with an estimate, of {len(D)})",
          f"   all     mean abs error {_mae(ok.err):5.1f} bpm, median {np.nanmedian(np.abs(ok.err)):4.1f}, "
          f"within 5 bpm {np.mean(np.abs(ok.err) <= 5):.0%}"]
    for name, part in (("still", ok[ok.still]), ("moving", ok[~ok.still])):
        L.append(f"   {name:7s} mean abs error {_mae(part.err):5.1f} bpm, median "
                 f"{np.nanmedian(np.abs(part.err)) if len(part) else float('nan'):4.1f}, "
                 f"within 5 bpm {np.mean(np.abs(part.err) <= 5) if len(part) else float('nan'):.0%}"
                 f"  ({len(part)} windows)")
    L.append(f"RESULT hr: mean abs error {_mae(ok.err):.1f} bpm (still {_mae(ok[ok.still].err):.1f}, "
             f"moving {_mae(ok[~ok.still].err):.1f})")
    L.append("")

    keep = ok[ok.quality >= 0.5]
    drop = ok[ok.quality < 0.5]
    frac = len(keep) / max(len(ok), 1)
    m_k, m_d = _mae(keep.err), _mae(drop.err)
    her_low = ok[ok.her_quality < 0.7]
    verdict = "KEEP" if (frac >= 0.5 and m_k <= 0.5 * m_d) else "CHANGE"
    L += ["2  QUALITY RULE (agreement of peak and spectral HR)",
          f"   trusted (q >= 0.5): {len(keep)} windows ({frac:.0%}), mean abs error {m_k:.1f} bpm",
          f"   rejected (q < 0.5): {len(drop)} windows, mean abs error {m_d:.1f} bpm",
          f"   her quality score below 0.70 in {len(her_low)} windows "
          f"(min {ok.her_quality.min():.2f})",
          f"RESULT quality: trusted {frac:.0%} with error {m_k:.1f} bpm vs rejected {m_d:.1f} bpm",
          f"VERDICT {verdict} the agreement rule "
          "(rule: trusted error at most half of rejected, trusted at least half)", ""]

    low = D[D.ref_hr < 60]
    L += [f"3  SLOW HEART RATE: {len(low)} windows with reference below 60 bpm"]
    if len(low):
        L.append(f"   estimate below 60 in {np.mean(low.hr < 60):.0%}; model said "
                 + ", ".join(f"{k} {np.mean(low.state == k):.0%}" for k in BANDS))
    L.append(f"RESULT low: {len(low)} windows below 60 bpm; model said LOW in "
             f"{np.mean(low.state == 'LOW') if len(low) else float('nan'):.0%}")
    sure = D[D.state.isin(list(BANDS))]
    if len(sure):
        from sklearn.metrics import balanced_accuracy_score as bal
        L.append(f"   (her deployed model vs reference band, IN-SAMPLE: balanced "
                 f"{bal(band(sure.ref_hr), sure.state):.3f} on {len(sure)} windows)")
    L.append("")

    dt = (D.temp_her - D.temp_ok).abs()
    L += ["4  TEMPERATURE",
          f"   her indexing vs correct: mean difference {dt.mean():.2f} C, max {dt.max():.2f} C"]
    loso(D, L)


def loso(D, L):
    from sklearn.metrics import balanced_accuracy_score as bal
    from sklearn.utils.class_weight import compute_sample_weight
    from xgboost import XGBClassifier
    warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
    X = D[[f"f_{k}" for k in FEATS]].copy()
    X.columns = FEATS
    ok = X.notna().all(axis=1).values
    X, y, g = X[ok], pd.Series(band(D.ref_hr[ok])).map(BANDS).values, D.subject[ok].values
    if len(set(g)) < 3:
        L.append("   leave-one-person-out skipped (fewer than 3 people)")
        return
    out = {}
    for name, cols in (("temperature read correctly", FEATS),
                       ("without temperature", [c for c in FEATS if c != "skin_temperature"])):
        scores, skipped = [], 0
        for s in sorted(set(g)):
            tr, te = g != s, g == s
            if len(set(y[tr])) < 3:                 # a band missing from training
                skipped += 1
                continue
            m = XGBClassifier(objective="multi:softprob", n_estimators=200, max_depth=4,
                              learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                              random_state=42, tree_method="hist", verbosity=0)
            m.fit(X.loc[tr, cols], y[tr], sample_weight=compute_sample_weight("balanced", y[tr]))
            scores.append(bal(y[te], m.predict(X.loc[te, cols])))
        out[name] = (np.mean(scores), np.std(scores)) if scores else (float("nan"), float("nan"))
        L.append(f"   leave-one-person-out, {name:28s} balanced {out[name][0]:.3f} "
                 f"(sd {out[name][1]:.3f})" + (f", {skipped} folds skipped" if skipped else ""))
    L.append(f"RESULT temperature: LOSO with temperature read correctly "
             f"{out['temperature read correctly'][0]:.3f}, without "
             f"{out['without temperature'][0]:.3f} (her table, wrong temperature: see check_physio_inputs)")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="*", default=None)
    a = ap.parse_args(argv)
    files = sorted(DALIA.glob("S*/S*.pkl"), key=lambda p: int(p.stem[1:]))
    if a.subjects:
        files = [f for f in files if f.stem in set(a.subjects)]
    if not files:
        print(f"No PPG-DaLiA files under {DALIA}\nExpected S1/S1.pkl ... S15/S15.pkl")
        sys.exit(1)
    stream = PhysiologicalStream.load()
    t0, rows, L = time.time(), [], ["Physiological module on REAL PPG (PPG-DaLiA)", ""]
    for f in files:
        r, med, scale = windows(f, stream)
        rows += r
        print(f"  {f.stem}: {len(r)} windows (wrist acc median {med:.1f}, divided by {scale:g} -> g)",
              flush=True)
    D = pd.DataFrame(rows)
    L.append(f"{len(files)} people, {len(D)} windows of {WIN_S} s (step {STEP_S} s)")
    L.append("")
    report(D, L)
    L.append(f"time: {time.time() - t0:.0f} s")
    OUT.mkdir(parents=True, exist_ok=True)
    D.to_csv(OUT / "physio_dalia_windows.csv", index=False)
    (OUT / "physio_dalia.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"saved {OUT / 'physio_dalia.txt'}")


if __name__ == "__main__":
    main()

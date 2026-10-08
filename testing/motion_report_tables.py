"""
testing/motion_report_tables.py -- terminal tables for the motion module
report (Stage 6.5). Reads SAVED results only; nothing is retrained.

    py -m testing.motion_report_tables ss1    which model (6 compared, then top 3 on FallAllD)
    py -m testing.motion_report_tables ss2    hyperparameter tuning of the chosen model
    py -m testing.motion_report_tables ss4    performance on the same dataset (new people)
    py -m testing.motion_report_tables ss5    performance on unseen datasets

Sources (all in testing/outputs/):
    compare_first_run_6models.csv (the "First run (6 models)" sheet of
    Motion_Model_Selection.xlsx, saved as CSV), compare_summary.csv, fallalld_selection.csv,
    stage35/step3_single.csv, stage35/tune_weda_sweep.csv, tune_weda_fine.csv,
    tune_weda_current.json, tune_weda_pick.json, stage35/step5_matrix.csv,
    stage35/step6_test.csv, stage4/e3_post_window.csv, stage4/final_config.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
O = ROOT / "testing" / "outputs"
W = 78


def head(title, sub=None):
    print("=" * W)
    print(title)
    if sub:
        print(sub)
    print("=" * W)


def rule():
    print("-" * W)


# ---------------------------------------------------------------------------
def ss1():
    head("MOTION MODULE - CHOOSING THE MODEL",
         "Step 1: five models plus the gate alone, UMAFall,\n"
         "leave-one-person-out (15 people)")
    try:
        rows = pd.read_csv(O / "compare_first_run_6models.csv")    # copy of the Excel sheet
        print(f"  {'model':24s} {'AUC':>6s} {'caught at':>10s} {'false':>9s} {'ms per':>8s}")
        print(f"  {'':24s} {'gated':>6s} {'95% spec':>10s} {'alarms/h':>9s} {'segment':>8s}")
        rule()
        for _, r in rows.iterrows():
            print(f"  {str(r['Model']):24s} {float(r['AUC (gated)']):6.3f} "
                  f"{float(r['Sens. at 95% spec.']):10.3f} {float(r['False alarms / hour']):9.1f} "
                  f"{float(r['ms per segment']):8.1f}")
        rule()
        print("  The two deep models (CNN-LSTM, 1D-CNN) caught fewer falls at 95%")
        print("  specificity, raised 28-40 false alarms per hour and were the slowest.")
        print("  (This first run was before a gyroscope unit fix; the top three were")
        print("  re-run after it, below.)")
    except Exception as e:                                       # noqa: BLE001
        print(f"  could not read compare_first_run_6models.csv ({e})")
    print()
    print("Step 2: top three re-run after the fix (UMAFall, same protocol)")
    c = pd.read_csv(O / "compare_summary.csv")
    names = {"svm": "SVM", "xgb": "XGBoost", "rf": "Random Forest", "threshold": "Gate only"}
    rule()
    for _, r in c.iterrows():
        print(f"  {names.get(r.model, r.model):24s} AUC {r.auc_gated:.3f}   "
              f"false alarms/h {r.false_alarms_per_hour:5.1f}   {r.ms_per_segment:5.1f} ms")
    rule()
    print()
    print("Step 3: decided on FallAllD (different lab, people and sensor), never")
    print("trained on. Rule fixed before running: highest gated AUC on the")
    print("selection half.")
    f = pd.read_csv(O / "fallalld_selection.csv")
    f = f[(f.half == "select") & (f.model != "threshold")]
    rule()
    print(f"  {'model':24s} {'AUC, native rate':>17s} {'AUC, 20 Hz path':>16s}")
    for m in ("xgb", "rf", "svm"):
        a = f[(f.model == m) & (f.path == "native")].auc_gated.iloc[0]
        b = f[(f.model == m) & (f.path == "matched")].auc_gated.iloc[0]
        print(f"  {names[m]:24s} {a:17.3f} {b:16.3f}")
    rule()
    print("  CHOSEN: XGBoost - first on FallAllD both ways, and the fastest of the")
    print("  three (about 1 ms per segment).")


# ---------------------------------------------------------------------------
def ss2():
    cur = json.loads((O / "stage35" / "tune_weda_current.json").read_text())
    pick = json.loads((O / "stage35" / "tune_weda_pick.json").read_text())
    head("MOTION MODULE - TUNING XGBOOST (trained on WEDA-FALL)",
         "Score = gated AUC: WEDA new people (LOSO) and the mean on two datasets\n"
         "never trained on (UMAFall dev, FallAllD dev)")
    s = pd.read_csv(O / "stage35" / "step3_single.csv")
    s = s[s.train == "wedafall"]
    print(f"Round 1: {len(s)} settings (depth 2/3/4 x trees 50/100/300 x lambda 1/10)")
    print(f"  WEDA new people {s.own_loso.min():.3f}-{s.own_loso.max():.3f}   "
          f"unseen datasets {s.cross_mean.min():.3f}-{s.cross_mean.max():.3f}")
    b = s.sort_values("cross_mean", ascending=False).iloc[0]
    print(f"  picked: depth {b.max_depth}, {b.n_estimators} trees, lambda {b.reg_lambda:g}"
          f"  (WEDA {b.own_loso:.3f}, unseen {b.cross_mean:.3f})")
    print()
    sw = pd.read_csv(O / "stage35" / "tune_weda_sweep.csv")
    print(f"Round 2: sweep, {len(sw)} settings - mean AUC on unseen datasets")
    piv = sw.pivot_table(index="max_depth", columns="n_estimators", values="unseen")
    rule()
    print("  depth \\ trees " + "".join(f"{c:>8d}" for c in piv.columns))
    for d, r in piv.iterrows():
        print(f"  {d:>13d} " + "".join(f"{v:8.3f}" for v in r.values))
    rule()
    print("  Too simple underfits (depth 1, 25 trees). Depth 2-5 with 100-400 trees")
    print("  all sit near 0.80-0.815: a plateau.")
    print()
    fi = pd.read_csv(O / "stage35" / "tune_weda_fine.csv")
    print(f"Round 3: fine search, {len(fi)} settings (depth, trees, learning rate,")
    print("  min child weight, lambda, subsampling). Best five:")
    rule()
    print(f"  {'depth':>5s} {'trees':>5s} {'lr':>5s} {'mcw':>4s} {'lambda':>6s} {'subs':>5s}"
          f" {'WEDA':>7s} {'unseen':>7s}")
    for _, r in fi.sort_values("unseen", ascending=False).head(5).iterrows():
        print(f"  {int(r.max_depth):5d} {int(r.n_estimators):5d} {r.learning_rate:5.2f} "
              f"{r.min_child_weight:4.0f} {r.reg_lambda:6.0f} {r.subsample:5.1f}"
              f" {r.weda:7.3f} {r.unseen:7.3f}")
    print("  current setting:")
    print(f"  {cur['max_depth']:5d} {cur['n_estimators']:5d} {cur['learning_rate']:5.2f} "
          f"{cur['min_child_weight']:4.0f} {cur['reg_lambda']:6.0f} {cur['subsample']:5.1f}"
          f" {cur['weda']:7.3f} {cur['unseen']:7.3f}")
    rule()
    print("  Rule fixed before running: switch only if the unseen mean rises by")
    print(f"  0.01 or more. Gain {pick['gain']:+.3f} -> KEEP the current setting.")
    cfg = json.loads((O / "stage4" / "final_config.json").read_text())
    p = cfg["params"]
    print()
    print(f"FINAL: depth {p['max_depth']}, {p['n_estimators']} trees, learning rate "
          f"{p['learning_rate']}, lambda {p['reg_lambda']:g},")
    print(f"  subsample {p['subsample']}, {cfg['n_segments']} training segments "
          f"({cfg['n_falls']} falls), 18 features")


# ---------------------------------------------------------------------------
def ss4():
    t = pd.read_csv(O / "stage35" / "step6_test.csv")
    r = t[(t.model.str.startswith("WEDA")) & (t.test == "WEDA-FALL")].iloc[0]
    head("MOTION MODULE - SAME DATASET, NEW PEOPLE (WEDA-FALL)",
         "One-time test on 7 locked people (U04 U07 U08 U10 young,\n"
         "U21 U22 U29 aged 77-95), never used for any choice")
    falls, adls = int(r.falls), int(r.adls)
    tp = int(round(r.falls_caught * falls))
    print(f"  falls {falls}, daily activities {adls}, threshold {r.threshold:.3f}")
    print("  (threshold: 95% of development daily activities score below it)")
    rule()
    print(f"  gated AUC                      {r.auc_gated:.3f}")
    print(f"  AUC, all segments              {r.auc_all:.3f}")
    print(f"  falls caught                   {r.falls_caught:.0%}  ({tp} of {falls})")
    print(f"  false alarms per hour          {r.false_alarms_per_hour:.1f}")
    print(f"  faint-type falls caught        {r.faint_caught:.1%}")
    print(f"  elderly daily activities       {r.elder_adl_false_alarms} false alarms")
    rule()
    if r.false_alarms_per_hour == 0:
        print("  confusion at the threshold        predicted fall   predicted not")
        print(f"    real fall                       {tp:12d} {falls - tp:15d}")
        print(f"    daily activity                  {0:12d} {adls:15d}")
        rule()
    e3 = pd.read_csv(O / "stage4" / "e3_post_window.csv").set_index("after_s")
    print("  Model in use now looks 3 s after the impact (Stage 4), not 1.5 s.")
    print("  The locked test was spent on the 1.5 s version, so the 3 s model has")
    print("  development numbers only (WEDA leave-one-person-out, gated AUC):")
    print(f"    1.5 s  {e3.loc[1.5, 'weda']:.3f}    3.0 s  {e3.loc[3.0, 'weda']:.3f}")
    print("  WEDA falls were staged onto a mattress by young volunteers.")


# ---------------------------------------------------------------------------
def ss5():
    t = pd.read_csv(O / "stage35" / "step6_test.csv")
    w = t[t.model.str.startswith("WEDA")]
    head("MOTION MODULE - DATASETS NEVER TRAINED ON",
         "Trained on WEDA-FALL only, then scored on other labs' data")
    print("One-time test on locked people (1.5 s version):")
    rule()
    print(f"  {'test set':12s} {'falls':>6s} {'ADLs':>6s} {'gated AUC':>10s} {'caught':>8s}"
          f" {'false alarms/h':>15s}")
    for _, r in w.iterrows():
        tag = "" if r.seen == "unseen dataset" else "  (same dataset)"
        print(f"  {r.test:12s} {int(r.falls):6d} {int(r.adls):6d} {r.auc_gated:10.3f} "
              f"{r.falls_caught:8.0%} {r.false_alarms_per_hour:15.1f}{tag}")
    rule()
    print("  Threshold 0.886 set on WEDA. On other devices it turns cautious: few")
    print("  false alarms, many falls missed. The real band needs its own threshold.")
    print()
    m = pd.read_csv(O / "stage35" / "step5_matrix.csv")
    names = {"umafall": "UMAFall", "fallalld": "FallAllD", "wedafall": "WEDA-FALL"}
    print("Why WEDA-FALL: each dataset trained alone, scored on the other two (dev):")
    rule()
    print(f"  {'trained on':12s} {'UMAFall':>9s} {'FallAllD':>9s} {'WEDA':>9s} {'mean unseen':>12s}")
    for _, r in m[m.train.isin(list(names))].iterrows():
        cell = {k: (f"({r[k]:.3f})" if k == r.train else f"{r[k]:.3f}")
                for k in ("umafall", "fallalld", "wedafall")}
        print(f"  {names[r.train]:12s} {cell['umafall']:>9s} {cell['fallalld']:>9s} "
              f"{cell['wedafall']:>9s} {r.cross_mean:12.3f}")
    rule()
    print("  (brackets = its own dataset, new people)")
    e3 = pd.read_csv(O / "stage4" / "e3_post_window.csv").set_index("after_s")
    print(f"  Model in use (3 s after impact), dev only: UMAFall "
          f"{e3.loc[3.0, 'umafall']:.3f}, FallAllD {e3.loc[3.0, 'fallalld']:.3f}")


def main():
    cmds = {"ss1": ss1, "ss2": ss2, "ss4": ss4, "ss5": ss5}
    if len(sys.argv) != 2 or sys.argv[1] not in cmds:
        print(__doc__)
        sys.exit(1)
    pd.set_option("display.width", W)
    cmds[sys.argv[1]]()


if __name__ == "__main__":
    main()

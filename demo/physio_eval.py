"""
demo/physio_eval.py -- review output 3, RUN LIVE: the physiological module
on real wrist PPG (PPG-DaLiA, 15 people).

1. Every 30 s window of raw PPG goes through the module as the band would
   send it: heart rate, quality, band. Error against the ECG reference.
2. Leave-one-person-out with the settings in use (v2): each person is scored
   by a model trained now on the other 14, on the features just computed.

    py -m demo.physio_eval --root "D:\\MVP\\Sem VII\\PDP\\data"
    py -m demo.physio_eval --root ... --people 3      quick try on 3 people

--root: the folder holding S1..S15 (or any folder above it).
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from demo.physio import show


def _files(root):
    root = Path(root)
    files = [f for f in root.glob("S*/S*.pkl") if f.stem == f.parent.name]
    if not files:
        files = [f for f in root.rglob("S*.pkl") if f.stem == f.parent.name and f.stem[1:].isdigit()]
    return sorted(files, key=lambda p: int(p.stem[1:]))


def loso(D):
    from sklearn.metrics import balanced_accuracy_score as bal
    from testing import train_physio_v2 as tp
    X = D[[f"f_{k}" for k in tp.FEATURES]].copy()
    X.columns = tp.FEATURES
    ok = X.notna().all(axis=1).values
    X = X[ok].reset_index(drop=True)
    y = tp.band(D.ref_hr.values[ok])
    g = D.subject.values[ok]
    people = sorted(set(g), key=lambda s: int(s[1:]))
    print(f"  leave-one-person-out, {len(X)} windows, {len(people)} people:", flush=True)
    t0, scores, yl, pl = time.time(), [], [], []
    for k, s in enumerate(people, 1):
        print(f"    person {k:2d}/{len(people)}  train on the others, test on {s}", end="\r",
              flush=True)
        tr, te = g != s, g == s
        if len(set(y[tr])) < 3:
            continue
        p = tp.fit(X[tr], y[tr], True).predict(X[te])
        scores.append(bal(y[te], p))
        yl += list(y[te])
        pl += list(p)
    print(f"    {len(scores)} models trained and tested ({time.time() - t0:.0f} s)" + " " * 20)
    yl, pl = np.array(yl), np.array(pl)
    L = tp.LABELS
    return dict(windows=len(X), people=len(people),
                LOW=int((y == L["LOW"]).sum()), NORMAL=int((y == L["NORMAL"]).sum()),
                HIGH=int((y == L["HIGH"]).sum()), balanced=float(np.mean(scores)),
                low=float(np.mean(pl[yl == L["LOW"]] == L["LOW"])),
                normal=float(np.mean(pl[yl == L["NORMAL"]] == L["NORMAL"])),
                high=float(np.mean(pl[yl == L["HIGH"]] == L["HIGH"])))


def main(argv=None):
    from streams.physiological import PhysiologicalStream
    from testing import check_physio_dalia as cpd
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(cpd.DALIA))
    ap.add_argument("--people", type=int, default=None)
    a = ap.parse_args(argv)
    warnings.filterwarnings("ignore")
    files = _files(a.root)
    if a.people:
        files = files[:a.people]
    if not files:
        print(f"\n  No PPG-DaLiA files under {a.root}\n  Use --root with the folder holding S1..S15\n")
        return
    print(f"\n  PPG-DaLiA: {len(files)} people, 30 s windows every 15 s, model in use")
    stream = PhysiologicalStream.load()
    t0, rows = time.time(), []
    for f in files:
        t1 = time.time()
        r, _, _ = cpd.windows(f, stream)
        rows += [dict(x, subject=f.stem) for x in r]
        print(f"    {f.stem:4s} {len(r):4d} windows  ({time.time() - t1:.0f} s)", flush=True)
    D = pd.DataFrame(rows)
    print(f"  {len(D)} windows through the module ({time.time() - t0:.0f} s)")
    show(loso(D), D)


if __name__ == "__main__":
    main()

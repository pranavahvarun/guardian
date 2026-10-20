"""
demo/activity_eval.py -- review output 2, RUN LIVE: the activity module.

Same dataset: leave-one-person-out on WEDA-FALL daily activities (each of
the 25 people is scored by a model trained on the other 24, trained now).
Never trained on: the model in use (models/activity_rf_even.joblib) run
frozen through the band's activity code on UMAFall wrist recordings.

    py -m demo.activity_eval              features from the saved table
    py -m demo.activity_eval --from-raw   features recomputed from the raw files

Needs data/raw/WEDA-FALL and data/raw/UMAFall.
"""

from __future__ import annotations

import argparse
import tempfile
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

from demo.activity import show


def _pack(res, y, people):
    acc, bal, f1, rec = res
    y = np.asarray(y)
    return ({"accuracy": acc * 100, "balanced accuracy": bal * 100, "macro F1": f1 * 100},
            {c: (int(round(v * 100)), int((y == c).sum())) for c, v in rec.items()},
            f"{people} people")


def same(from_raw):
    from testing import activity_loso_timing as lt
    from testing.activity_report_tables import metrics
    if from_raw:
        lt.OUT = Path(tempfile.mkdtemp())             # no cache: read every raw file
    print("  WEDA-FALL: features for every 2.5 s window ...", flush=True)
    ev, _ = lt.build("even")
    X, y, g, rec = ev[lt.NAMES], ev.label.values, ev.subject.values, ev.rec.values
    subs = sorted(set(g))
    print(f"  WEDA-FALL: {len(ev)} windows, {len(subs)} people; leave-one-person-out:", flush=True)
    t0, y_all, p_all = time.time(), [], []
    for k, s in enumerate(subs, 1):
        print(f"    person {k:2d}/{len(subs)}  train on {len(subs) - 1}, test on {s}", end="\r",
              flush=True)
        tr, te = g != s, g == s
        m = lt._rf().fit(X[tr], y[tr])
        P = lt._smooth(m.predict_proba(X[te]), rec[te])
        y_all += list(y[te])
        p_all += list(m.classes_[P.argmax(1)])
    print(f"    {len(subs)} models trained and tested ({time.time() - t0:.0f} s)" + " " * 20)
    return _pack(metrics(y_all, p_all), y_all, len(subs))


def cross():
    from data import loader as uma
    from streams.activity import ActivityStream
    from testing.activity_report_tables import MODEL, WRIST_ID, metrics
    from testing.check_activity_inputs import UMA_ROOT, _run, _uma_label
    files = [f for f in sorted(UMA_ROOT.glob("*.csv")) if _uma_label(f.name)[1]]
    print(f"  UMAFall: {len(files)} daily-activity recordings, wrist sensor; "
          f"model {MODEL.name}", flush=True)
    s = ActivityStream.load(MODEL)
    t0, y, p, people = time.time(), [], [], set()
    per = defaultdict(int)
    for i, f in enumerate(files, 1):
        if i % 25 == 0 or i == len(files):
            print(f"    recording {i}/{len(files)}", end="\r", flush=True)
        act, lab = _uma_label(f.name)
        try:
            r = uma.read_umafall(f, sensor_id=WRIST_ID)
        except Exception:                                    # noqa: BLE001
            continue
        if r.gyro is None:
            continue
        pred = _run(s, r)
        y += [lab] * len(pred)
        p += pred
        people.add(r.subject)
        per[lab] += len(pred)
    print(f"    {len(y)} windows scored ({time.time() - t0:.0f} s)" + " " * 20)
    return _pack(metrics(y, p), y, len(people))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-raw", action="store_true")
    a = ap.parse_args(argv)
    warnings.filterwarnings("ignore")
    print()
    s = same(a.from_raw)
    c = cross()
    show(s, c)


if __name__ == "__main__":
    main()

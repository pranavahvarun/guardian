"""
demo/motion_eval.py -- review output 1, RUN LIVE: the motion module in use
(models/motion.joblib) on the raw recordings of the locked test people of
all three datasets. Reads the files, gates, cuts segments, computes the 18
features, scores them and works out the results. Nothing is trained.

    py -m demo.motion_eval

Needs the raw datasets under data/raw/ (same paths as testing/stage35.py).
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from demo.common import OUT, ROOT, rows
from demo.motion import show

SETS = [("wedafall", "WEDA-FALL"), ("umafall", "UMAFall"), ("fallalld", "FallAllD")]


def _root(rel):
    return ROOT.joinpath(*rel.replace("\\", "/").split("/"))


def _scanner(name, people):
    keep = set(people)
    if name == "wedafall":
        from data import loader_weda
        return lambda root: loader_weda.scan(root, subjects=people)
    if name == "fallalld":
        from data import loader_fallalld as fa

        def scan(root):
            files = sorted(Path(root).glob(f"S*_D{fa.WRIST_DEVICE}_A*_T*_A.dat"))
            for f in files:
                m = fa.parse_name(f.name)
                if m and m["subject"] in keep:
                    try:
                        yield fa.read_record(f, fa.WRIST_DEVICE, False)
                    except Exception:                         # noqa: BLE001
                        continue
        return scan
    from data.loader import scan as uma_scan
    return lambda root: (r for r in uma_scan(root) if r.subject in keep)


def main():
    from streams.motion import MotionStream, harmonise
    from testing.compare_motion import collect
    from testing.stage35 import ROOTS, _test_metrics
    warnings.filterwarnings("ignore")

    splits = json.loads((OUT / "stage35" / "splits.json").read_text())
    ms = MotionStream.load(str(ROOT / "models" / "motion.joblib"))
    thr = ms.threshold
    print(f"\n  model: models/motion.joblib (threshold {thr:.3f})")
    t_all, results = time.time(), {}
    for key, name in SETS:
        people = splits[key]["test"]
        t0 = time.time()
        print(f"  {name:10s} reading {len(people)} test people ...", end="", flush=True)
        meta, X, _ = collect(_root(ROOTS[key]), scanner=_scanner(key, people), transform=harmonise)
        if not len(meta):
            print(f"\n  no recordings found under {_root(ROOTS[key])}")
            return
        p = ms.impact_probability(X)
        r = _test_metrics(meta, p, thr)
        alarm = meta.gated.values & (p >= thr)
        r["false_alarm_count"] = int(alarm[meta.label.values == 0].sum())
        results[name] = r
        print(f" {len(meta)} recordings, {int(meta.gated.sum())} passed the 2.25 g gate, "
              f"scored ({time.time() - t0:.0f} s)", flush=True)
    print(f"  done in {time.time() - t_all:.0f} s")
    t = pd.DataFrame(results).T
    for c in ("falls", "adls", "auc_gated", "falls_caught", "false_alarms_per_hour", "faint_caught"):
        if c in t:
            t[c] = pd.to_numeric(t[c], errors="coerce")
    show(t, "locked test people, model in use", "locked test people, model in use")


if __name__ == "__main__":
    main()

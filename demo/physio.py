"""
demo/physio.py -- review output 3: the physiological module, SAVED results.
Reads testing/outputs/stage6/physio_v2.txt and physio_dalia_windows.csv.
To run the module on the data instead: py -m demo.physio_eval

    py -m demo.physio
"""

from __future__ import annotations

import re

import pandas as pd

from demo.common import OUT, need, rows, section, title


def show(loso, win):
    """loso: dict(windows, people, LOW, NORMAL, HIGH (counts), balanced,
    low, normal, high (share caught, 0-1)) or None. win: one row per 30 s
    window with hr, ref_hr, quality, acc_std_g."""
    title("3. HEART RATE MONITORING MODULE", "Is the heart rate normal, and can the reading be trusted?")
    if loso:
        section(f"SAME DATASET, NEW PEOPLE   PPG-DaLiA, {loso['people']} people, leave-one-person-out")
        rows([
            f"{'windows (30 s)':34s}{loso['windows']}   (LOW {loso['LOW']}, NORMAL {loso['NORMAL']}, "
            f"HIGH {loso['HIGH']})",
            f"{'balanced accuracy':34s}{loso['balanced']:.3f}",
            f"{'LOW (below 60) caught':34s}{loso['low']:.0%}",
            f"{'NORMAL (60 to 100) caught':34s}{loso['normal']:.0%}",
            f"{'HIGH (above 100) caught':34s}{loso['high']:.0%}",
        ])
    ok = win[win.hr.notna()].copy()
    ok["err"] = (ok.hr - ok.ref_hr).abs()
    keep, drop = ok[ok.quality >= 0.5], ok[ok.quality < 0.5]
    still = ok[ok.acc_std_g < 0.03]
    moving = ok[ok.acc_std_g >= 0.03]
    section("READING QUALITY   real wrist PPG against the ECG reference")
    rows([
        f"{'':34s}{'windows':>9s}{'heart-rate error':>20s}",
        f"{'all windows':34s}{len(ok):9d}{ok.err.mean():17.1f} bpm",
        f"{'  wrist still':34s}{len(still):9d}{still.err.mean():17.1f} bpm",
        f"{'  wrist moving':34s}{len(moving):9d}{moving.err.mean():17.1f} bpm",
        f"{'trusted by quality (0.5 or more)':34s}{len(keep):9d}{keep.err.mean():17.1f} bpm",
        f"{'rejected by quality':34s}{len(drop):9d}{drop.err.mean():17.1f} bpm",
    ])
    print()


def main():
    v2 = need(OUT / "stage6" / "physio_v2.txt", "py -m testing.train_physio_v2").read_text()
    win = pd.read_csv(need(OUT / "stage6" / "physio_dalia_windows.csv",
                           'py -m testing.check_physio_dalia --root "<data folder>"'))
    g = re.search(r"B balanced class weights\s+([\d.]+)\s+([\d.]+)\s+(\d+)%\s+(\d+)%\s+(\d+)%", v2)
    n = re.search(r"(\d+) windows, (\d+) people; bands: LOW (\d+), NORMAL (\d+), HIGH (\d+)", v2)
    loso = None
    if g and n:
        loso = dict(windows=int(n.group(1)), people=int(n.group(2)), LOW=int(n.group(3)),
                    NORMAL=int(n.group(4)), HIGH=int(n.group(5)), balanced=float(g.group(1)),
                    low=int(g.group(3)) / 100, normal=int(g.group(4)) / 100,
                    high=int(g.group(5)) / 100)
    show(loso, win)


if __name__ == "__main__":
    main()

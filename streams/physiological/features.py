"""
streams/physiological/features.py -- Mithuna's physiological module (HRV features).
Copied from github.com/Mithuna-05/guardian-physiological, src/guardian_physio/.
Original file: features/features.py. One line changed: the import of hr.py
is relative so it runs inside this repo.
"""

import numpy as np


def mean_rr(rr_intervals):
    """Mean RR interval in seconds."""
    rr = np.asarray(rr_intervals, dtype=float)

    if len(rr) == 0:
        return np.nan

    return float(np.mean(rr))


def rmssd(rr_intervals):
    """Root mean square of successive RR differences."""
    rr = np.asarray(rr_intervals, dtype=float)

    if len(rr) < 2:
        return np.nan

    differences = np.diff(rr)

    return float(np.sqrt(np.mean(differences ** 2)))


def sdnn(rr_intervals):
    """Standard deviation of RR intervals."""
    rr = np.asarray(rr_intervals, dtype=float)

    if len(rr) < 2:
        return np.nan

    return float(np.std(rr, ddof=1))


def pulse_amplitude(ppg):
    """Estimate pulse amplitude from the processed PPG signal."""
    ppg = np.asarray(ppg, dtype=float).reshape(-1)

    if len(ppg) == 0:
        return np.nan

    return float(np.percentile(ppg, 95) - np.percentile(ppg, 5))


def extract_features(
    ppg,
    fs,
    skin_temperature=None,
    quality_score=None,
):
    """
    Extract physiological features from one PPG window.

    Returns:
        dict containing HR, HRV, pulse amplitude,
        temperature and signal quality.
    """

    from .hr import (
        estimate_hr,
        calculate_rr_intervals,
    )

    hr, peaks = estimate_hr(ppg, fs)
    rr = calculate_rr_intervals(peaks, fs)

    features = {
        "hr": float(hr) if np.isfinite(hr) else np.nan,
        "mean_rr": mean_rr(rr),
        "rmssd": rmssd(rr),
        "sdnn": sdnn(rr),
        "pulse_amplitude": pulse_amplitude(ppg),
        "skin_temperature": (
            float(skin_temperature)
            if skin_temperature is not None
            else np.nan
        ),
        "quality_score": (
            float(quality_score)
            if quality_score is not None
            else np.nan
        ),
    }

    return features
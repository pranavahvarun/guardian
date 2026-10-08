"""
streams/physiological/quality.py -- Mithuna's physiological module (signal quality).
Copied from github.com/Mithuna-05/guardian-physiological, src/guardian_physio/.
Original file: preprocessing/quality.py. Unchanged.
"""

import numpy as np
from scipy.signal import welch


def _normalize_score(value, low, high):
    """Map a value to 0-1."""
    if value <= low:
        return 0.0
    if value >= high:
        return 1.0
    return (value - low) / (high - low)


def amplitude_score(ppg):
    """
    Measure whether the PPG has enough usable amplitude variation.
    """
    std = np.std(ppg)

    # Very small variation usually indicates poor contact
    # or a flat/near-flat signal.
    return _normalize_score(std, 0.01, 0.5)


def spectral_score(ppg, fs):
    """
    Measure how strongly the signal is concentrated
    in the physiological heart-rate band.
    """
    if len(ppg) < fs * 5:
        return 0.0

    frequencies, power = welch(
        ppg,
        fs=fs,
        nperseg=min(len(ppg), int(fs * 10)),
    )

    total_power = np.sum(power)

    if total_power <= 1e-12:
        return 0.0

    # Approximate normal adult HR range: 40–200 BPM.
    heart_band = (frequencies >= 40 / 60) & (frequencies <= 200 / 60)

    band_power = np.sum(power[heart_band])

    return float(np.clip(band_power / total_power, 0.0, 1.0))


def quality_score(ppg, fs):
    """
    Calculate a 0-1 PPG signal-quality score.

    The score combines amplitude variation and
    physiological-band spectral concentration.
    """
    ppg = np.asarray(ppg, dtype=float).reshape(-1)

    if len(ppg) < fs * 5:
        return 0.0

    if not np.all(np.isfinite(ppg)):
        return 0.0

    amp = amplitude_score(ppg)
    spectral = spectral_score(ppg, fs)

    # Equal weighting for the initial baseline.
    score = 0.5 * amp + 0.5 * spectral

    return float(np.clip(score, 0.0, 1.0))


def quality_state(score):
    """
    Convert numerical quality score into a simple state.
    """
    if score >= 0.70:
        return "GOOD"

    if score >= 0.40:
        return "ACCEPTABLE"

    return "POOR"


def assess_quality(ppg, fs):
    """
    Complete signal-quality assessment.

    Returns:
        Dictionary containing score and state.
    """
    score = quality_score(ppg, fs)

    return {
        "quality_score": score,
        "quality_state": quality_state(score),
    }
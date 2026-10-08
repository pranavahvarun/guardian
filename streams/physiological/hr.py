"""
streams/physiological/hr.py -- Mithuna's physiological module (heart rate).
Copied from github.com/Mithuna-05/guardian-physiological, src/guardian_physio/.
Original file: features/hr.py. Unchanged.
"""

import numpy as np
from scipy.signal import find_peaks, welch


def detect_peaks(ppg, fs, min_hr=40, max_hr=200):
    """Detect pulse peaks from a preprocessed PPG signal."""

    ppg = np.asarray(ppg, dtype=float).reshape(-1)

    if len(ppg) < 3:
        return np.array([], dtype=int)

    min_distance = int(fs * 60.0 / max_hr)

    # Use prominence relative to the signal rather than a fixed threshold.
    prominence = max(0.25 * np.std(ppg), 1e-6)

    peaks, properties = find_peaks(
        ppg,
        distance=min_distance,
        prominence=prominence,
    )

    if len(peaks) < 2:
        return peaks.astype(int)

    rr = np.diff(peaks) / float(fs)

    min_rr = 60.0 / max_hr
    max_rr = 60.0 / min_hr

    # Remove peaks that create impossible RR intervals.
    keep = np.ones(len(peaks), dtype=bool)

    for i, interval in enumerate(rr):
        if interval < min_rr or interval > max_rr:
            keep[i + 1] = False

    return peaks[keep].astype(int)


def peak_based_hr(ppg, fs, min_hr=40, max_hr=200):
    """Estimate HR using pulse-to-pulse intervals."""

    peaks = detect_peaks(ppg, fs, min_hr, max_hr)

    if len(peaks) < 2:
        return np.nan, peaks

    rr = np.diff(peaks) / float(fs)

    min_rr = 60.0 / max_hr
    max_rr = 60.0 / min_hr

    rr = rr[(rr >= min_rr) & (rr <= max_rr)]

    if len(rr) == 0:
        return np.nan, peaks

    hr = 60.0 / np.median(rr)

    return float(hr), peaks


def spectral_hr(ppg, fs, min_hr=40, max_hr=200):
    """
    Estimate dominant heart rate using the PPG power spectrum.
    """

    ppg = np.asarray(ppg, dtype=float).reshape(-1)

    if len(ppg) < fs * 10:
        return np.nan

    frequencies, power = welch(
        ppg,
        fs=fs,
        nperseg=min(len(ppg), int(fs * 30)),
        noverlap=None,
    )

    min_freq = min_hr / 60.0
    max_freq = max_hr / 60.0

    mask = (frequencies >= min_freq) & (frequencies <= max_freq)

    if not np.any(mask):
        return np.nan

    candidate_power = power[mask]
    candidate_freq = frequencies[mask]

    dominant_frequency = candidate_freq[np.argmax(candidate_power)]

    return float(dominant_frequency * 60.0)


def estimate_hr(ppg, fs, min_hr=40, max_hr=200):
    """
    Estimate HR using both peak-based and spectral methods.

    The two estimates are combined when they agree reasonably well.
    """

    peak_hr, peaks = peak_based_hr(
        ppg,
        fs,
        min_hr,
        max_hr,
    )

    spec_hr = spectral_hr(
        ppg,
        fs,
        min_hr,
        max_hr,
    )

    if np.isnan(peak_hr) and np.isnan(spec_hr):
        return np.nan, peaks

    if np.isnan(peak_hr):
        return spec_hr, peaks

    if np.isnan(spec_hr):
        return peak_hr, peaks

    # If both methods agree, average them.
    if abs(peak_hr - spec_hr) <= 10:
        return float((peak_hr + spec_hr) / 2.0), peaks

    # If they disagree strongly, prefer the spectral estimate
    # because wrist PPG can contain extra waveform peaks.
    return float(spec_hr), peaks


def calculate_rr_intervals(peaks, fs):
    """Convert pulse peak locations to RR intervals in seconds."""

    peaks = np.asarray(peaks, dtype=int)

    if len(peaks) < 2:
        return np.array([], dtype=float)

    return np.diff(peaks) / float(fs)
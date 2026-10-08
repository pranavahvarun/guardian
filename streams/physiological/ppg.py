"""
streams/physiological/ppg.py -- Mithuna's physiological module (PPG filtering).
Copied from github.com/Mithuna-05/guardian-physiological, src/guardian_physio/.
Original file: preprocessing/ppg.py. Unchanged.
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt


def validate_ppg(ppg):
    """
    Validate and clean basic PPG input.

    Returns:
        1-D float NumPy array
    """
    ppg = np.asarray(ppg, dtype=float).reshape(-1)

    if len(ppg) < 10:
        raise ValueError("PPG signal is too short.")

    if not np.all(np.isfinite(ppg)):
        raise ValueError("PPG signal contains NaN or infinite values.")

    return ppg


def bandpass_filter(ppg, fs, lowcut=0.5, highcut=4.0, order=3):
    """
    Apply a Butterworth band-pass filter to PPG.

    The default 0.5–4 Hz range corresponds approximately
    to 30–240 BPM.
    """
    ppg = validate_ppg(ppg)

    nyquist = fs / 2.0

    if highcut >= nyquist:
        raise ValueError(
            f"highcut ({highcut} Hz) must be below Nyquist frequency "
            f"({nyquist} Hz)."
        )

    sos = butter(
        order,
        [lowcut, highcut],
        btype="bandpass",
        fs=fs,
        output="sos",
    )

    return sosfiltfilt(sos, ppg)


def normalize_ppg(ppg):
    """Standardize PPG to zero mean and unit variance."""
    ppg = validate_ppg(ppg)

    mean = np.mean(ppg)
    std = np.std(ppg)

    if std < 1e-8:
        return np.zeros_like(ppg)

    return (ppg - mean) / std


def preprocess_ppg(ppg, fs):
    """
    Complete offline PPG preprocessing pipeline.

    Returns:
        Cleaned and normalized PPG signal.
    """
    ppg = validate_ppg(ppg)
    filtered = bandpass_filter(ppg, fs)
    normalized = normalize_ppg(filtered)

    return normalized
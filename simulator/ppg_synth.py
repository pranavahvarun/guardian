"""
simulator/ppg_synth.py -- SYNTHETIC wrist PPG for the simulator and the
acceptance check (Stage 6.4).

Not measured data. It exists so the physiological stream can run inside the
scenarios, which only have a scripted heart-rate trace. The waveform is a
textbook shape, not a model of any person:
  - one pulse per beat, timed by the heart-rate trace (phase = integral of HR)
  - systolic peak plus a smaller diastolic wave
  - slow baseline wander (breathing-like, 0.25 Hz)
  - small sensor noise
  - optional motion artefact taken from the accelerometer: wrist movement
    corrupts real PPG, so it corrupts this one too

PPG_FS = 100 Hz, the band's design target for the MAX30102 (not measured).
The physiological stream resamples to the 64 Hz its model was trained on.
"""

from __future__ import annotations

import numpy as np

PPG_FS = 100


def _pulse(phase):
    """Pulse shape for a within-beat phase in [0, 1)."""
    return (np.exp(-0.5 * ((phase - 0.15) / 0.055) ** 2)
            + 0.4 * np.exp(-0.5 * ((phase - 0.45) / 0.09) ** 2))


def from_hr(hr_bpm, fs=PPG_FS, acc_mag=None, seed=0, artefact=2.0, noise=0.03):
    """PPG samples for a heart-rate trace already sampled at fs.

    acc_mag: |acc| in g at the same rate (optional). Movement away from 1 g
    adds artefact; a still wrist adds none.
    """
    hr = np.asarray(hr_bpm, float)
    rng = np.random.default_rng(seed)
    t = np.arange(len(hr)) / fs
    phase = np.cumsum(hr / 60.0) / fs
    x = _pulse(phase % 1.0)
    x += 0.15 * np.sin(2 * np.pi * 0.25 * t)
    x += rng.normal(0, noise, len(hr))
    if acc_mag is not None:
        dev = np.abs(np.asarray(acc_mag, float) - 1.0)
        k = max(1, int(0.2 * fs))
        dev = np.convolve(dev, np.ones(k) / k, mode="same")
        wobble = np.convolve(rng.normal(0, 1, len(hr)), np.ones(k) / np.sqrt(k), mode="same")
        x += artefact * dev * wobble
    return x


def at_times(t, hr_bpm):
    """PPG at absolute times t (s) for a CONSTANT heart rate. Same t gives
    the same value, so overlapping windows agree (used by check_stream)."""
    t = np.asarray(t, float)
    return _pulse((t * hr_bpm / 60.0) % 1.0) + 0.15 * np.sin(2 * np.pi * 0.25 * t)

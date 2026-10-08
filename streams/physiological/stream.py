"""
streams/physiological/stream.py -- contract wrapper around Mithuna's
physiological model (Stage 6.4). Her code (ppg.py, quality.py, hr.py,
features.py, inference.py) does the work; this file only adapts it.

What it does, every window:
1. Takes the window's raw PPG (window["ppg"], window["ppg_fs"]) and adds the
   NEW part to a rolling buffer. Windows overlap by half, so only samples
   after the end of the buffer are added. A time gap of more than 1 s, or
   time going backwards, empties the buffer (gaps stay gaps).
2. Once the buffer holds 30 s (her model was trained on 30 s windows), takes
   the last 30 s, resamples to 64 Hz (her training rate) and runs her
   inference: filter, heart rate, HRV, her quality score, XGBoost.
3. Stores her outputs as last_* values for fusion.

Her three classes are heart-rate BANDS made from the reference heart rate in
PPG-DaLiA, not diagnoses. Her names do not match the bands, so they are
renamed here:
    her "Normal"   = reference HR below 60 bpm      -> LOW
    her "Elevated" = reference HR 60 to 100 bpm     -> NORMAL
    her "Abnormal" = reference HR above 100 bpm     -> HIGH
(60 to 100 bpm is the usual adult resting range. A high band while walking
is normal, so fusion must read this together with activity and motion.)

score   = probability of LOW or HIGH, i.e. that heart rate is outside 60 to
          100 bpm. Context for fusion, not a risk score.
quality = 0 until 30 s of usable PPG, or with no PPG, NaN, her POOR quality
          gate, or no skin temperature when her original model is loaded
          (v2 does not use temperature).
          Otherwise the agreement of her two heart-rate estimates (peak
          counting and spectrum): agree within 5 bpm -> 1.0, 25 bpm or more
          apart -> 0.0, linear between. Her own quality score is kept as
          last_ppg_quality, but on her training data it never went below
          0.80, so it cannot tell good from bad windows. The agreement rule
          was checked on real PPG (PPG-DaLiA, 15 people): windows it trusts
          are off by 3.3 bpm, windows it rejects by 38.9 bpm.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from contract import Stream  # noqa: E402

from .hr import peak_based_hr, spectral_hr  # noqa: E402
from .inference import PhysiologicalInference  # noqa: E402
from .ppg import preprocess_ppg  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
# v2 (Stage 6.4): her settings retrained without skin temperature on real
# PPG-DaLiA features (testing/train_physio_v2.py). Her original is the fallback.
DEFAULT_MODEL = ROOT / "models" / "physiological_xgb_v2.joblib"
if not DEFAULT_MODEL.exists():
    DEFAULT_MODEL = ROOT / "models" / "physiological_xgb.joblib"
MODEL_FS = 64            # her training rate (PPG-DaLiA wrist BVP)
BUFFER_S = 30.0          # her training window
GAP_S = 1.0
RENAME = {"Normal": "LOW", "Elevated": "NORMAL", "Abnormal": "HIGH"}   # her names; v2 already uses LOW/NORMAL/HIGH
AGREE_FULL, AGREE_ZERO = 5.0, 25.0     # bpm, starting values


class PhysiologicalStream(Stream):
    def __init__(self, inference: PhysiologicalInference | None = None):
        super().__init__()
        self.inference = inference
        self.version_note = None
        self.reset()

    @classmethod
    def load(cls, path=None):
        path = Path(path) if path else DEFAULT_MODEL
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            inf = PhysiologicalInference(model_path=path)
        s = cls(inf)
        if any("version" in str(w.message).lower() for w in caught):
            s.version_note = "model saved with xgboost 3.4.1; loaded with a different version"
        return s

    def reset(self):
        self._buf = np.zeros(0)
        self._fs = None
        self._t_end = None
        self._clear("NO_DATA")

    def _clear(self, state):
        self.last_state, self.last_confidence, self.last_probs = state, 0.0, {}
        self.last_hr = self.last_hr_peak = self.last_hr_spectral = None
        self.last_mean_rr = self.last_rmssd = self.last_sdnn = None
        self.last_pulse_amplitude = self.last_skin_temperature = None
        self.last_ppg_quality, self.last_quality_state = None, None
        self.last_buffer_s = 0.0 if not len(getattr(self, "_buf", [])) or not self._fs \
            else round(len(self._buf) / self._fs, 1)

    # -- buffer --------------------------------------------------------------

    def _add(self, t, ppg, fs):
        """Append only the part of this window that is newer than the buffer."""
        if self._fs is not None and fs != self._fs:
            self._buf, self._t_end = np.zeros(0), None
        self._fs = fs
        t_s = t + np.arange(len(ppg)) / fs
        if self._t_end is None or t < self._t_end - 3.0 or t > self._t_end + GAP_S:
            self._buf, new = np.zeros(0), np.ones(len(ppg), bool)
        else:
            new = t_s >= self._t_end - 0.5 / fs
        self._buf = np.concatenate([self._buf, ppg[new]])[-int(BUFFER_S * fs * 1.5):]
        self._t_end = t + len(ppg) / fs

    # -- contract ------------------------------------------------------------

    def score(self, window: dict) -> float:
        try:
            return self._score(window)
        except Exception as e:                                    # noqa: BLE001
            print(f"[physiological] score failed: {e}", file=sys.stderr)
            self._clear("ERROR")
            self.last_quality = 0.0
            return 0.0

    def _none(self, state):
        self._clear(state)
        self.last_quality = 0.0
        return 0.0

    def _score(self, window):
        ppg, fs, t = window.get("ppg"), window.get("ppg_fs"), window.get("t")
        if self.inference is None:
            return self._none("NO_MODEL")
        if ppg is None or fs is None or t is None:
            return self._none("NO_PPG")
        ppg = np.asarray(ppg, float).reshape(-1)
        if len(ppg) == 0 or not np.isfinite(ppg).all():
            self._buf, self._t_end = np.zeros(0), None    # damaged: start again
            return self._none("BAD_PPG")
        self._add(float(t), ppg, float(fs))
        have = len(self._buf) / self._fs
        if have < BUFFER_S:
            self._none("WARMING_UP")
            self.last_buffer_s = round(have, 1)
            return 0.0

        n = int(round(BUFFER_S * self._fs))
        seg = self._buf[-n:]
        grid = np.arange(int(BUFFER_S * MODEL_FS)) / MODEL_FS
        seg64 = np.interp(grid, np.arange(n) / self._fs, seg)

        return self.assess(seg64, window.get("temp"))

    def assess(self, seg64, temp):
        """Run her model on 30 s of PPG at 64 Hz and set every last_* value.
        Used every window by score(), and directly by the real-data test
        (testing/check_physio_dalia.py) so both run exactly the same steps."""
        res = self.inference.predict(seg64, fs=MODEL_FS, skin_temperature=temp)
        f = res.get("features", {})
        clean = preprocess_ppg(seg64, MODEL_FS)
        hp, _ = peak_based_hr(clean, MODEL_FS)
        hs = spectral_hr(clean, MODEL_FS)

        self._clear(RENAME.get(res["state"], res["state"]))
        self.last_buffer_s = BUFFER_S
        self.last_hr = _num(f.get("hr"))
        self.last_hr_peak, self.last_hr_spectral = _num(hp), _num(hs)
        self.last_mean_rr, self.last_rmssd = _num(f.get("mean_rr")), _num(f.get("rmssd"))
        self.last_sdnn = _num(f.get("sdnn"))
        self.last_pulse_amplitude = _num(f.get("pulse_amplitude"))
        self.last_skin_temperature = _num(f.get("skin_temperature"))
        self.last_ppg_quality = _num(res.get("quality_score"))
        self.last_quality_state = res.get("quality_state")

        if res["state"] == "UNAVAILABLE":
            needs_temp = "skin_temperature" in self.inference.features
            self.last_state = "NO_TEMP" if (temp is None and needs_temp) else "UNAVAILABLE"
            self.last_quality = 0.0
            return 0.0
        probs = {RENAME.get(k, k): round(float(v), 3)
                 for k, v in res.get("probabilities", {}).items()}
        self.last_probs = probs
        self.last_confidence = float(res["confidence"])
        if self.last_hr_peak is None or self.last_hr_spectral is None:
            agree = 0.5
        else:
            gap = abs(self.last_hr_peak - self.last_hr_spectral)
            agree = float(np.clip((AGREE_ZERO - gap) / (AGREE_ZERO - AGREE_FULL), 0, 1))
        self.last_quality = agree
        return float(np.clip(probs.get("LOW", 0.0) + probs.get("HIGH", 0.0), 0.0, 1.0))


def _num(v):
    try:
        v = float(v)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    s = PhysiologicalStream.load()
    print("loaded", DEFAULT_MODEL.name, "| note:", s.version_note or "none")

"""
streams/activity/stream.py -- contract wrapper around Sudharshan's activity
model (Stage 6.2). His model code is in engine.py and is not changed here.

What it does, every window:
1. Takes our window (acc in g, gyro in deg/s, 125 x 3 each).
2. Converts to the units his model was trained on (WEDA-FALL raw units):
   acc x 9.80665 -> m/s^2, gyro x pi/180 -> rad/s. Without this every
   feature is off by a factor of about 10 (acc) or 57 (gyro).
3. Calls his engine: 4 classes, causal mean of the last 3 windows.
   Model in use: models/activity_rf_even.joblib = his features and forest
   settings, retrained on our evenly spaced WEDA windows (Stage 6 LOSO:
   smoothed balanced accuracy 0.801, vs 0.765 for his model on our input).
4. Stores his output as last_* values, which the pipeline saves for fusion.

Classes (from his config): LOCOMOTION (walking, jogging, stairs: D01-D03),
SITTING_STANDING (sit down / stand up: D04-D05), OTHER (D06-D08),
UPPER_BODY (arm and hand tasks: D09-D11). There is no lying or "still"
class; motion's still_s covers stillness.

score   = his confidence for the chosen state (0 to 1). Activity is context
          for fusion, not a risk score and not a fall decision.
quality = 1.0 for a full window with gyro. 0.0 (and state NO_DATA) when the
          window is short, empty, has NaN/inf, or has no gyro, because his
          model needs all 6 channels. Gaps stay gaps: nothing is filled in.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from contract import Stream  # noqa: E402

from .engine import ActivityEngine  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
# Deployed: his features and settings, retrained on our loader's even timing
# (testing/activity_loso_timing.py, Stage 6). His original model is the fallback.
DEFAULT_MODEL = ROOT / "models" / "activity_rf_even.joblib"
if not DEFAULT_MODEL.exists():
    DEFAULT_MODEL = ROOT / "models" / "activity_rf.joblib"
G_MS2 = 9.80665
DEG_TO_RAD = np.pi / 180.0
N = 125
NO_DATA = "NO_DATA"
try:
    CLASSES = json.loads((ROOT / "models" / "activity_config.json").read_text())["classes"]
except Exception:                                                 # noqa: BLE001
    CLASSES = ["LOCOMOTION", "OTHER", "SITTING_STANDING", "UPPER_BODY"]


class ActivityStream(Stream):
    def __init__(self, engine: ActivityEngine | None = None):
        super().__init__()
        self.engine = engine
        self.version_note = None
        self._clear()

    def _clear(self):
        self.last_state, self.last_confidence = NO_DATA, 0.0
        self.last_probs = {}
        self.last_state_changed, self.last_previous_state = False, None
        self.last_state_duration_s = 0.0

    @classmethod
    def load(cls, path=None):
        """Load his model; the config is activity_config.json next to it."""
        path = Path(path) if path else DEFAULT_MODEL
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            eng = ActivityEngine(model_path=path,
                                 config_path=path.with_name("activity_config.json"))
        s = cls(eng)
        old = [w for w in caught if "version" in str(w.message).lower()]
        if old:
            s.version_note = ("model saved with scikit-learn 1.6.1 (his pin); "
                              "loaded with a different version")
        # Trained with n_jobs=-1: for one window at a time that starts a
        # thread pool per call. One job gives the same trees and answers.
        try:
            eng.model.n_jobs = 1
        except Exception:                                         # noqa: BLE001
            pass
        return s

    def reset(self):
        if self.engine is not None:
            self.engine.reset()
        self._clear()

    def score(self, window: dict) -> float:
        try:
            acc, gyro = window.get("acc"), window.get("gyro")
            if self.engine is None or acc is None or gyro is None:
                return self._none()
            acc, gyro = np.asarray(acc, float), np.asarray(gyro, float)
            if acc.shape != (N, 3) or gyro.shape != (N, 3):
                return self._none()
            x = np.hstack([acc * G_MS2, gyro * DEG_TO_RAD])
            if not np.isfinite(x).all():
                return self._none()
            t = window.get("t")
            out = self.engine.process(x, timestamp=None if t is None else float(t),
                                      quality=1.0)
            self.last_state = out["state"]
            self.last_confidence = float(out["confidence"])
            self.last_probs = {k: round(float(v), 3) for k, v in out["probabilities"].items()}
            self.last_state_changed = bool(out["state_changed"])
            self.last_previous_state = out["previous_state"]
            self.last_state_duration_s = float(out["state_duration_s"])
            self.last_quality = 1.0
            return float(np.clip(self.last_confidence, 0.0, 1.0))
        except Exception as e:                                    # noqa: BLE001
            print(f"[activity] score failed: {e}", file=sys.stderr)
            return self._none()

    def _none(self) -> float:
        """No usable input: say nothing (state NO_DATA), keep the 3-window
        history as it was, quality 0."""
        self.last_state, self.last_confidence, self.last_probs = NO_DATA, 0.0, {}
        self.last_state_changed = False
        self.last_quality = 0.0
        return 0.0


if __name__ == "__main__":
    s = ActivityStream.load()
    print("loaded", DEFAULT_MODEL.name, "classes", CLASSES,
          "| note:", s.version_note or "none")

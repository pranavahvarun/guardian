"""
streams/physiological/inference.py -- Mithuna's physiological module (model inference).
Copied from github.com/Mithuna-05/guardian-physiological, src/guardian_physio/.
Original file: models/physiological_inference.py. Changed: imports are
relative, the model path points at this repo's models/ folder (was D:/...),
the result also returns the class probabilities (needed for the score), and
the model input is built from the saved model's own feature list, so the v2
model without skin temperature loads too.
"""

from pathlib import Path

import joblib
import numpy as np

from .ppg import preprocess_ppg
from .quality import assess_quality
from .features import extract_features


class PhysiologicalInference:
    """
    Real-time physiological-state inference.

    Input:
        PPG window + sampling frequency + optional skin temperature

    Output:
        HR
        HRV features
        pulse amplitude
        skin temperature
        signal quality
        physiological state
        confidence
    """

    def __init__(
        self,
        model_path=Path(__file__).resolve().parents[2] / "models" / "physiological_xgb.joblib",
    ):
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {self.model_path}"
            )

        package = joblib.load(self.model_path)

        self.model = package["model"]
        self.features = package["features"]
        self.label_map = package["label_map"]

        self.reverse_label_map = {
            value: key
            for key, value in self.label_map.items()
        }

    def predict(
        self,
        ppg,
        fs=64,
        skin_temperature=None,
    ):
        # --------------------------------------------------
        # 1. Validate input
        # --------------------------------------------------

        ppg = np.asarray(
            ppg,
            dtype=float,
        ).reshape(-1)

        if len(ppg) < fs * 10:
            raise ValueError(
                "PPG window must contain at least 10 seconds of data."
            )

        if not np.all(np.isfinite(ppg)):
            raise ValueError(
                "PPG contains NaN or infinite values."
            )

        # --------------------------------------------------
        # 2. Preprocess PPG
        # --------------------------------------------------

        clean_ppg = preprocess_ppg(
            ppg,
            fs,
        )

        # --------------------------------------------------
        # 3. Signal quality
        # --------------------------------------------------

        quality = assess_quality(
            clean_ppg,
            fs,
        )

        # --------------------------------------------------
        # 4. Extract physiological features
        # --------------------------------------------------

        feature_values = extract_features(
            clean_ppg,
            fs,
            skin_temperature=skin_temperature,
            quality_score=quality["quality_score"],
        )

        # --------------------------------------------------
        # 5. Build model input
        # --------------------------------------------------

        # The model's own feature list, in its order (her model: 7 with skin
        # temperature; v2: 6 without). Was a fixed list of 7 (Stage 6.4).
        X = np.array(
            [[feature_values[name] for name in self.features]],
            dtype=float,
        )

        # --------------------------------------------------
        # 6. Check feature validity
        # --------------------------------------------------

        if not np.all(np.isfinite(X)):
            return {
                "state": "UNAVAILABLE",
                "confidence": 0.0,
                "quality_score": quality["quality_score"],
                "quality_state": quality["quality_state"],
                "features": feature_values,
            }

        # --------------------------------------------------
        # 7. Poor-quality safety gate
        # --------------------------------------------------

        if quality["quality_state"] == "POOR":
            return {
                "state": "UNAVAILABLE",
                "confidence": 0.0,
                "quality_score": quality["quality_score"],
                "quality_state": quality["quality_state"],
                "features": feature_values,
            }

        # --------------------------------------------------
        # 8. XGBoost prediction
        # --------------------------------------------------

        probabilities = self.model.predict_proba(X)[0]

        predicted_class = int(
            np.argmax(probabilities)
        )

        state = self.reverse_label_map[
            predicted_class
        ]

        confidence = float(
            probabilities[predicted_class]
        )

        # --------------------------------------------------
        # 9. Return result
        # --------------------------------------------------

        return {
            "state": state,
            "confidence": confidence,
            "probabilities": {
                self.reverse_label_map[i]: float(p)
                for i, p in enumerate(probabilities)
            },
            "quality_score": quality["quality_score"],
            "quality_state": quality["quality_state"],
            "features": feature_values,
        }
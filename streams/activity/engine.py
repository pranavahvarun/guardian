"""
streams/activity/engine.py -- Sudharshan's activity model (activity_v1).

Copied from github.com/Sudharshan-dev405/Guardian-Activity-Review-2,
src/activity.py. Only three lines changed, so it runs inside this repo:
  - class renamed ActivityStream -> ActivityEngine (the contract wrapper in
    stream.py is the ActivityStream the pipeline uses)
  - "from features import" -> "from .features import"
  - default model/config paths point at this repo's models/ folder
The model logic is his and unchanged. Input here is HIS units:
acc in m/s^2, gyro in rad/s (WEDA-FALL raw units). stream.py converts.
"""

import json
from collections import deque
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .features import calculate_features


class ActivityEngine:
    """
    Guardian Activity inference module.

    Input:
        One 2.5-second window of 50 Hz IMU data
        with shape (125, 6):

        [ax, ay, az, gx, gy, gz]

    Output:
        Activity state, probabilities, confidence,
        quality, and temporal state information.
    """

    def __init__(
        self,
        model_path=None,
        config_path=None,
        smoothing_windows=3,
    ):
        project_root = Path(__file__).resolve().parents[2]   # repo root (was src/..)

        if model_path is None:
            model_path = (
                project_root
                / "models"
                / "activity_rf.joblib"
            )

        if config_path is None:
            config_path = (
                project_root
                / "models"
                / "activity_config.json"
            )

        self.model_path = Path(model_path)
        self.config_path = Path(config_path)

        # ----------------------------------------------------
        # Load model and configuration
        # ----------------------------------------------------

        self.model = joblib.load(self.model_path)

        with open(self.config_path, "r") as f:
            self.config = json.load(f)

        self.classes = self.config["classes"]

        self.feature_names = self.config[
            "features"
        ]["names"]

        expected_smoothing = self.config[
            "temporal_smoothing"
        ]["window_count"]

        if smoothing_windows != expected_smoothing:
            raise ValueError(
                "smoothing_windows does not match "
                "the trained model configuration. "
                f"Expected {expected_smoothing}, "
                f"got {smoothing_windows}."
            )

        self.smoothing_windows = smoothing_windows

        # ----------------------------------------------------
        # Causal probability history
        # ----------------------------------------------------

        self.probability_history = deque(
            maxlen=self.smoothing_windows
        )

        # ----------------------------------------------------
        # Temporal activity state
        # ----------------------------------------------------

        self.current_state = None
        self.previous_state = None
        self.state_since_timestamp = None

    def process(
        self,
        window,
        timestamp=None,
        quality=1.0,
    ):
        """
        Process one 2.5-second IMU window.

        Parameters
        ----------
        window:
            NumPy array with shape (125, 6).

        timestamp:
            Timestamp associated with this prediction.

        quality:
            Input/sensor quality in [0, 1].

        Returns
        -------
        dict
            Guardian Activity output contract.
        """

        window = np.asarray(
            window,
            dtype=float,
        )

        # ----------------------------------------------------
        # Validate input
        # ----------------------------------------------------

        expected_shape = (
            self.config["input"]["window_samples"],
            len(self.config["input"]["channels"]),
        )

        if window.shape != expected_shape:
            raise ValueError(
                f"Expected window shape "
                f"{expected_shape}, "
                f"got {window.shape}"
            )

        if not np.isfinite(window).all():
            raise ValueError(
                "Activity window contains "
                "NaN or infinite values."
            )

        if not 0.0 <= quality <= 1.0:
            raise ValueError(
                "quality must be between 0 and 1."
            )

        # ----------------------------------------------------
        # Extract the exact 78 features
        # ----------------------------------------------------

        feature_dict = calculate_features(window)

        feature_df = pd.DataFrame(
            [feature_dict]
        )

        # ----------------------------------------------------
        # Enforce exact training feature order
        # ----------------------------------------------------

        missing_features = [
            feature
            for feature in self.feature_names
            if feature not in feature_df.columns
        ]

        if missing_features:
            raise ValueError(
                "Missing model features: "
                f"{missing_features}"
            )

        feature_df = feature_df[
            self.feature_names
        ]

        # ----------------------------------------------------
        # Model prediction
        # ----------------------------------------------------

        probabilities = (
            self.model.predict_proba(
                feature_df
            )[0]
        )

        # ----------------------------------------------------
        # Causal probability smoothing
        # ----------------------------------------------------

        self.probability_history.append(
            probabilities
        )

        smoothed_probabilities = np.mean(
            np.array(
                self.probability_history
            ),
            axis=0,
        )

        # ----------------------------------------------------
        # Determine current state
        # ----------------------------------------------------

        state_index = int(
            np.argmax(
                smoothed_probabilities
            )
        )

        state = self.classes[state_index]

        confidence = float(
            smoothed_probabilities[
                state_index
            ]
        )

        # ----------------------------------------------------
        # Temporal state tracking
        # ----------------------------------------------------

        previous_state = self.current_state

        state_changed = (
            previous_state is not None
            and state != previous_state
        )

        if state != self.current_state:
            self.previous_state = self.current_state
            self.current_state = state

            if timestamp is not None:
                self.state_since_timestamp = timestamp

        if (
            timestamp is not None
            and self.state_since_timestamp is not None
        ):
            state_duration_s = (
                timestamp
                - self.state_since_timestamp
            )
        else:
            state_duration_s = 0.0

        # ----------------------------------------------------
        # Build output
        # ----------------------------------------------------

        probability_dict = {
            class_name: float(
                smoothed_probabilities[i]
            )
            for i, class_name
            in enumerate(self.classes)
        }

        return {
            "state": state,
            "probabilities": probability_dict,
            "confidence": confidence,
            "quality": float(quality),
            "state_changed": state_changed,
            "previous_state": previous_state,
            "state_duration_s": float(
                state_duration_s
            ),
            "timestamp": timestamp,
        }

    def reset(self):
        """
        Reset temporal state.

        Call this when starting a new
        recording/session.
        """

        self.probability_history.clear()

        self.current_state = None
        self.previous_state = None
        self.state_since_timestamp = None
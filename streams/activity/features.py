import numpy as np
import pandas as pd


CHANNELS = ["ax", "ay", "az", "gx", "gy", "gz"]


def calculate_features(window):
    features = {}

    # ---------------------------------------------------------
    # 1. Existing statistical features
    # ---------------------------------------------------------

    for i, channel in enumerate(CHANNELS):
        signal = window[:, i]

        features[f"{channel}_mean"] = np.mean(signal)
        features[f"{channel}_std"] = np.std(signal)
        features[f"{channel}_min"] = np.min(signal)
        features[f"{channel}_max"] = np.max(signal)
        features[f"{channel}_range"] = np.max(signal) - np.min(signal)
        features[f"{channel}_median"] = np.median(signal)
        features[f"{channel}_rms"] = np.sqrt(np.mean(signal ** 2))
        features[f"{channel}_energy"] = np.sum(signal ** 2)
        features[f"{channel}_mean_abs_change"] = np.mean(
            np.abs(np.diff(signal))
        )

    # ---------------------------------------------------------
    # 2. Acceleration magnitude
    # ---------------------------------------------------------

    acceleration = window[:, 0:3]

    accel_magnitude = np.sqrt(
        np.sum(acceleration ** 2, axis=1)
    )

    features["acc_mag_mean"] = np.mean(accel_magnitude)
    features["acc_mag_std"] = np.std(accel_magnitude)
    features["acc_mag_min"] = np.min(accel_magnitude)
    features["acc_mag_max"] = np.max(accel_magnitude)
    features["acc_mag_range"] = (
        np.max(accel_magnitude) - np.min(accel_magnitude)
    )
    features["acc_mag_rms"] = np.sqrt(
        np.mean(accel_magnitude ** 2)
    )
    features["acc_mag_energy"] = np.sum(
        accel_magnitude ** 2
    )

    # ---------------------------------------------------------
    # 3. Gyroscope magnitude
    # ---------------------------------------------------------

    gyroscope = window[:, 3:6]

    gyro_magnitude = np.sqrt(
        np.sum(gyroscope ** 2, axis=1)
    )

    features["gyro_mag_mean"] = np.mean(gyro_magnitude)
    features["gyro_mag_std"] = np.std(gyro_magnitude)
    features["gyro_mag_min"] = np.min(gyro_magnitude)
    features["gyro_mag_max"] = np.max(gyro_magnitude)
    features["gyro_mag_range"] = (
        np.max(gyro_magnitude) - np.min(gyro_magnitude)
    )
    features["gyro_mag_rms"] = np.sqrt(
        np.mean(gyro_magnitude ** 2)
    )
    features["gyro_mag_energy"] = np.sum(
        gyro_magnitude ** 2
    )

    # =========================================================
    # 4. NEW: Dynamic / temporal features
    # =========================================================

    # Acceleration magnitude changes
    accel_diff = np.diff(accel_magnitude)

    features["acc_mag_mean_abs_change"] = np.mean(
        np.abs(accel_diff)
    )

    features["acc_mag_diff_std"] = np.std(
        accel_diff
    )

    features["acc_mag_diff_rms"] = np.sqrt(
        np.mean(accel_diff ** 2)
    )

    # Gyroscope magnitude changes
    gyro_diff = np.diff(gyro_magnitude)

    features["gyro_mag_mean_abs_change"] = np.mean(
        np.abs(gyro_diff)
    )

    features["gyro_mag_diff_std"] = np.std(
        gyro_diff
    )

    features["gyro_mag_diff_rms"] = np.sqrt(
        np.mean(gyro_diff ** 2)
    )

    # ---------------------------------------------------------
    # 5. Peak counts
    # ---------------------------------------------------------

    # Simple local-peak detector.
    # A point is considered a peak if it is greater than
    # its immediate neighbours.

    accel_peaks = (
        (accel_magnitude[1:-1] > accel_magnitude[:-2]) &
        (accel_magnitude[1:-1] > accel_magnitude[2:])
    )

    gyro_peaks = (
        (gyro_magnitude[1:-1] > gyro_magnitude[:-2]) &
        (gyro_magnitude[1:-1] > gyro_magnitude[2:])
    )

    features["acc_mag_peak_count"] = np.sum(accel_peaks)
    features["gyro_mag_peak_count"] = np.sum(gyro_peaks)

    # ---------------------------------------------------------
    # 6. Zero-crossing rates of dynamic signals
    # ---------------------------------------------------------

    accel_diff_sign = np.sign(accel_diff)
    gyro_diff_sign = np.sign(gyro_diff)

    accel_zero_crossings = np.sum(
        accel_diff_sign[1:] != accel_diff_sign[:-1]
    )

    gyro_zero_crossings = np.sum(
        gyro_diff_sign[1:] != gyro_diff_sign[:-1]
    )

    features["acc_mag_diff_zero_crossings"] = (
        accel_zero_crossings
    )

    features["gyro_mag_diff_zero_crossings"] = (
        gyro_zero_crossings
    )

    return features


def extract_features(windows):
    return pd.DataFrame([
        calculate_features(window)
        for window in windows
    ])
from __future__ import annotations

from typing import Any
import math

import numpy as np
import parselmouth


def _praat_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return np.nan

    if math.isnan(result) or math.isinf(result):
        return np.nan

    return result


def extract_praat_voice_quality_features(
    y: np.ndarray,
    sr: int,
    f0_min: float = 75.0,
    f0_max: float = 600.0,
    pitch_time_step: float = 0.01,
) -> dict[str, float]:
    features = {
        "glottal_f0_mean": np.nan,
        "glottal_f0_std": np.nan,
        "glottal_f0_min": np.nan,
        "glottal_f0_max": np.nan,
        "glottal_f0_voiced_ratio": np.nan,
        "glottal_hnr_mean": np.nan,
        "glottal_hnr_std": np.nan,
        "glottal_jitter_local": np.nan,
        "glottal_shimmer_local": np.nan,
    }

    y = np.asarray(y, dtype=np.float64)

    if y.size == 0:
        return features

    try:
        sound = parselmouth.Sound(y, sampling_frequency=sr)

        pitch = sound.to_pitch(
            time_step=pitch_time_step,
            pitch_floor=f0_min,
            pitch_ceiling=f0_max,
        )

        f0_values = pitch.selected_array["frequency"]
        voiced_f0 = f0_values[f0_values > 0]

        if voiced_f0.size > 0:
            features["glottal_f0_mean"] = float(np.mean(voiced_f0))
            features["glottal_f0_std"] = float(np.std(voiced_f0))
            features["glottal_f0_min"] = float(np.min(voiced_f0))
            features["glottal_f0_max"] = float(np.max(voiced_f0))
            features["glottal_f0_voiced_ratio"] = float(
                voiced_f0.size / max(f0_values.size, 1)
            )

        harmonicity = sound.to_harmonicity_cc(
            time_step=pitch_time_step,
            minimum_pitch=f0_min,
        )

        hnr_values = np.asarray(harmonicity.values, dtype=np.float64).flatten()
        hnr_values = hnr_values[np.isfinite(hnr_values)]
        hnr_values = hnr_values[hnr_values != -200]

        if hnr_values.size > 0:
            features["glottal_hnr_mean"] = float(np.mean(hnr_values))
            features["glottal_hnr_std"] = float(np.std(hnr_values))

        point_process = parselmouth.praat.call(
            sound,
            "To PointProcess (periodic, cc)",
            f0_min,
            f0_max,
        )

        jitter_value = parselmouth.praat.call(
            point_process,
            "Get jitter (local)",
            0.0,
            0.0,
            0.0001,
            0.02,
            1.3,
        )

        shimmer_value = parselmouth.praat.call(
            [sound, point_process],
            "Get shimmer (local)",
            0.0,
            0.0,
            0.0001,
            0.02,
            1.3,
            1.6,
        )

        features["glottal_jitter_local"] = _praat_float(jitter_value)
        features["glottal_shimmer_local"] = _praat_float(shimmer_value)

    except Exception:
        pass

    return features
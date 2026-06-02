from typing import Any

import numpy as np

from classes.vpd.features.energy_features import get_energy_area_A3
from classes.vpd.features.entropy_features import get_entropy_C2
from classes.vpd.features.harmonic_features import find_harmonics, get_top_n_harmonics
from classes.vpd.features.mfcc_features import mfcc_features
from classes.vpd.features.zcr_features import get_zcr_B3_optimized, zcr_numpy
from classes.vpd.features.glothal_features import extract_praat_voice_quality_features
from classes.vpd.vpd_feature_config import VPDFeatureConfig


class VPDFeatureExtractor:
    def __init__(self, config: VPDFeatureConfig):
        self.config = config

    def extract(self, sample) -> dict[str, Any]:
        y = np.asarray(sample.signal, dtype=np.float64)
        sr = int(sample.sr)

        features: dict[str, Any] = {
            "sample_id": sample.sample_id,
            "base": sample.base,
            "label": sample.label,
            "speaker_id": sample.speaker_id,
            "sex": sample.sex,
            "age": sample.age,
            "pathology": sample.pathology,
            "pathology_code": sample.pathology_code,
            "vowel": sample.vowel,
            "pitch": sample.pitch,
            "sr": sr,
            "duration": len(y) / sr if sr > 0 else np.nan,
        }

        features.update(self._extract_harmonics(y, sr))
        features.update(self._extract_energy_area(y))
        features.update(self._extract_entropy_c2(y))
        features.update(self._extract_zcr_b3(y))
        features.update(self._extract_mfcc(y, sr))
        features.update(self._extract_glottal_features(y, sr))

        return features

    def _extract_harmonics(self, y: np.ndarray, sr: int) -> dict[str, Any]:
        harmonic_matrix = find_harmonics(
            y=y,
            sr=sr,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
        )

        top = get_top_n_harmonics(
            harmonic_matrix=harmonic_matrix,
            top_n=self.config.top_n_harmonics,
            min_freq=self.config.harmonic_min_freq,
            max_freq=self.config.harmonic_max_freq,
        )

        features = {}

        for i in range(self.config.top_n_harmonics):
            if i < len(top):
                features[f"harmonic_{i + 1:02d}_freq"] = float(top[i, 0])
                features[f"harmonic_{i + 1:02d}_amp"] = float(top[i, 1])
            else:
                features[f"harmonic_{i + 1:02d}_freq"] = np.nan
                features[f"harmonic_{i + 1:02d}_amp"] = np.nan

        if len(top) > 0:
            features["harmonic_freq_mean"] = float(np.mean(top[:, 0]))
            features["harmonic_freq_std"] = float(np.std(top[:, 0]))
            features["harmonic_amp_mean"] = float(np.mean(top[:, 1]))
            features["harmonic_amp_std"] = float(np.std(top[:, 1]))
        else:
            features["harmonic_freq_mean"] = np.nan
            features["harmonic_freq_std"] = np.nan
            features["harmonic_amp_mean"] = np.nan
            features["harmonic_amp_std"] = np.nan

        return features

    def _extract_energy_area(self, y: np.ndarray) -> dict[str, float]:
        values = get_energy_area_A3(
            y,
            percent_step=self.config.energy_percent_steps
        )

        features = {}

        for i, value in enumerate(values, start=1):
            percent = i * self.config.energy_percent_steps
            features[f"energy_area_{percent:02d}"] = float(value)

        return features

    def _extract_entropy_c2(self, y: np.ndarray) -> dict[str, float]:
        values = get_entropy_C2(
            y,
            bins=self.config.entropy_bins,
        )

        return {
            f"entropy_c2_{i + 1:02d}" : float(value)
            for i, value in enumerate(values)
        }

    def _extract_zcr_b3(self, y: np.ndarray) -> dict[str, float]:
        values = get_zcr_B3_optimized(
            y,
            percent=self.config.zcr_percent
        )

        features = {}

        for i, value in enumerate(values, start=1):
            percent = i * self.config.zcr_percent
            features[f"zcr_b3_{percent:02d}"] = float(value)

        features["zcr_total"] = float(zcr_numpy(y))
        features["zcr_rate"] = float(zcr_numpy(y) / max(len(y) -1, 1))

        return features

    def _extract_mfcc(self, y: np.ndarray, sr: int) -> dict[str, float]:
        return mfcc_features(
            y=y,
            sr=sr,
            n_mfcc=self.config.n_mfcc,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            include_delta=self.config.include_mfcc_delta,
        )

    def _extract_glottal_features(self, y: np.ndarray, sr: int) -> dict[str, float]:
        if not self.config.include_glottal_features:
            return {}

        try:
            return extract_praat_voice_quality_features(
                y=y,
                sr=sr,
                f0_min=self.config.glottal_f0_min,
                f0_max=self.config.glottal_f0_max,
                pitch_time_step=self.config.glottal_pitch_time_step,
            )
        except Exception as exc:
            return {
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
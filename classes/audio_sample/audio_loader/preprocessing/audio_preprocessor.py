import math
from pathlib import Path
from typing import Any, Optional, cast, SupportsFloat, SupportsIndex

import numpy as np
import pandas as pd
import soundfile as sf
from math import gcd

from scipy.signal import resample_poly

from classes.audio_sample.audio_loader.preprocessing.audio_preprocess_config import AudioPreprocessConfig
from classes.audio_sample.audio_sample import AudioSample

FloatConvertible = str | bytes | bytearray | SupportsFloat | SupportsIndex


class AudioPreprocessor:
    def __init__(self, config: AudioPreprocessConfig):
        self.config = config


    def load_from_manifest_row(self, row: pd.Series) -> AudioSample:
        filepath = Path(row["filepath"])

        signal, sr = sf.read(filepath, dtype = "float32", always_2d=True)

        sample = AudioSample(
            sample_id=str(row["sample_id"]),
            base=str(row["base"]),
            filepath=filepath,
            signal=np.asarray(signal, dtype=np.float32),
            sr=int(sr),

            label=self._get_optional(row, "label"),
            speaker_id=self._get_optional(row, "speaker_id"),
            sex=self._get_optional(row, "sex"),
            age=self._get_optional_float(row, "age"),
            pathology=self._get_optional(row, "pathology"),
            pathology_code=self._get_optional(row, "pathology_code"),
            vowel=self._get_optional(row, "vowel"),
            pitch=self._get_optional(row, "pitch"),

            metadata=row.to_dict(),
        )

        return sample

    def process(self, sample: AudioSample) -> AudioSample:
        signal = sample.signal
        sr = sample.sr

        if self.config.convert_to_mono:
            signal = self.to_mono(signal)

        if self.config.remove_dc:
            signal = self.remove_dc(signal)

        if sr != self.config.target_sr:
            signal = self.resample(signal, sr, self.config.target_sr)
            sr = self.config.target_sr

        if self.config.normalize_rms:
            signal = self.normalize_rms(
                signal,
                target_dbfs=self.config.target_dbfs,
                peak_limit=self.config.peak_limit,
            )

        if self.config.center_crop:
            signal = self.center_crop_or_pad(
                signal,
                sr=sr,
                duration_sec=self.config.crop_duration_sec,
                pad_if_short=self.config.pad_if_short,
            )

        if self.config.min_duration_sec is not None:
            duration = len(signal) / sr
            if duration < self.config.min_duration_sec:
                raise ValueError(
                    f"Sample {sample.sample_id} has too short duration "
                    f"{duration:.3f}s < {self.config.min_duration_sec:.3f}s"
                )

        return AudioSample(
            sample_id=sample.sample_id,
            base=sample.base,
            filepath=sample.filepath,
            signal=signal.astype(np.float32),
            sr=sr,
            label=sample.label,
            speaker_id=sample.speaker_id,
            sex=sample.sex,
            age=sample.age,
            pathology=sample.pathology,
            pathology_code=sample.pathology_code,
            vowel=sample.vowel,
            pitch=sample.pitch,
            metadata=sample.metadata,
        )

    def process_manifest_row(self, row: pd.Series) -> AudioSample:
        sample = self.load_from_manifest_row(row)
        return self.process(sample)

    @staticmethod
    def to_mono(signal: np.ndarray) -> np.ndarray:
        if signal.ndim == 1:
            return signal

        return np.mean(signal, axis=1)

    @staticmethod
    def remove_dc(signal: np.ndarray) -> np.ndarray:
        return signal - np.mean(signal)

    @staticmethod
    def resample(signal: np.ndarray, original_sr: int, target_sr: int) -> np.ndarray:
        divisor = gcd(original_sr, target_sr)
        up = target_sr // divisor
        down = original_sr // divisor

        return resample_poly(signal, up, down).astype(np.float32)

    @staticmethod
    def normalize_rms(
        signal: np.ndarray,
        target_dbfs: float = -20.0,
        peak_limit: float = 0.99,
    ) -> np.ndarray:

        eps = 1e-12
        rms = np.sqrt(np.mean(signal ** 2))

        if rms < eps:
            return signal

        target_rms = 10 ** (target_dbfs / 20.0)
        gain = target_rms / rms

        normalized = signal * gain

        peak = float(np.max(np.abs(normalized)))
        if peak > peak_limit:
            normalized = normalized * (peak_limit / peak)

        return normalized.astype(np.float32)

    @staticmethod
    def center_crop_or_pad(
            signal: np.ndarray,
            sr: int,
            duration_sec: float,
            pad_if_short: bool = False,
    ) -> np.ndarray:

        target_len = int(duration_sec * sr)
        current_len = len(signal)

        if current_len == target_len:
            return signal

        if current_len > target_len:
            start = (current_len - target_len) // 2
            end = start + target_len
            return signal[start:end]

        if not pad_if_short:
            return signal

        pad_total = target_len - current_len
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left

        return np.pad(signal, (pad_left, pad_right), mode="constant")

    @staticmethod
    def _is_missing(value: Any) -> bool:
        if value is None:
            return True

        try:
            return bool(pd.isna(value))
        except (TypeError, ValueError):
            return False

    @classmethod
    def _get_scalar_from_row(cls, row: pd.Series, column: str) -> Any | None:
        if column not in row.index:
            return None

        value = row.loc[column]

        # May happen if columns are duplicated after merge.
        if isinstance(value, pd.Series):
            value = value.dropna()

            if value.empty:
                return None

            value = value.iloc[0]

        # Convert numpy scalars to Python scalars
        if isinstance(value, np.generic):
            value = value.item()

        if cls._is_missing(value):
            return None

        return value

    @classmethod
    def _get_optional(cls, row: pd.Series, column: str) -> Optional[str]:
        value = cls._get_scalar_from_row(row, column)

        if value is None:
            return None

        return str(value)

    @classmethod
    def _get_optional_float(cls, row: pd.Series, column: str) -> Optional[float]:
        value = cls._get_scalar_from_row(row, column)

        if value is None:
            return None

        if isinstance(value, np.generic):
            value = value.item()

        if not isinstance(value, (str, int, float, bytes, bytearray)):
            try:
                value = cast(SupportsFloat, value)
            except TypeError:
                return None

        try:
            result = float(cast(FloatConvertible, value))
        except (TypeError, ValueError):
            return None

        if math.isnan(result):
            return None

        return result




























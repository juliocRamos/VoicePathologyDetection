from math import gcd

import numpy as np

from scipy.signal import resample_poly

from classes.audio_sample.audio_loader.preprocessing.audio_preprocess_config import (
    AudioPreprocessConfig,
)
from classes.audio_sample.audio_sample import AudioSample


class AudioPreprocessor:
    def __init__(
        self,
        config: AudioPreprocessConfig,
    ):
        self.config = config

    def process(self, sample: AudioSample) -> AudioSample:
        signal = np.asarray(sample.signal)
        sr = sample.sr

        self._validate_input_signal(
            signal=signal,
            sample_rate=sr,
            sample_id=sample.sample_id,
        )

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

        self._validate_processed_signal(
            signal=signal,
            sample_rate=sr,
            sample_id=sample.sample_id,
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

    @staticmethod
    def to_mono(signal: np.ndarray) -> np.ndarray:
        signal = np.asarray(signal)

        if signal.ndim == 1:
            return signal.astype(np.float32, copy=False)

        if signal.ndim != 2:
            raise ValueError(
                f"Unsupported audio shape: {signal.shape}. "
                "Expected mono or multichannel signal."
            )

        if signal.shape[1] == 1:
            return signal[:, 0].astype(
                np.float32,
                copy=False,
            )

        return np.mean(
            signal,
            axis=1,
            dtype=np.float64,
        ).astype(np.float32)

    @staticmethod
    def remove_dc(signal: np.ndarray) -> np.ndarray:
        return signal - np.mean(signal)

    @staticmethod
    def resample(
        signal: np.ndarray,
        original_sr: int,
        target_sr: int,
    ) -> np.ndarray:
        if original_sr <= 0 or target_sr <= 0:
            raise ValueError(
                "Sample rates must be positive for resampling. "
                f"Received original_sr={original_sr}, "
                f"target_sr={target_sr}."
            )

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
        if signal.ndim != 1:
            raise ValueError(
                "center_crop_or_pad expects a mono signal, "
                f"but received shape={signal.shape}"
            )

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

        return np.pad(
            signal,
            (pad_left, pad_right),
            mode="constant",
        )

    @staticmethod
    def _validate_input_signal(
        signal: np.ndarray,
        sample_rate: int,
        sample_id: str,
    ) -> None:
        if sample_rate <= 0:
            raise ValueError(
                f"Sample {sample_id} has invalid sample rate: "
                f"{sample_rate}."
            )

        if signal.ndim not in {1, 2}:
            raise ValueError(
                f"Sample {sample_id} has unsupported signal shape "
                f"{signal.shape}. Expected mono or multichannel audio."
            )

        if signal.shape[0] == 0:
            raise ValueError(
                f"Sample {sample_id} contains an empty audio signal."
            )

        if not np.issubdtype(signal.dtype, np.number):
            raise ValueError(
                f"Sample {sample_id} has non-numeric audio dtype: "
                f"{signal.dtype}."
            )

        if not np.all(np.isfinite(signal)):
            raise ValueError(
                f"Sample {sample_id} contains NaN or infinite values."
            )

    @staticmethod
    def _validate_processed_signal(
        signal: np.ndarray,
        sample_rate: int,
        sample_id: str,
    ) -> None:
        if sample_rate <= 0:
            raise RuntimeError(
                f"Preprocessing produced an invalid sample rate for "
                f"{sample_id}: {sample_rate}."
            )

        if signal.shape[0] == 0:
            raise RuntimeError(
                f"Preprocessing produced an empty signal for {sample_id}."
            )

        if not np.all(np.isfinite(signal)):
            raise RuntimeError(
                f"Preprocessing produced NaN or infinite values for "
                f"{sample_id}."
            )

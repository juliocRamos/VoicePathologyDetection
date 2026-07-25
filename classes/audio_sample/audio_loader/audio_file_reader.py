from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import nspfile
import numpy as np
import soundfile as sf


class AudioReadError(RuntimeError):
    """Erro ao carregar um arquivo de áudio."""


class UnsupportedAudioFormatError(AudioReadError):
    """Formato de áudio não suportado pelo pipeline."""


@dataclass(frozen=True)
class LoadedAudio:
    signal: np.ndarray
    sample_rate: int
    filepath: Path

    @property
    def num_samples(self) -> int:
        return int(self.signal.shape[0])

    @property
    def channels(self) -> int:
        if self.signal.ndim == 1:
            return 1

        return int(self.signal.shape[1])

    @property
    def duration(self) -> float:
        if self.sample_rate <= 0:
            return 0.0

        return self.num_samples / self.sample_rate


class AudioFileReader:
    SUPPORTED_EXTENSIONS = {
        ".wav",
        ".nsp",
    }

    def read(
        self,
        filepath: str | Path,
    ) -> LoadedAudio:
        path = Path(filepath)

        self._validate_file(path)

        suffix = path.suffix.lower()

        try:
            if suffix == ".wav":
                signal, sample_rate = self._read_wav(path)

            elif suffix == ".nsp":
                signal, sample_rate = self._read_nsp(path)

            else:
                raise UnsupportedAudioFormatError(
                    f"Unsupported audio format: {suffix}"
                )

        except AudioReadError:
            raise

        except Exception as exc:
            raise AudioReadError(
                f"Could not read audio file '{path}': {exc}"
            ) from exc

        signal = self._to_float32(signal)
        signal = self._normalize_signal_shape(signal)

        return LoadedAudio(
            signal=signal,
            sample_rate=int(sample_rate),
            filepath=path,
        )

    @staticmethod
    def _read_wav(
        filepath: Path,
    ) -> tuple[np.ndarray, int]:
        signal, sample_rate = sf.read(
            str(filepath),
            dtype="float32",
            always_2d=False,
        )

        return np.asarray(signal), int(sample_rate)

    @staticmethod
    def _read_nsp(
        filepath: Path,
    ) -> tuple[np.ndarray, int]:
        sample_rate, signal = nspfile.read(
            str(filepath)
        )

        return np.asarray(signal), int(sample_rate)

    @staticmethod
    def _to_float32(
        signal: np.ndarray,
    ) -> np.ndarray:
        signal = np.asarray(signal)

        if np.issubdtype(signal.dtype, np.floating):
            return signal.astype(
                np.float32,
                copy=False,
            )

        if np.issubdtype(signal.dtype, np.signedinteger):
            dtype_info = np.iinfo(signal.dtype)

            scale = float(
                max(
                    abs(dtype_info.min),
                    abs(dtype_info.max),
                )
            )

            return signal.astype(np.float32) / scale

        if np.issubdtype(signal.dtype, np.unsignedinteger):
            dtype_info = np.iinfo(signal.dtype)

            midpoint = (
                float(dtype_info.max) + 1.0
            ) / 2.0

            return (
                signal.astype(np.float32) - midpoint
            ) / midpoint

        raise AudioReadError(
            f"Unsupported signal dtype: {signal.dtype}"
        )

    @staticmethod
    def _normalize_signal_shape(
        signal: np.ndarray,
    ) -> np.ndarray:
        if signal.ndim == 1:
            return signal

        if signal.ndim == 2:
            return signal

        raise AudioReadError(
            "Unsupported signal shape: "
            f"{signal.shape}. Expected mono or multichannel audio."
        )

    def _validate_file(
        self,
        filepath: Path,
    ) -> None:
        if not filepath.exists():
            raise FileNotFoundError(
                f"Audio file not found: {filepath}"
            )

        if not filepath.is_file():
            raise AudioReadError(
                f"Audio path is not a file: {filepath}"
            )

        suffix = filepath.suffix.lower()

        if suffix not in self.SUPPORTED_EXTENSIONS:
            raise UnsupportedAudioFormatError(
                f"Unsupported audio extension '{suffix}'. "
                f"Supported extensions: "
                f"{sorted(self.SUPPORTED_EXTENSIONS)}"
            )
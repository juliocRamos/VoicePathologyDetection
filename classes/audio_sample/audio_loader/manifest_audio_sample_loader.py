from __future__ import annotations

import math
from pathlib import Path
from typing import Any, SupportsFloat, SupportsIndex, cast

import numpy as np
import pandas as pd

from classes.audio_sample.audio_loader.audio_file_reader import (
    AudioFileReader,
)
from classes.audio_sample.audio_sample import AudioSample


FloatConvertible = (
    str
    | bytes
    | bytearray
    | SupportsFloat
    | SupportsIndex
)


class ManifestAudioSampleLoader:
    """Map one raw or curated manifest row to an AudioSample."""

    def __init__(
        self,
        audio_reader: AudioFileReader | None = None,
    ):
        self.audio_reader = audio_reader or AudioFileReader()

    def load(self, row: pd.Series) -> AudioSample:
        filepath = Path(str(row["filepath"]))
        loaded_audio = self.audio_reader.read(filepath)

        return AudioSample(
            sample_id=str(row["sample_id"]),
            base=str(row["base"]),
            filepath=filepath,
            signal=loaded_audio.signal,
            sr=loaded_audio.sample_rate,
            label=self._get_optional(row, "label"),
            speaker_id=self._get_optional(row, "speaker_id"),
            sex=self._get_optional(row, "sex"),
            age=self._get_optional_float(row, "age"),
            pathology=self._get_optional(row, "pathology"),
            pathology_code=self._get_optional(
                row,
                "pathology_code",
            ),
            vowel=self._get_optional(row, "vowel"),
            pitch=self._get_optional(row, "pitch"),
            metadata=row.to_dict(),
        )

    @staticmethod
    def _is_missing(value: Any) -> bool:
        if value is None:
            return True

        try:
            return bool(pd.isna(value))
        except (TypeError, ValueError):
            return False

    @classmethod
    def _get_scalar_from_row(
        cls,
        row: pd.Series,
        column: str,
    ) -> Any | None:
        if column not in row.index:
            return None

        value = row.loc[column]

        # A duplicated column may produce a Series after a merge.
        if isinstance(value, pd.Series):
            value = value.dropna()

            if value.empty:
                return None

            value = value.iloc[0]

        if isinstance(value, np.generic):
            value = value.item()

        if cls._is_missing(value):
            return None

        return value

    @classmethod
    def _get_optional(
        cls,
        row: pd.Series,
        column: str,
    ) -> str | None:
        value = cls._get_scalar_from_row(row, column)

        if value is None:
            return None

        return str(value)

    @classmethod
    def _get_optional_float(
        cls,
        row: pd.Series,
        column: str,
    ) -> float | None:
        value = cls._get_scalar_from_row(row, column)

        if value is None:
            return None

        if isinstance(value, np.generic):
            value = value.item()

        if not isinstance(
            value,
            (str, int, float, bytes, bytearray),
        ):
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

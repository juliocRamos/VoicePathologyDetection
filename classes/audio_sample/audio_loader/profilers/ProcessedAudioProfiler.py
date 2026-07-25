from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from classes.audio_sample.audio_loader.manifest_audio_sample_loader import (
    ManifestAudioSampleLoader,
)
from classes.audio_sample.audio_loader.preprocessing.audio_preprocessor import (
    AudioPreprocessor,
)


class ProcessedAudioProfiler:
    def __init__(
        self,
        processor: AudioPreprocessor,
        sample_loader: ManifestAudioSampleLoader | None = None,
    ):
        self.preprocessor = processor
        self.sample_loader = (
            sample_loader
            or ManifestAudioSampleLoader()
        )

    def profile_manifest(
            self,
            manifest: pd.DataFrame,
            max_samples: Optional[int] = None
    ) -> pd.DataFrame:
        rows = manifest

        if max_samples is not None:
            rows = rows.head(max_samples)

        records = []

        for _, row in rows.iterrows():
            try:
                sample = self.sample_loader.load(row)
                sample = self.preprocessor.process(sample)

                signal = sample.signal

                rms = float(np.sqrt(np.mean(signal**2)))
                peak = float(np.max(np.abs(signal)))

                records.append({
                    "sample_id": sample.sample_id,
                    "base": sample.base,
                    "filepath": str(sample.filepath),
                    "label": sample.label,
                    "speaker_id": sample.speaker_id,
                    "recording_id": sample.metadata.get(
                        "recording_id"
                    ),
                    "sex": sample.sex,
                    "age": sample.age,
                    "pathology": sample.pathology,
                    "pathology_code": sample.pathology_code,
                    "vowel": sample.vowel,
                    "condition": sample.metadata.get("condition"),
                    "pitch": sample.pitch,
                    "processed_sr": sample.sr,
                    "processed_duration": sample.duration,
                    "processed_samples": len(signal),
                    "processed_rms": rms,
                    "processed_peak": peak,
                    "status": "ok",
                    "error": None
                })
            except Exception as exc:
                records.append({
                    "sample_id": row.get("sample_id"),
                    "base": row.get("base"),
                    "filepath": row.get("filepath"),
                    "label": row.get("label"),
                    "speaker_id": row.get("speaker_id"),
                    "recording_id": row.get("recording_id"),
                    "sex": row.get("sex"),
                    "age": row.get("age"),
                    "pathology": row.get("pathology"),
                    "pathology_code": row.get("pathology_code"),
                    "vowel": row.get("vowel"),
                    "condition": row.get("condition"),
                    "pitch": row.get("pitch"),
                    "processed_sr": None,
                    "processed_duration": None,
                    "processed_samples": None,
                    "processed_rms": None,
                    "processed_peak": None,
                    "status": "error",
                    "error": str(exc),
                })

        return pd.DataFrame(records)

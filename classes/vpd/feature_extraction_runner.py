import pandas as pd

from classes.audio_sample.audio_loader.manifest_audio_sample_loader import (
    ManifestAudioSampleLoader,
)
from classes.audio_sample.audio_loader.preprocessing.audio_preprocessor import (
    AudioPreprocessor,
)
from classes.vpd.vpd_feature_extractor import VPDFeatureExtractor


class FeatureExtractionRunner:
    def __init__(
        self,
        preprocessor: AudioPreprocessor,
        extractor: VPDFeatureExtractor,
        sample_loader: ManifestAudioSampleLoader | None = None,
    ):
        self.preprocessor = preprocessor
        self.extractor = extractor
        self.sample_loader = (
            sample_loader
            or ManifestAudioSampleLoader()
        )

    def extract_from_manifest(
        self,
        manifest: pd.DataFrame,
        max_samples: int | None = None,
    ) -> pd.DataFrame:

        rows = manifest.copy()

        if max_samples is not None:
            rows = rows.head(max_samples)

        records = []

        for _, row in rows.iterrows():
            try:
                sample = self.sample_loader.load(row)
                sample = self.preprocessor.process(sample)
                record = self.extractor.extract(sample)

                record["status"] = "ok"
                record["error"] = None

            except Exception as exc:
                record = {
                    "sample_id": row.get("sample_id"),
                    "base": row.get("base"),
                    "filepath": row.get("filepath"),
                    "label": row.get("label"),
                    "speaker_id": row.get("speaker_id"),
                    "speaker_id_source": row.get(
                        "speaker_id_source"
                    ),
                    "recording_id": row.get("recording_id"),
                    "sex": row.get("sex"),
                    "age": row.get("age"),
                    "pathology": row.get("pathology"),
                    "pathology_code": row.get("pathology_code"),
                    "pathology_group": row.get(
                        "pathology_group"
                    ),
                    "pathology_groups": row.get(
                        "pathology_groups"
                    ),
                    "vowel": row.get("vowel"),
                    "condition": row.get("condition"),
                    "pitch": row.get("pitch"),
                    "file_sha256": row.get("file_sha256"),
                    "source_count": row.get("source_count"),
                    "is_consolidated_duplicate": row.get(
                        "is_consolidated_duplicate"
                    ),
                    "metadata_conflict_columns": row.get(
                        "metadata_conflict_columns"
                    ),
                    "status": "error",
                    "error": str(exc),
                }

            records.append(record)

        features_df = pd.DataFrame(records)

        if len(features_df) != len(rows):
            raise ValueError(
                f"Feature extraction size mismatch: "
                f"manifest={len(rows)}, features_df={len(features_df)}"
            )

        if "sample_id" in features_df.columns:
            duplicated = features_df["sample_id"].duplicated().sum()

            if duplicated > 0:
                duplicated_rows = features_df[
                    features_df["sample_id"].duplicated(keep=False)
                ]

                raise ValueError(
                    f"Duplicated sample_id values found after feature extraction: "
                    f"{duplicated}\n"
                    f"{duplicated_rows[['sample_id', 'status', 'error']].head(30)}"
                )

        return features_df

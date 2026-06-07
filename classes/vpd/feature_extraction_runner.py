import pandas as pd

from classes.vpd.vpd_feature_extractor import VPDFeatureExtractor


class FeatureExtractionRunner:
    def __init__(self, preprocessor, extractor: VPDFeatureExtractor):
        self.preprocessor = preprocessor
        self.extractor = extractor

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
                sample = self.preprocessor.process_manifest_row(row)
                record = self.extractor.extract(sample)

                record["status"] = "ok"
                record["error"] = None

            except Exception as exc:
                record = {
                    "sample_id": row.get("sample_id"),
                    "base": row.get("base"),
                    "label": row.get("label"),
                    "speaker_id": row.get("speaker_id"),
                    "sex": row.get("sex"),
                    "age": row.get("age"),
                    "pathology": row.get("pathology"),
                    "pathology_code": row.get("pathology_code"),
                    "vowel": row.get("vowel"),
                    "pitch": row.get("pitch"),
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
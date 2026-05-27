import pandas as pd

from classes.vpd.vpd_feature_config import VPDFeatureConfig
from classes.vpd.vpd_feature_extractor import VPDFeatureExtractor


class FeatureExtractionRunner:
    def __init__(self, preprocessor, extractor: VPDFeatureExtractor):
        self.preprocessor = preprocessor
        self.extractor = extractor

    def extract_from_manifest(
            self,
            manifest: pd.DataFrame,
            max_samples: int | None = None
    ) -> pd.DataFrame:

        rows = manifest

        if max_samples is not None:
            rows = rows.head(max_samples)


        records = []

        for _, row in rows.iterrows():
            try:
                sample = self.preprocessor.process_manifest_row(row)
                record = self.extractor.extract(sample)

                record["status"] = "ok"
                record["error"] = None
                records.append(record)

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

        return pd.DataFrame(records)
from __future__ import annotations

from pathlib import Path
import json

from classes.dataset.preparation.training_manifest import (
    TrainingManifestResult,
)


class TrainingManifestWriter:
    """Persist all auditable artifacts produced by a cohort builder."""

    def __init__(
        self,
        manifests_dir: str | Path,
        reports_dir: str | Path,
        dataset_slug: str,
    ):
        self.manifests_dir = Path(manifests_dir)
        self.reports_dir = Path(reports_dir)
        self.dataset_slug = dataset_slug.strip().lower()

        if not self.dataset_slug:
            raise ValueError("dataset_slug cannot be empty.")

    def write(self, result: TrainingManifestResult) -> None:
        self.manifests_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        prefix = self.dataset_slug

        result.training_manifest.to_parquet(
            self.manifests_dir
            / f"{prefix}_training_manifest.parquet",
            index=False,
        )
        result.training_manifest.to_csv(
            self.manifests_dir
            / f"{prefix}_training_manifest.csv",
            index=False,
        )
        result.excluded_samples.to_csv(
            self.manifests_dir
            / f"{prefix}_excluded_samples.csv",
            index=False,
        )
        result.duplicate_groups.to_csv(
            self.manifests_dir
            / f"{prefix}_duplicate_groups.csv",
            index=False,
        )

        with (
            self.reports_dir
            / f"{prefix}_preparation_summary.json"
        ).open("w", encoding="utf-8") as file:
            json.dump(
                result.summary,
                file,
                indent=4,
                ensure_ascii=False,
            )

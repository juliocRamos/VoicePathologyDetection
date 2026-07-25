from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import re

import pandas as pd

from classes.dataset.preparation.training_manifest import (
    TrainingManifestBuilder,
    TrainingManifestConfig,
    TrainingManifestResult,
)


@dataclass(frozen=True)
class SVDTrainingManifestConfig(TrainingManifestConfig):
    """Cohort definition for the primary SVD experiment."""

    adults_only: bool = True
    vowels: tuple[str, ...] = ("a",)
    conditions: tuple[str, ...] = ("n",)

    def __post_init__(self) -> None:
        super().__post_init__()

        if not self.vowels:
            raise ValueError("At least one vowel must be selected.")

        if not self.conditions:
            raise ValueError(
                "At least one recording condition must be selected."
            )


class SVDTrainingManifestBuilder(TrainingManifestBuilder):
    """Create one auditable SVD row per physical acoustic signal."""

    ACOUSTIC_KEY_COLUMNS = [
        "recording_id",
        "vowel",
        "condition",
    ]

    def __init__(self, config: SVDTrainingManifestConfig):
        super().__init__(config)
        self.config = config

    def build(
        self,
        raw_manifest: pd.DataFrame,
    ) -> TrainingManifestResult:
        self._validate_input_manifest(
            raw_manifest,
            dataset_required_columns={
                "recording_id",
                "speaker_id",
                "vowel",
                "condition",
                "pathology_group",
            },
        )

        candidates = raw_manifest.copy()
        excluded_frames: list[pd.DataFrame] = []

        candidates, excluded = self._exclude_audio_read_errors(
            candidates
        )
        excluded_frames.append(excluded)

        candidates, excluded = self._exclude_rows(
            dataframe=candidates,
            mask=~candidates["vowel"].isin(self.config.vowels),
            reason="vowel_not_selected",
            detail=self._observed_value(candidates, "vowel"),
        )
        excluded_frames.append(excluded)

        candidates, excluded = self._exclude_rows(
            dataframe=candidates,
            mask=~candidates["condition"].isin(
                self.config.conditions
            ),
            reason="condition_not_selected",
            detail=self._observed_value(candidates, "condition"),
        )
        excluded_frames.append(excluded)

        candidates, excluded = self._exclude_missing_labels(
            candidates
        )
        excluded_frames.append(excluded)

        candidates, excluded = self._exclude_missing_speakers(
            candidates
        )
        excluded_frames.append(excluded)

        candidates, age_exclusions = self._exclude_by_age(candidates)
        excluded_frames.extend(age_exclusions)

        candidates, duration_exclusions = self._exclude_by_duration(
            candidates
        )
        excluded_frames.extend(duration_exclusions)

        if candidates.empty:
            raise ValueError(
                "No SVD samples remain after applying cohort criteria."
            )

        candidates, hash_exclusions = self._calculate_file_hashes(
            candidates
        )
        excluded_frames.append(hash_exclusions)

        if candidates.empty:
            raise ValueError(
                "No SVD samples remain after calculating file hashes."
            )

        self._validate_binary_labels(candidates)
        self._validate_hash_relationships(candidates)

        training_manifest, duplicate_groups = (
            self._consolidate_duplicates(candidates)
        )
        excluded_samples = self._combine_exclusions(
            raw_manifest=raw_manifest,
            excluded_frames=excluded_frames,
        )

        self._validate_training_manifest(training_manifest)

        summary = self._base_summary(
            raw_manifest=raw_manifest,
            candidates_before_deduplication=candidates,
            training_manifest=training_manifest,
            excluded_samples=excluded_samples,
            duplicate_groups=duplicate_groups,
        )
        summary["dataset"] = "SVD"
        summary["unique_recordings"] = int(
            training_manifest["recording_id"].nunique()
        )

        return TrainingManifestResult(
            training_manifest=training_manifest,
            excluded_samples=excluded_samples,
            duplicate_groups=duplicate_groups,
            summary=summary,
        )

    def _validate_hash_relationships(
        self,
        candidates: pd.DataFrame,
    ) -> None:
        hashes_per_acoustic_key = (
            candidates.groupby(
                self.ACOUSTIC_KEY_COLUMNS,
                dropna=False,
            )[self.HASH_COLUMN]
            .nunique(dropna=False)
        )
        inconsistent_keys = hashes_per_acoustic_key[
            hashes_per_acoustic_key > 1
        ]

        if not inconsistent_keys.empty:
            keys = inconsistent_keys.reset_index()[
                self.ACOUSTIC_KEY_COLUMNS
            ]
            conflicts = candidates.merge(
                keys,
                on=self.ACOUSTIC_KEY_COLUMNS,
                how="inner",
            )
            diagnostic_columns = [
                *self.ACOUSTIC_KEY_COLUMNS,
                "sample_id",
                "filepath",
                self.HASH_COLUMN,
                "pathology_group",
            ]
            raise ValueError(
                "Different hashes were found for the same SVD acoustic "
                "key. The copies cannot be consolidated safely:\n"
                f"{conflicts[diagnostic_columns].head(50).to_string(index=False)}"
            )

        conflicting_groups: list[pd.DataFrame] = []

        for _, group in candidates.groupby(
            self.HASH_COLUMN,
            sort=False,
            dropna=False,
        ):
            acoustic_keys = group[
                self.ACOUSTIC_KEY_COLUMNS
            ].drop_duplicates()
            labels = group["label"].astype("string").drop_duplicates()
            speakers = (
                group["speaker_id"]
                .astype("string")
                .drop_duplicates()
            )

            if (
                len(acoustic_keys) != 1
                or len(labels) != 1
                or len(speakers) != 1
            ):
                conflicting_groups.append(group)

        if conflicting_groups:
            conflicts = pd.concat(
                conflicting_groups,
                ignore_index=True,
            )
            diagnostic_columns = [
                self.HASH_COLUMN,
                *self.ACOUSTIC_KEY_COLUMNS,
                "sample_id",
                "speaker_id",
                "label",
                "filepath",
            ]
            raise ValueError(
                "A file hash is associated with conflicting acoustic "
                "keys, speakers, or labels:\n"
                f"{conflicts[diagnostic_columns].head(50).to_string(index=False)}"
            )

    def _consolidate_duplicates(
        self,
        candidates: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        sort_columns = [
            column
            for column in [
                self.HASH_COLUMN,
                "pathology_group_key",
                "relative_path",
                "sample_id",
            ]
            if column in candidates.columns
        ]
        ordered = candidates.sort_values(
            sort_columns,
            kind="stable",
        )

        canonical_rows: list[pd.Series] = []
        duplicate_records: list[dict[str, Any]] = []

        for file_hash, group in ordered.groupby(
            self.HASH_COLUMN,
            sort=True,
            dropna=False,
        ):
            canonical = group.iloc[0].copy()
            source_sample_id = str(canonical["sample_id"])
            canonical_sample_id = self._build_canonical_sample_id(
                canonical
            )

            pathology_groups = self._join_unique_values(
                group["pathology_group"]
            )
            pathology_group_keys = self._join_unique_values(
                group["pathology_group_key"]
                if "pathology_group_key" in group.columns
                else group["pathology_group"]
            )
            pathologies = self._join_unique_values(
                group["pathology"]
                if "pathology" in group.columns
                else group["pathology_group"]
            )

            canonical["source_sample_id"] = source_sample_id
            canonical["sample_id"] = canonical_sample_id
            canonical[self.HASH_COLUMN] = str(file_hash)
            canonical["pathology_group"] = pathology_groups
            canonical["pathology_group_key"] = pathology_group_keys
            canonical["pathology_groups"] = pathology_groups
            canonical["pathology"] = pathologies
            canonical["pathologies"] = pathologies
            canonical["source_count"] = int(len(group))
            canonical["is_consolidated_duplicate"] = bool(
                len(group) > 1
            )
            canonical["source_sample_ids"] = self._join_unique_values(
                group["sample_id"]
            )
            canonical["source_filepaths"] = self._join_unique_values(
                group["filepath"]
            )

            if "relative_path" in group.columns:
                canonical["source_relative_paths"] = (
                    self._join_unique_values(group["relative_path"])
                )

            if "metadata_path" in group.columns:
                canonical["source_metadata_paths"] = (
                    self._join_unique_values(group["metadata_path"])
                )

            canonical_rows.append(canonical)

            if len(group) > 1:
                for _, source in group.iterrows():
                    duplicate_records.append({
                        self.HASH_COLUMN: str(file_hash),
                        "canonical_sample_id": canonical_sample_id,
                        "group_size": int(len(group)),
                        "is_canonical_source": (
                            str(source["sample_id"])
                            == source_sample_id
                        ),
                        "sample_id": source["sample_id"],
                        "filepath": source["filepath"],
                        "relative_path": source.get("relative_path"),
                        "recording_id": source["recording_id"],
                        "speaker_id": source["speaker_id"],
                        "label": source["label"],
                        "vowel": source["vowel"],
                        "condition": source["condition"],
                        "pathology_group": source["pathology_group"],
                        "pathology": source.get("pathology"),
                    })

        training_manifest = pd.DataFrame(canonical_rows).reset_index(
            drop=True
        )
        duplicate_groups = pd.DataFrame(
            duplicate_records,
            columns=[
                self.HASH_COLUMN,
                "canonical_sample_id",
                "group_size",
                "is_canonical_source",
                "sample_id",
                "filepath",
                "relative_path",
                "recording_id",
                "speaker_id",
                "label",
                "vowel",
                "condition",
                "pathology_group",
                "pathology",
            ],
        )
        return training_manifest, duplicate_groups

    @staticmethod
    def _build_canonical_sample_id(row: pd.Series) -> str:
        components = [
            "svd",
            row["recording_id"],
            row["vowel"],
            row["condition"],
        ]
        normalized = [
            re.sub(
                r"[^a-z0-9]+",
                "_",
                str(component).strip().lower(),
            ).strip("_")
            for component in components
        ]
        return "_".join(normalized)

    def _validate_training_manifest(
        self,
        training_manifest: pd.DataFrame,
    ) -> None:
        if training_manifest.empty:
            raise RuntimeError("Curated SVD training manifest is empty.")

        for column in ["sample_id", self.HASH_COLUMN]:
            if training_manifest[column].duplicated().any():
                raise RuntimeError(
                    f"Duplicated {column} values remain in the curated "
                    "SVD training manifest."
                )

        if training_manifest.duplicated(
            subset=self.ACOUSTIC_KEY_COLUMNS
        ).any():
            raise RuntimeError(
                "Duplicated acoustic keys remain in the curated SVD "
                "training manifest."
            )

        if (
            self.config.require_speaker_id
            and self._missing_mask(
                training_manifest["speaker_id"]
            ).any()
        ):
            raise RuntimeError(
                "Missing speaker_id values remain in the curated SVD "
                "training manifest."
            )

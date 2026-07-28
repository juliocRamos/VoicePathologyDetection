from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from classes.dataset.preparation.training_manifest import (
    TrainingManifestBuilder,
    TrainingManifestConfig,
    TrainingManifestResult,
)


@dataclass(frozen=True)
class HUPATrainingManifestConfig(TrainingManifestConfig):
    """Cohort definition for HUPA independent experiments."""

    adults_only: bool = False


class HUPATrainingManifestBuilder(TrainingManifestBuilder):
    """Deduplicate and audit the already-segmented HUPA cohort."""

    SOURCE_IDENTITY_COLUMNS = {
        "sample_id",
        "speaker_id",
        "speaker_id_source",
        "filepath",
        "relative_path",
        "filename",
        "file_stem",
        "file_key",
        "metadata_filename",
        "metadata_file_key",
        "metadata_sheet",
        "folder",
    }
    AGGREGATED_CLINICAL_COLUMNS = {
        "pathology",
        "pathology_code",
    }
    INVARIANT_COLUMNS = {
        "base",
        "label",
        "vowel",
        "samplerate",
        "duration",
        "channels",
        "audio_read_status",
    }

    def __init__(self, config: HUPATrainingManifestConfig):
        super().__init__(config)
        self.config = config

    def build(
        self,
        raw_manifest: pd.DataFrame,
    ) -> TrainingManifestResult:
        self._validate_input_manifest(
            raw_manifest,
            dataset_required_columns={
                "speaker_id",
                "vowel",
            },
        )

        candidates = raw_manifest.copy()
        excluded_frames: list[pd.DataFrame] = []

        candidates, excluded = self._exclude_audio_read_errors(
            candidates
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
                "No HUPA samples remain after applying cohort criteria."
            )

        candidates, hash_exclusions = self._calculate_file_hashes(
            candidates
        )
        excluded_frames.append(hash_exclusions)

        candidates, age_conflict_exclusions = (
            self._exclude_conflicting_adult_ages(candidates)
        )
        excluded_frames.append(age_conflict_exclusions)

        if candidates.empty:
            raise ValueError(
                "No HUPA samples remain after calculating file hashes."
            )

        self._validate_binary_labels(candidates)
        self._validate_duplicate_labels(candidates)

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
        summary["dataset"] = "HUPA"
        summary["speaker_id_policy"] = (
            "one canonical acoustic file per assumed speaker; "
            "the source metadata has no independent speaker identifier"
        )
        summary["metadata_conflict_groups"] = int(
            duplicate_groups.loc[
                duplicate_groups["metadata_conflict_columns"]
                .astype("string")
                .str.strip()
                .ne("")
                .fillna(False),
                self.HASH_COLUMN,
            ].nunique()
            if not duplicate_groups.empty
            else 0
        )

        return TrainingManifestResult(
            training_manifest=training_manifest,
            excluded_samples=excluded_samples,
            duplicate_groups=duplicate_groups,
            summary=summary,
        )

    def _exclude_conflicting_adult_ages(
        self,
        candidates: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if not self.config.adults_only:
            return (
                candidates.copy(),
                self._empty_exclusions(candidates),
            )

        age = pd.to_numeric(candidates["age"], errors="coerce")
        distinct_ages = (
            candidates.assign(_numeric_age=age)
            .groupby(self.HASH_COLUMN)["_numeric_age"]
            .nunique(dropna=True)
        )
        conflicting_hashes = distinct_ages[
            distinct_ages > 1
        ].index
        mask = candidates[self.HASH_COLUMN].isin(conflicting_hashes)
        detail = (
            candidates.loc[mask]
            .groupby(self.HASH_COLUMN)["age"]
            .transform(
                lambda values: (
                    "conflicting_adult_ages="
                    + self._join_unique_values(values)
                )
            )
            .reindex(candidates.index)
        )

        return self._exclude_rows(
            dataframe=candidates,
            mask=mask,
            reason="conflicting_age_for_duplicate_audio",
            detail=detail,
        )

    def _validate_duplicate_labels(
        self,
        candidates: pd.DataFrame,
    ) -> None:
        labels_per_hash = candidates.groupby(
            self.HASH_COLUMN,
            dropna=False,
        )["label"].nunique(dropna=False)
        conflicting_hashes = labels_per_hash[labels_per_hash > 1].index

        if len(conflicting_hashes) == 0:
            return

        conflicts = candidates.loc[
            candidates[self.HASH_COLUMN].isin(conflicting_hashes),
            [
                self.HASH_COLUMN,
                "sample_id",
                "filepath",
                "label",
            ],
        ]
        raise ValueError(
            "Identical HUPA audio files have conflicting binary labels:\n"
            f"{conflicts.to_string(index=False)}"
        )

    def _consolidate_duplicates(
        self,
        candidates: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        sort_columns = [
            column
            for column in [
                self.HASH_COLUMN,
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
            canonical_sample_id = f"hupa_{str(file_hash)[:16]}"
            conflict_columns = self._metadata_conflict_columns(group)

            for column in conflict_columns:
                if (
                    column not in self.AGGREGATED_CLINICAL_COLUMNS
                    and column in canonical.index
                ):
                    canonical[column] = pd.NA

            for column in self.AGGREGATED_CLINICAL_COLUMNS:
                if column in group.columns:
                    canonical[column] = self._join_unique_values(
                        group[column]
                    )

            canonical["source_sample_id"] = source_sample_id
            canonical["sample_id"] = canonical_sample_id
            canonical["speaker_id"] = canonical_sample_id
            canonical["speaker_id_source"] = (
                "assumed_unique_acoustic_sample"
            )
            canonical[self.HASH_COLUMN] = str(file_hash)
            canonical["source_count"] = int(len(group))
            canonical["is_consolidated_duplicate"] = bool(
                len(group) > 1
            )
            canonical["metadata_conflict_columns"] = (
                self.VALUE_SEPARATOR.join(conflict_columns)
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

            canonical_rows.append(canonical)

            if len(group) > 1:
                for _, source in group.iterrows():
                    record = source.to_dict()
                    record.update({
                        "canonical_sample_id": canonical_sample_id,
                        "group_size": int(len(group)),
                        "is_canonical_source": (
                            str(source["sample_id"])
                            == source_sample_id
                        ),
                        "metadata_conflict_columns": (
                            self.VALUE_SEPARATOR.join(conflict_columns)
                        ),
                    })
                    duplicate_records.append(record)

        training_manifest = pd.DataFrame(canonical_rows).reset_index(
            drop=True
        )
        duplicate_groups = pd.DataFrame(duplicate_records)

        if duplicate_groups.empty:
            duplicate_groups = pd.DataFrame(
                columns=[
                    self.HASH_COLUMN,
                    "canonical_sample_id",
                    "group_size",
                    "is_canonical_source",
                    "metadata_conflict_columns",
                ]
            )

        return training_manifest, duplicate_groups

    def _metadata_conflict_columns(
        self,
        group: pd.DataFrame,
    ) -> list[str]:
        if len(group) == 1:
            return []

        conflicts: list[str] = []

        for column in group.columns:
            if (
                column in self.SOURCE_IDENTITY_COLUMNS
                or column == self.HASH_COLUMN
            ):
                continue

            distinct_values = (
                group[column]
                .astype("string")
                .fillna("<NA>")
                .drop_duplicates()
            )

            if len(distinct_values) <= 1:
                continue

            if column in self.INVARIANT_COLUMNS:
                raise ValueError(
                    "Identical HUPA files have conflicting invariant "
                    f"metadata in column '{column}':\n"
                    f"{group[['sample_id', column]].to_string(index=False)}"
                )

            conflicts.append(column)

        return sorted(conflicts)

    def _validate_training_manifest(
        self,
        training_manifest: pd.DataFrame,
    ) -> None:
        if training_manifest.empty:
            raise RuntimeError("Curated HUPA training manifest is empty.")

        for column in [
            "sample_id",
            "speaker_id",
            self.HASH_COLUMN,
        ]:
            if training_manifest[column].duplicated().any():
                raise RuntimeError(
                    f"Duplicated {column} values remain in the curated "
                    "HUPA training manifest."
                )

        if (
            self.config.require_speaker_id
            and self._missing_mask(
                training_manifest["speaker_id"]
            ).any()
        ):
            raise RuntimeError(
                "Missing speaker_id values remain in the curated HUPA "
                "training manifest."
            )

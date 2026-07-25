from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class TrainingManifestConfig:
    """Dataset-independent cohort inclusion criteria."""

    adults_only: bool = False
    minimum_age: float = 18.0
    minimum_duration_sec: float | None = 0.5
    require_audio_status_ok: bool = True
    require_speaker_id: bool = True

    def __post_init__(self) -> None:
        if self.adults_only and self.minimum_age < 0:
            raise ValueError("minimum_age must be non-negative.")

        if (
            self.minimum_duration_sec is not None
            and self.minimum_duration_sec < 0
        ):
            raise ValueError("minimum_duration_sec must be non-negative.")


@dataclass(frozen=True)
class TrainingManifestResult:
    """Auditable outputs of a dataset cohort builder."""

    training_manifest: pd.DataFrame
    excluded_samples: pd.DataFrame
    duplicate_groups: pd.DataFrame
    summary: dict[str, Any]


class TrainingManifestBuilder(ABC):
    """Shared mechanics for dataset-specific training cohort builders."""

    ALLOWED_LABELS = {
        "healthy",
        "pathological",
    }
    HASH_COLUMN = "file_sha256"
    HASH_CHUNK_SIZE = 1024 * 1024
    VALUE_SEPARATOR = " | "

    def __init__(self, config: TrainingManifestConfig):
        self.config = config

    @abstractmethod
    def build(
        self,
        raw_manifest: pd.DataFrame,
    ) -> TrainingManifestResult:
        """Build an immutable, auditable training cohort."""

    def _validate_input_manifest(
        self,
        manifest: pd.DataFrame,
        dataset_required_columns: set[str] | None = None,
    ) -> None:
        required_columns = {
            "sample_id",
            "base",
            "filepath",
            "label",
        }
        required_columns.update(dataset_required_columns or set())

        if self.config.require_audio_status_ok:
            required_columns.add("audio_read_status")

        if self.config.require_speaker_id:
            required_columns.add("speaker_id")

        if self.config.adults_only:
            required_columns.add("age")

        if self.config.minimum_duration_sec is not None:
            required_columns.add("duration")

        missing_columns = required_columns - set(manifest.columns)

        if missing_columns:
            raise ValueError(
                "Raw manifest is missing required columns: "
                f"{sorted(missing_columns)}"
            )

        if manifest.empty:
            raise ValueError("Raw manifest is empty.")

        duplicated_mask = manifest["sample_id"].duplicated(keep=False)

        if duplicated_mask.any():
            duplicated_rows = manifest.loc[
                duplicated_mask,
                ["sample_id", "filepath"],
            ]
            raise ValueError(
                "Raw manifest contains duplicated sample_id values:\n"
                f"{duplicated_rows.head(30).to_string(index=False)}"
            )

    def _exclude_audio_read_errors(
        self,
        dataframe: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if not self.config.require_audio_status_ok:
            return dataframe.copy(), self._empty_exclusions(dataframe)

        status = (
            dataframe["audio_read_status"]
            .astype("string")
            .str.strip()
            .str.lower()
        )
        detail = (
            dataframe["audio_read_error"]
            if "audio_read_error" in dataframe.columns
            else None
        )

        return self._exclude_rows(
            dataframe=dataframe,
            mask=status.ne("ok").fillna(True),
            reason="audio_read_error",
            detail=detail,
        )

    def _exclude_missing_labels(
        self,
        dataframe: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        return self._exclude_rows(
            dataframe=dataframe,
            mask=self._missing_mask(dataframe["label"]),
            reason="missing_label",
        )

    def _exclude_missing_speakers(
        self,
        dataframe: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if not self.config.require_speaker_id:
            return dataframe.copy(), self._empty_exclusions(dataframe)

        return self._exclude_rows(
            dataframe=dataframe,
            mask=self._missing_mask(dataframe["speaker_id"]),
            reason="missing_speaker_id",
        )

    def _exclude_by_age(
        self,
        dataframe: pd.DataFrame,
    ) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
        if not self.config.adults_only:
            return dataframe.copy(), []

        exclusions: list[pd.DataFrame] = []
        age = pd.to_numeric(dataframe["age"], errors="coerce")

        remaining, excluded = self._exclude_rows(
            dataframe=dataframe,
            mask=age.isna(),
            reason="missing_age",
        )
        exclusions.append(excluded)

        age = pd.to_numeric(remaining["age"], errors="coerce")
        detail = (
            "age="
            + age.astype("string")
            + f"; minimum_age={self.config.minimum_age:g}"
        )
        remaining, excluded = self._exclude_rows(
            dataframe=remaining,
            mask=age.lt(self.config.minimum_age),
            reason="age_below_minimum",
            detail=detail,
        )
        exclusions.append(excluded)

        return remaining, exclusions

    def _exclude_by_duration(
        self,
        dataframe: pd.DataFrame,
    ) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
        minimum_duration = self.config.minimum_duration_sec

        if minimum_duration is None:
            return dataframe.copy(), []

        exclusions: list[pd.DataFrame] = []
        duration = pd.to_numeric(
            dataframe["duration"],
            errors="coerce",
        )
        remaining, excluded = self._exclude_rows(
            dataframe=dataframe,
            mask=duration.isna(),
            reason="missing_duration",
        )
        exclusions.append(excluded)

        duration = pd.to_numeric(
            remaining["duration"],
            errors="coerce",
        )
        detail = (
            "duration="
            + duration.astype("string")
            + f"; minimum_duration_sec={minimum_duration:g}"
        )
        remaining, excluded = self._exclude_rows(
            dataframe=remaining,
            mask=duration.lt(minimum_duration),
            reason="duration_below_minimum",
            detail=detail,
        )
        exclusions.append(excluded)

        return remaining, exclusions

    @staticmethod
    def _exclude_rows(
        dataframe: pd.DataFrame,
        mask: pd.Series,
        reason: str,
        detail: pd.Series | str | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        normalized_mask = (
            mask.reindex(dataframe.index, fill_value=False)
            .fillna(False)
            .astype(bool)
        )
        excluded = dataframe.loc[normalized_mask].copy()
        excluded["exclusion_reason"] = reason

        if isinstance(detail, pd.Series):
            excluded["exclusion_detail"] = detail.reindex(excluded.index)
        else:
            excluded["exclusion_detail"] = detail

        remaining = dataframe.loc[~normalized_mask].copy()
        return remaining, excluded

    @staticmethod
    def _missing_mask(values: pd.Series) -> pd.Series:
        as_string = values.astype("string")
        return values.isna() | as_string.str.strip().eq("").fillna(True)

    @staticmethod
    def _observed_value(
        dataframe: pd.DataFrame,
        column: str,
    ) -> pd.Series:
        return (
            f"{column}="
            + dataframe[column].astype("string").fillna("<NA>")
        )

    def _calculate_file_hashes(
        self,
        dataframe: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        hashed = dataframe.copy()
        hash_values = pd.Series(index=hashed.index, dtype="string")
        hash_errors = pd.Series(index=hashed.index, dtype="string")

        for index, filepath in hashed["filepath"].items():
            try:
                hash_values.loc[index] = self._sha256_file(
                    Path(str(filepath))
                )
            except Exception as exc:
                hash_errors.loc[index] = str(exc)

        hashed[self.HASH_COLUMN] = hash_values

        return self._exclude_rows(
            dataframe=hashed,
            mask=hashed[self.HASH_COLUMN].isna(),
            reason="hash_error",
            detail=hash_errors,
        )

    @classmethod
    def _sha256_file(cls, filepath: Path) -> str:
        digest = sha256()

        with filepath.open("rb") as stream:
            for chunk in iter(
                lambda: stream.read(cls.HASH_CHUNK_SIZE),
                b"",
            ):
                digest.update(chunk)

        return digest.hexdigest()

    def _validate_binary_labels(
        self,
        dataframe: pd.DataFrame,
    ) -> None:
        labels = set(
            dataframe["label"]
            .astype("string")
            .dropna()
            .str.strip()
            .str.lower()
            .tolist()
        )
        unexpected_labels = labels - self.ALLOWED_LABELS
        missing_labels = self.ALLOWED_LABELS - labels

        if unexpected_labels:
            raise ValueError(
                "Unexpected labels in training cohort: "
                f"{sorted(unexpected_labels)}"
            )

        if missing_labels:
            raise ValueError(
                "Training cohort must contain both binary classes. "
                f"Missing labels: {sorted(missing_labels)}"
            )

    @classmethod
    def _join_unique_values(cls, values: pd.Series) -> str:
        unique_values = {
            str(value).strip()
            for value in values
            if not pd.isna(value) and str(value).strip()
        }
        return cls.VALUE_SEPARATOR.join(
            sorted(unique_values, key=str.casefold)
        )

    @staticmethod
    def _combine_exclusions(
        raw_manifest: pd.DataFrame,
        excluded_frames: list[pd.DataFrame],
    ) -> pd.DataFrame:
        nonempty_frames = [
            frame for frame in excluded_frames if not frame.empty
        ]

        if not nonempty_frames:
            return TrainingManifestBuilder._empty_exclusions(raw_manifest)

        excluded = pd.concat(nonempty_frames, ignore_index=False)

        if excluded["sample_id"].duplicated().any():
            raise RuntimeError(
                "A manifest row received more than one primary "
                "exclusion reason."
            )

        return excluded.reset_index(drop=True)

    @staticmethod
    def _empty_exclusions(dataframe: pd.DataFrame) -> pd.DataFrame:
        result = dataframe.head(0).copy()
        result["exclusion_reason"] = pd.Series(dtype="string")
        result["exclusion_detail"] = pd.Series(dtype="string")
        return result

    def _base_summary(
        self,
        raw_manifest: pd.DataFrame,
        candidates_before_deduplication: pd.DataFrame,
        training_manifest: pd.DataFrame,
        excluded_samples: pd.DataFrame,
        duplicate_groups: pd.DataFrame,
    ) -> dict[str, Any]:
        exclusion_counts = (
            excluded_samples["exclusion_reason"]
            .value_counts()
            .sort_index()
            .to_dict()
            if not excluded_samples.empty
            else {}
        )
        class_counts = (
            training_manifest["label"]
            .value_counts()
            .sort_index()
            .to_dict()
        )
        duplicate_group_count = (
            int(duplicate_groups[self.HASH_COLUMN].nunique())
            if not duplicate_groups.empty
            else 0
        )

        return {
            "config": asdict(self.config),
            "hash_algorithm": "sha256",
            "input_rows": int(len(raw_manifest)),
            "excluded_rows": int(len(excluded_samples)),
            "exclusions_by_reason": {
                str(key): int(value)
                for key, value in exclusion_counts.items()
            },
            "candidate_rows_before_deduplication": int(
                len(candidates_before_deduplication)
            ),
            "training_rows": int(len(training_manifest)),
            "duplicate_groups": duplicate_group_count,
            "duplicate_source_rows_collapsed": int(
                len(candidates_before_deduplication)
                - len(training_manifest)
            ),
            "class_distribution": {
                str(key): int(value)
                for key, value in class_counts.items()
            },
            "unique_speakers": int(
                training_manifest["speaker_id"].nunique()
            ),
        }

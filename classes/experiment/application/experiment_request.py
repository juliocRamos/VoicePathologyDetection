from __future__ import annotations

import argparse
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from classes.experiment.runners.cross_database_direction import (
    CrossDatabaseDirection,
)
from classes.experiment.runners.experiment_stage import ExperimentStage
from classes.experiment.training.compute_backend import ComputeBackend


class ExperimentDataset(str, Enum):
    HUPA = "hupa"
    SVD = "svd"
    CROSS = "cross"
    POOLED = "pooled"


@dataclass(frozen=True)
class ExperimentRequest:
    dataset: ExperimentDataset
    stage: ExperimentStage
    compute_backend: ComputeBackend
    experiment_name: str
    resume_experiment: Path | None = None
    hupa_source_experiment: Path | None = None
    svd_source_experiment: Path | None = None
    svd_vowels: tuple[str, ...] = ("a",)
    cross_direction: CrossDatabaseDirection = (
        CrossDatabaseDirection.BOTH
    )

    def __post_init__(self) -> None:
        allowed_vowels = {"a", "i", "u"}

        if not self.svd_vowels:
            raise ValueError(
                "svd_vowels must contain at least one vowel."
            )

        if len(set(self.svd_vowels)) != len(self.svd_vowels):
            raise ValueError("svd_vowels cannot contain duplicates.")

        unexpected_vowels = set(self.svd_vowels) - allowed_vowels

        if unexpected_vowels:
            raise ValueError(
                "Unsupported SVD vowels: "
                f"{sorted(unexpected_vowels)}."
            )

        canonical_vowels = tuple(
            vowel
            for vowel in ("a", "i", "u")
            if vowel in self.svd_vowels
        )

        if self.svd_vowels != canonical_vowels:
            raise ValueError(
                "svd_vowels must use canonical order: "
                f"{canonical_vowels}."
            )

        if not isinstance(
            self.cross_direction,
            CrossDatabaseDirection,
        ):
            raise TypeError(
                "cross_direction must be a "
                "CrossDatabaseDirection instance."
            )

        if (
            self.dataset is not ExperimentDataset.CROSS
            and self.cross_direction
            is not CrossDatabaseDirection.BOTH
        ):
            raise ValueError(
                "cross_direction can only be changed for the cross "
                "dataset."
            )

    @classmethod
    def from_namespace(
        cls,
        namespace: argparse.Namespace,
    ) -> ExperimentRequest:
        dataset = ExperimentDataset(namespace.dataset)
        experiment_name = (
            namespace.experiment_name
            or cls.default_experiment_name(dataset)
        )
        requested_vowels = tuple(
            getattr(namespace, "svd_vowels", ("a",))
        )

        if len(set(requested_vowels)) != len(requested_vowels):
            raise ValueError("svd_vowels cannot contain duplicates.")

        unexpected_vowels = (
            set(requested_vowels) - {"a", "i", "u"}
        )

        if unexpected_vowels:
            raise ValueError(
                "Unsupported SVD vowels: "
                f"{sorted(unexpected_vowels)}."
            )

        svd_vowels = tuple(
            vowel
            for vowel in ("a", "i", "u")
            if vowel in requested_vowels
        )

        return cls(
            dataset=dataset,
            stage=ExperimentStage(namespace.stage),
            compute_backend=ComputeBackend(
                namespace.compute_backend
            ),
            experiment_name=experiment_name,
            resume_experiment=(
                Path(namespace.resume_experiment).resolve()
                if getattr(namespace, "resume_experiment", None)
                is not None
                else None
            ),
            hupa_source_experiment=(
                Path(namespace.hupa_source_experiment).resolve()
                if getattr(
                    namespace,
                    "hupa_source_experiment",
                    None,
                )
                is not None
                else None
            ),
            svd_source_experiment=(
                Path(namespace.svd_source_experiment).resolve()
                if getattr(
                    namespace,
                    "svd_source_experiment",
                    None,
                )
                is not None
                else None
            ),
            svd_vowels=svd_vowels,
            cross_direction=CrossDatabaseDirection(
                getattr(namespace, "cross_direction", "both")
            ),
        )

    @staticmethod
    def default_experiment_name(
        dataset: ExperimentDataset,
    ) -> str:
        return (
            f"{dataset.value}_pre16k_rms20_"
            "fullsignal_features_v1"
        )

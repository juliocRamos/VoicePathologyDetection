from __future__ import annotations

import argparse
from dataclasses import dataclass
from enum import Enum

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

        return cls(
            dataset=dataset,
            stage=ExperimentStage(namespace.stage),
            compute_backend=ComputeBackend(
                namespace.compute_backend
            ),
            experiment_name=experiment_name,
        )

    @staticmethod
    def default_experiment_name(
        dataset: ExperimentDataset,
    ) -> str:
        return (
            f"{dataset.value}_pre16k_rms20_"
            "fullsignal_features_v1"
        )

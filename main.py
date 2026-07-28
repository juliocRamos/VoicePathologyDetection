from __future__ import annotations

import argparse

from classes.experiment.application.experiment_application import (
    ExperimentApplication,
)
from classes.experiment.application.experiment_config_factory import (
    ExperimentConfigFactory,
)
from classes.experiment.application.experiment_request import (
    ExperimentDataset,
    ExperimentRequest,
)
from classes.experiment.application.experiment_settings import (
    ExperimentSettings,
)
from classes.experiment.runners.experiment_stage import ExperimentStage
from classes.experiment.training.compute_backend import ComputeBackend


# Backward-compatible aliases for callers that imported builders from main.
build_preprocess_config = (
    ExperimentConfigFactory.build_preprocess_config
)
build_feature_config = ExperimentConfigFactory.build_feature_config
build_hupa_manifest_config = (
    ExperimentConfigFactory.build_hupa_manifest_config
)
build_cross_hupa_manifest_config = (
    ExperimentConfigFactory.build_cross_hupa_manifest_config
)
build_svd_manifest_config = (
    ExperimentConfigFactory.build_svd_manifest_config
)
build_cross_svd_manifest_config = (
    ExperimentConfigFactory.build_cross_svd_manifest_config
)
build_training_config = (
    ExperimentConfigFactory.build_training_config
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute voice pathology detection experiments."
    )
    parser.add_argument(
        "--dataset",
        choices=[dataset.value for dataset in ExperimentDataset],
        required=True,
        help=(
            "Experiment to execute. 'cross' runs HUPA->SVD and "
            "SVD->HUPA. 'pooled' mixes both databases with a "
            "speaker-disjoint holdout."
        ),
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Optional experiment name.",
    )
    parser.add_argument(
        "--stage",
        choices=[stage.value for stage in ExperimentStage],
        default=ExperimentStage.PREPARE.value,
        help=(
            "Last pipeline stage to execute. 'prepare' is the safe "
            "default; 'features' also extracts attributes; 'train' "
            "runs the complete experiment."
        ),
    )
    parser.add_argument(
        "--compute-backend",
        choices=[backend.value for backend in ComputeBackend],
        default=ComputeBackend.CPU.value,
        help=(
            "Backend used during model training. "
            "CUDA requires requirements-gpu.txt."
        ),
    )
    return parser.parse_args()


def main() -> None:
    request = ExperimentRequest.from_namespace(parse_arguments())
    application = ExperimentApplication(
        settings=ExperimentSettings.default(),
    )
    application.run(request)


if __name__ == "__main__":
    main()

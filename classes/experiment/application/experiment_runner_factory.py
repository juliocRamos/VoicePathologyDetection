from __future__ import annotations

from typing import Any

from classes.experiment.application.experiment_config_factory import (
    ExperimentConfigBundle,
)
from classes.experiment.application.experiment_request import (
    ExperimentDataset,
    ExperimentRequest,
)
from classes.experiment.application.experiment_settings import (
    ExperimentSettings,
)


class ExperimentRunnerFactory:
    """Create experiment runners after the compute backend is active."""

    def __init__(self, settings: ExperimentSettings) -> None:
        self.settings = settings

    def create(
        self,
        request: ExperimentRequest,
        configs: ExperimentConfigBundle,
    ) -> Any:
        if request.dataset is ExperimentDataset.HUPA:
            return self._create_hupa_runner(
                experiment_name=request.experiment_name,
                configs=configs,
            )

        if request.dataset is ExperimentDataset.SVD:
            return self._create_svd_runner(
                experiment_name=request.experiment_name,
                configs=configs,
            )

        hupa_runner = self._create_hupa_runner(
            experiment_name=f"{request.experiment_name}_hupa",
            configs=configs,
        )
        svd_runner = self._create_svd_runner(
            experiment_name=f"{request.experiment_name}_svd",
            configs=configs,
        )

        if request.dataset is ExperimentDataset.CROSS:
            from classes.experiment.runners.cross_database_experiment_runner import (
                CrossDatabaseExperimentRunner,
            )

            return CrossDatabaseExperimentRunner(
                hupa_runner=hupa_runner,
                svd_runner=svd_runner,
                data_root=self.settings.data_root,
                experiment_name=request.experiment_name,
                training_config=configs.training,
            )

        if request.dataset is ExperimentDataset.POOLED:
            from classes.experiment.runners.pooled_database_experiment_runner import (
                PooledDatabaseExperimentRunner,
            )

            return PooledDatabaseExperimentRunner(
                hupa_runner=hupa_runner,
                svd_runner=svd_runner,
                data_root=self.settings.data_root,
                experiment_name=request.experiment_name,
                training_config=configs.training,
            )

        raise ValueError(
            f"Unsupported dataset: {request.dataset.value}"
        )

    def _create_hupa_runner(
        self,
        experiment_name: str,
        configs: ExperimentConfigBundle,
    ) -> Any:
        from classes.experiment.runners.hupa_experiment_runner import (
            HUPAExperimentRunner,
        )

        return HUPAExperimentRunner(
            dataset_root=self.settings.hupa_root,
            data_root=self.settings.data_root,
            experiment_name=experiment_name,
            preprocess_config=configs.preprocess,
            feature_config=configs.features,
            manifest_config=configs.hupa_manifest,
            training_config=configs.training,
        )

    def _create_svd_runner(
        self,
        experiment_name: str,
        configs: ExperimentConfigBundle,
    ) -> Any:
        from classes.experiment.runners.svd_experiment_runner import (
            SVDExperimentRunner,
        )

        return SVDExperimentRunner(
            dataset_root=self.settings.svd_root,
            data_root=self.settings.data_root,
            experiment_name=experiment_name,
            preprocess_config=configs.preprocess,
            feature_config=configs.features,
            manifest_config=configs.svd_manifest,
            training_config=configs.training,
        )

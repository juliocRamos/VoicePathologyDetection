from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
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
from classes.experiment.training.training_config import TrainingConfig


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
                experiment_root=request.resume_experiment,
            )

        if request.dataset is ExperimentDataset.SVD:
            return self._create_svd_runner(
                experiment_name=request.experiment_name,
                configs=configs,
                experiment_root=request.resume_experiment,
            )

        source_roots = self._source_roots(
            experiment_root=request.resume_experiment,
            hupa_source_experiment=(
                request.hupa_source_experiment
            ),
            svd_source_experiment=(
                request.svd_source_experiment
            ),
        )
        hupa_runner = self._create_hupa_runner(
            experiment_name=f"{request.experiment_name}_hupa",
            configs=configs,
            experiment_root=source_roots.get("hupa"),
        )
        svd_runner = self._create_svd_runner(
            experiment_name=f"{request.experiment_name}_svd",
            configs=configs,
            experiment_root=source_roots.get("svd"),
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
                direction=request.cross_direction,
                experiment_root=request.resume_experiment,
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
                experiment_root=request.resume_experiment,
            )

        raise ValueError(
            f"Unsupported dataset: {request.dataset.value}"
        )

    def _create_hupa_runner(
        self,
        experiment_name: str,
        configs: ExperimentConfigBundle,
        experiment_root: str | Path | None = None,
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
            experiment_root=experiment_root,
        )

    def _create_svd_runner(
        self,
        experiment_name: str,
        configs: ExperimentConfigBundle,
        experiment_root: str | Path | None = None,
    ) -> Any:
        from classes.experiment.runners.svd_experiment_runner import (
            SVDExperimentRunner,
        )

        training_config = self._svd_training_config(configs)

        return SVDExperimentRunner(
            dataset_root=self.settings.svd_root,
            data_root=self.settings.data_root,
            experiment_name=experiment_name,
            preprocess_config=configs.preprocess,
            feature_config=configs.features,
            manifest_config=configs.svd_manifest,
            training_config=training_config,
            experiment_root=experiment_root,
        )

    @staticmethod
    def _svd_training_config(
        configs: ExperimentConfigBundle,
    ) -> TrainingConfig:
        if len(configs.svd_manifest.vowels) == 1:
            return configs.training

        protocol_version = (
            "gpu_multivowel_extension_v1"
            if configs.training.compute_backend.uses_cuda
            else "cpu_multivowel_development_v1"
        )
        return replace(
            configs.training,
            protocol_version=protocol_version,
            evaluation_subgroup_col="vowel",
        )

    @staticmethod
    def _source_roots(
        experiment_root: Path | None,
        hupa_source_experiment: Path | None,
        svd_source_experiment: Path | None,
    ) -> dict[str, Path]:
        explicit_roots = {
            name: path
            for name, path in (
                ("hupa", hupa_source_experiment),
                ("svd", svd_source_experiment),
            )
            if path is not None
        }
        if explicit_roots and set(explicit_roots) != {"hupa", "svd"}:
            raise ValueError(
                "Both HUPA and SVD source experiment roots are "
                "required."
            )
        if experiment_root is None:
            return explicit_roots

        config_path = experiment_root / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(
                f"Resume config does not exist: {config_path}"
            )

        config = json.loads(config_path.read_text(encoding="utf-8"))
        roots: dict[str, Path] = {}

        for key, name in (
            ("hupa_experiment_root", "hupa"),
            ("svd_experiment_root", "svd"),
        ):
            value = config.get(key)
            if value:
                roots[name] = Path(value)

        if set(roots) != {"hupa", "svd"}:
            raise ValueError(
                "Cross/pooled resume config must reference both source "
                "experiment roots."
            )

        if explicit_roots and explicit_roots != roots:
            raise ValueError(
                "Explicit source experiment roots do not match the "
                "resumed cross/pooled experiment config."
            )

        return roots

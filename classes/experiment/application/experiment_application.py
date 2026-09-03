from __future__ import annotations

from typing import Any

from classes.experiment.application.experiment_config_factory import (
    ExperimentConfigFactory,
)
from classes.experiment.application.experiment_request import (
    ExperimentRequest,
)
from classes.experiment.application.experiment_runner_factory import (
    ExperimentRunnerFactory,
)
from classes.experiment.application.experiment_settings import (
    ExperimentSettings,
)
from classes.experiment.training.compute_backend_runtime import (
    activate_compute_backend,
)


class ExperimentApplication:
    def __init__(self, settings: ExperimentSettings) -> None:
        self.settings = settings

    def run(self, request: ExperimentRequest) -> Any:
        if request.stage.includes_training:
            activate_compute_backend(request.compute_backend)

        configs = ExperimentConfigFactory.build(
            compute_backend=request.compute_backend,
            svd_vowels=request.svd_vowels,
        )
        runner = ExperimentRunnerFactory(
            settings=self.settings,
        ).create(
            request=request,
            configs=configs,
        )

        return runner.run(stage=request.stage)

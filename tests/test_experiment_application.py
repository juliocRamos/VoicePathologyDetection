import argparse
import unittest
from pathlib import Path
from unittest.mock import Mock, call, patch

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
from classes.experiment.application.experiment_runner_factory import (
    ExperimentRunnerFactory,
)
from classes.experiment.application.experiment_settings import (
    ExperimentSettings,
)
from classes.experiment.runners.experiment_stage import ExperimentStage
from classes.experiment.training.compute_backend import ComputeBackend


class ExperimentRequestTests(unittest.TestCase):
    def test_builds_typed_request_and_default_name(self) -> None:
        namespace = argparse.Namespace(
            dataset="cross",
            stage="train",
            compute_backend="cuda",
            experiment_name=None,
        )

        request = ExperimentRequest.from_namespace(namespace)

        self.assertIs(request.dataset, ExperimentDataset.CROSS)
        self.assertIs(request.stage, ExperimentStage.TRAIN)
        self.assertIs(
            request.compute_backend,
            ComputeBackend.CUDA,
        )
        self.assertEqual(
            request.experiment_name,
            "cross_pre16k_rms20_fullsignal_features_v1",
        )


class ExperimentSettingsTests(unittest.TestCase):
    def test_default_uses_given_project_root_for_data(self) -> None:
        project_root = Path("/tmp/voice-pathology-project")

        settings = ExperimentSettings.default(project_root)

        self.assertEqual(settings.project_root, project_root)
        self.assertEqual(
            settings.data_root,
            project_root / "data",
        )


class ExperimentConfigFactoryTests(unittest.TestCase):
    def test_cuda_training_uses_one_cpu_job(self) -> None:
        configs = ExperimentConfigFactory.build(
            compute_backend=ComputeBackend.CUDA,
        )

        self.assertEqual(configs.training.n_jobs, 1)
        self.assertTrue(configs.hupa_manifest.adults_only)
        self.assertTrue(configs.svd_manifest.adults_only)
        self.assertEqual(configs.svd_manifest.vowels, ("a",))
        self.assertEqual(configs.svd_manifest.conditions, ("n",))


class ExperimentRunnerFactoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = ExperimentSettings(
            project_root=Path("/project"),
            data_root=Path("/project/data"),
            hupa_root=Path("/datasets/hupa"),
            svd_root=Path("/datasets/svd"),
        )
        self.configs = ExperimentConfigFactory.build(
            compute_backend=ComputeBackend.CPU,
        )
        self.factory = ExperimentRunnerFactory(self.settings)

    @patch.object(
        ExperimentRunnerFactory,
        "_create_svd_runner",
    )
    @patch.object(
        ExperimentRunnerFactory,
        "_create_hupa_runner",
    )
    def test_cross_creates_named_source_runners(
        self,
        create_hupa: Mock,
        create_svd: Mock,
    ) -> None:
        request = ExperimentRequest(
            dataset=ExperimentDataset.CROSS,
            stage=ExperimentStage.TRAIN,
            compute_backend=ComputeBackend.CPU,
            experiment_name="cross_test",
        )

        with patch(
            "classes.experiment.runners."
            "cross_database_experiment_runner."
            "CrossDatabaseExperimentRunner"
        ) as cross_runner_class:
            runner = self.factory.create(request, self.configs)

        create_hupa.assert_called_once_with(
            experiment_name="cross_test_hupa",
            configs=self.configs,
        )
        create_svd.assert_called_once_with(
            experiment_name="cross_test_svd",
            configs=self.configs,
        )
        cross_runner_class.assert_called_once_with(
            hupa_runner=create_hupa.return_value,
            svd_runner=create_svd.return_value,
            data_root=self.settings.data_root,
            experiment_name="cross_test",
            training_config=self.configs.training,
        )
        self.assertIs(runner, cross_runner_class.return_value)


class ExperimentApplicationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = ExperimentSettings(
            project_root=Path("/project"),
            data_root=Path("/project/data"),
            hupa_root=Path("/datasets/hupa"),
            svd_root=Path("/datasets/svd"),
        )

    @patch(
        "classes.experiment.application.experiment_application."
        "ExperimentRunnerFactory"
    )
    @patch(
        "classes.experiment.application.experiment_application."
        "ExperimentConfigFactory"
    )
    @patch(
        "classes.experiment.application.experiment_application."
        "activate_compute_backend"
    )
    def test_activates_backend_before_building_training_runner(
        self,
        activate_backend: Mock,
        config_factory: Mock,
        runner_factory_class: Mock,
    ) -> None:
        request = ExperimentRequest(
            dataset=ExperimentDataset.HUPA,
            stage=ExperimentStage.TRAIN,
            compute_backend=ComputeBackend.CUDA,
            experiment_name="test",
        )
        runner = runner_factory_class.return_value.create.return_value
        manager = Mock()
        manager.attach_mock(activate_backend, "activate")
        manager.attach_mock(config_factory.build, "build")
        manager.attach_mock(
            runner_factory_class.return_value.create,
            "create",
        )
        manager.attach_mock(runner.run, "run")

        ExperimentApplication(self.settings).run(request)

        self.assertEqual(
            manager.mock_calls,
            [
                call.activate(ComputeBackend.CUDA),
                call.build(compute_backend=ComputeBackend.CUDA),
                call.create(
                    request=request,
                    configs=config_factory.build.return_value,
                ),
                call.run(stage=ExperimentStage.TRAIN),
            ],
        )

    @patch(
        "classes.experiment.application.experiment_application."
        "ExperimentRunnerFactory"
    )
    @patch(
        "classes.experiment.application.experiment_application."
        "ExperimentConfigFactory"
    )
    @patch(
        "classes.experiment.application.experiment_application."
        "activate_compute_backend"
    )
    def test_prepare_does_not_activate_compute_backend(
        self,
        activate_backend: Mock,
        config_factory: Mock,
        runner_factory_class: Mock,
    ) -> None:
        request = ExperimentRequest(
            dataset=ExperimentDataset.SVD,
            stage=ExperimentStage.PREPARE,
            compute_backend=ComputeBackend.CPU,
            experiment_name="test",
        )

        ExperimentApplication(self.settings).run(request)

        activate_backend.assert_not_called()
        runner = runner_factory_class.return_value.create.return_value
        runner.run.assert_called_once_with(
            stage=ExperimentStage.PREPARE,
        )


if __name__ == "__main__":
    unittest.main()

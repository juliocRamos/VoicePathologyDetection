import argparse
import json
from tempfile import TemporaryDirectory
from types import SimpleNamespace
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
from classes.dataset.preparation.svd_training_manifest_builder import (
    SVDTrainingManifestConfig,
)
from classes.experiment.runners.cross_database_direction import (
    CrossDatabaseDirection,
)
from classes.experiment.runners.experiment_stage import ExperimentStage
from classes.experiment.runners.svd_experiment_runner import (
    SVDExperimentRunner,
)
from classes.experiment.training.compute_backend import ComputeBackend
from classes.experiment.training.training_config import TrainingConfig


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
        self.assertIsNone(request.resume_experiment)
        self.assertIsNone(request.hupa_source_experiment)
        self.assertIsNone(request.svd_source_experiment)
        self.assertEqual(request.svd_vowels, ("a",))
        self.assertIs(
            request.cross_direction,
            CrossDatabaseDirection.BOTH,
        )

    def test_builds_multivowel_directional_cross_request(
        self,
    ) -> None:
        namespace = argparse.Namespace(
            dataset="cross",
            stage="train",
            compute_backend="cuda",
            experiment_name="svd_multivowel_to_hupa",
            svd_vowels=["u", "a", "i"],
            cross_direction="svd-to-hupa",
        )

        request = ExperimentRequest.from_namespace(namespace)

        self.assertEqual(request.svd_vowels, ("a", "i", "u"))
        self.assertIs(
            request.cross_direction,
            CrossDatabaseDirection.SVD_TO_HUPA,
        )

    def test_rejects_direction_for_non_cross_dataset(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "only be changed for the cross dataset",
        ):
            ExperimentRequest(
                dataset=ExperimentDataset.SVD,
                stage=ExperimentStage.TRAIN,
                compute_backend=ComputeBackend.CPU,
                experiment_name="invalid",
                cross_direction=(
                    CrossDatabaseDirection.SVD_TO_HUPA
                ),
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
        self.assertTrue(
            configs.training.run_grouped_svm_learning_curve
        )
        self.assertTrue(
            configs.training.run_repeated_nested_cv
        )
        self.assertEqual(configs.training.nested_cv_folds, 3)
        self.assertEqual(configs.training.nested_cv_repeats, 2)
        self.assertEqual(
            configs.training.protocol_version,
            "gpu_confirmatory_v2",
        )
        self.assertTrue(
            configs.training.eligible_for_final_reporting
        )

    def test_builds_multivowel_svd_manifest(self) -> None:
        configs = ExperimentConfigFactory.build(
            compute_backend=ComputeBackend.CPU,
            svd_vowels=("a", "i", "u"),
        )

        self.assertEqual(
            configs.svd_manifest.vowels,
            ("a", "i", "u"),
        )


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
            experiment_root=None,
        )
        create_svd.assert_called_once_with(
            experiment_name="cross_test_svd",
            configs=self.configs,
            experiment_root=None,
        )
        cross_runner_class.assert_called_once_with(
            hupa_runner=create_hupa.return_value,
            svd_runner=create_svd.return_value,
            data_root=self.settings.data_root,
            experiment_name="cross_test",
            training_config=self.configs.training,
            direction=CrossDatabaseDirection.BOTH,
            experiment_root=None,
        )
        self.assertIs(runner, cross_runner_class.return_value)

    @patch.object(
        ExperimentRunnerFactory,
        "_create_svd_runner",
    )
    @patch.object(
        ExperimentRunnerFactory,
        "_create_hupa_runner",
    )
    def test_cross_reuses_explicit_source_experiments(
        self,
        create_hupa: Mock,
        create_svd: Mock,
    ) -> None:
        request = ExperimentRequest(
            dataset=ExperimentDataset.CROSS,
            stage=ExperimentStage.TRAIN,
            compute_backend=ComputeBackend.CPU,
            experiment_name="cross_test",
            hupa_source_experiment=Path("/runs/hupa"),
            svd_source_experiment=Path("/runs/svd"),
        )

        with patch(
            "classes.experiment.runners."
            "cross_database_experiment_runner."
            "CrossDatabaseExperimentRunner"
        ):
            self.factory.create(request, self.configs)

        create_hupa.assert_called_once_with(
            experiment_name="cross_test_hupa",
            configs=self.configs,
            experiment_root=Path("/runs/hupa"),
        )
        create_svd.assert_called_once_with(
            experiment_name="cross_test_svd",
            configs=self.configs,
            experiment_root=Path("/runs/svd"),
        )

    @patch(
        "classes.experiment.runners.svd_experiment_runner."
        "SVDExperimentRunner"
    )
    def test_multivowel_svd_reports_metrics_by_vowel(
        self,
        runner_class: Mock,
    ) -> None:
        configs = ExperimentConfigFactory.build(
            compute_backend=ComputeBackend.CPU,
            svd_vowels=("a", "i", "u"),
        )
        request = ExperimentRequest(
            dataset=ExperimentDataset.SVD,
            stage=ExperimentStage.TRAIN,
            compute_backend=ComputeBackend.CPU,
            experiment_name="svd_multivowel",
            svd_vowels=("a", "i", "u"),
        )

        runner = self.factory.create(request, configs)

        training_config = runner_class.call_args.kwargs[
            "training_config"
        ]
        self.assertEqual(
            training_config.evaluation_subgroup_col,
            "vowel",
        )
        self.assertEqual(
            training_config.protocol_version,
            "cpu_multivowel_development_v1",
        )
        self.assertIs(runner, runner_class.return_value)


class SVDResumeConfigurationTests(unittest.TestCase):
    def test_rejects_resume_with_different_vowel_cohort(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            config_path.write_text(
                json.dumps({
                    "dataset_name": "SVD",
                    "manifest_config": {
                        "vowels": ["a"],
                        "conditions": ["n"],
                    },
                    "training_config": {
                        "protocol_version": "development_v1",
                        "compute_backend": "cpu",
                        "random_state": 42,
                        "evaluation_subgroup_col": "vowel",
                    },
                }),
                encoding="utf-8",
            )
            runner = SVDExperimentRunner.__new__(
                SVDExperimentRunner
            )
            runner.paths = SimpleNamespace(
                config_path=config_path
            )
            runner.manifest_config = SVDTrainingManifestConfig(
                vowels=("a", "i", "u"),
                conditions=("n",),
            )
            runner.training_config = TrainingConfig(
                group_col="speaker_id",
                evaluation_subgroup_col="vowel",
            )

            with self.assertRaisesRegex(
                ValueError,
                "does not match",
            ):
                runner._validate_resume_config()


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
                call.build(
                    compute_backend=ComputeBackend.CUDA,
                    svd_vowels=("a",),
                ),
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

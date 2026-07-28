import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from classes.experiment.training.experimental_protocol_writer import (
    ExperimentalProtocolWriter,
)
from classes.experiment.training.training_config import TrainingConfig
from classes.experiment.training.compute_backend import ComputeBackend
from classes.experiment.training.training_plan import (
    FeatureScenario,
    ModelSpec,
)


class ExperimentalProtocolWriterTests(unittest.TestCase):
    def test_hash_is_stable_and_artifacts_are_written(self) -> None:
        config = TrainingConfig(
            protocol_version="test_protocol_v1",
        )
        scenarios = [
            FeatureScenario(
                name="mfcc",
                include_prefixes=("mfcc",),
            )
        ]
        model_specs = [
            ModelSpec(
                name="svm_linear",
                estimator=SVC(kernel="linear"),
                param_grid=[{
                    "scaler": [StandardScaler()],
                    "classifier__C": [0.1, 1.0],
                }],
                rationale="Test rationale.",
            )
        ]

        with TemporaryDirectory() as directory:
            writer = ExperimentalProtocolWriter(directory)
            first_hash = writer.write(
                config=config,
                feature_scenarios=scenarios,
                model_specs=model_specs,
            )
            second_hash = writer.write(
                config=config,
                feature_scenarios=scenarios,
                model_specs=model_specs,
            )

            self.assertEqual(first_hash, second_hash)
            artifact_path = (
                Path(directory) / "experimental_protocol.json"
            )
            artifact = json.loads(
                artifact_path.read_text(encoding="utf-8")
            )
            self.assertEqual(
                artifact["protocol_hash"],
                first_hash,
            )
            self.assertEqual(
                artifact["protocol_version"],
                "test_protocol_v1",
            )
            self.assertEqual(
                artifact["seed_policy"]["holdout_repetitions"],
                1,
            )
            self.assertEqual(
                artifact["seed_policy"]["nested_outer_seeds"],
                [42, 43],
            )
            self.assertIn(
                "execution_environment",
                artifact,
            )
            self.assertTrue(
                (
                    Path(directory)
                    / "experimental_protocol.md"
                ).exists()
            )

    def test_confirmatory_protocol_rejects_divergent_preprocessing(
        self,
    ) -> None:
        config = TrainingConfig(
            protocol_version="gpu_confirmatory_test",
            compute_backend=ComputeBackend.CUDA,
            n_jobs=1,
            eligible_for_final_reporting=True,
        )
        model_spec = ModelSpec(
            name="svm_linear",
            estimator=SVC(kernel="linear"),
            param_grid=[{
                "imputer__strategy": ["median"],
                "scaler": ["passthrough"],
                "selector": ["passthrough"],
                "classifier__C": [1.0],
            }],
        )

        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                ValueError,
                "must share",
            ):
                ExperimentalProtocolWriter(directory).write(
                    config=config,
                    feature_scenarios=[],
                    model_specs=[model_spec],
                )


if __name__ == "__main__":
    unittest.main()

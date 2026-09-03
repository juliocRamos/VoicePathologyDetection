from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from classes.experiment.runners.model_training_runner import (
    ModelTrainingRunner,
)
from classes.experiment.runners.pooled_database_experiment_runner import (
    PooledDatabaseExperimentRunner,
)
from classes.experiment.training.training_config import TrainingConfig
from classes.experiment.training.training_plan import (
    FeatureScenario,
    ModelSpec,
)


class PooledDatabaseExperimentTests(unittest.TestCase):
    @staticmethod
    def _database_rows(database: str) -> pd.DataFrame:
        rows = []

        for speaker_index in range(16):
            target = speaker_index % 2
            label = "pathological" if target else "healthy"

            for session_index in range(2):
                rows.append({
                    "sample_id": (
                        f"{database}-{speaker_index}-{session_index}"
                    ),
                    "base": database,
                    "speaker_id": f"speaker-{speaker_index}",
                    "label": label,
                    "age": 20.0 + speaker_index,
                    "sex": (
                        "female"
                        if speaker_index % 2
                        else "male"
                    ),
                    "pathology": label,
                    "status": "ok",
                    "mfcc_shared": (
                        float(target) + session_index * 0.01
                    ),
                    f"{database.lower()}_only_feature": 1.0,
                })

        return pd.DataFrame(rows)

    @staticmethod
    def _source_runner(path: Path) -> SimpleNamespace:
        return SimpleNamespace(
            paths=SimpleNamespace(root_dir=path)
        )

    def test_pooled_protocol_is_grouped_and_reported_by_database(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            data_root = Path(directory)
            pooled_runner = PooledDatabaseExperimentRunner(
                hupa_runner=self._source_runner(data_root / "hupa"),
                svd_runner=self._source_runner(data_root / "svd"),
                data_root=data_root,
                experiment_name="unit",
                training_config=TrainingConfig(
                    group_col="speaker_id",
                    cv_folds=3,
                    n_jobs=1,
                    bootstrap_iterations=0,
                    save_models=False,
                    save_predictions=True,
                    save_cv_results=False,
                    save_split_assignments=True,
                ),
            )
            svd = self._database_rows("SVD")
            svd.loc[
                svd["sample_id"].eq("SVD-0-1"),
                ["label", "pathology"],
            ] = ["pathological", "pathological"]
            pooled = pooled_runner._build_pooled_features(
                hupa_features=self._database_rows("HUPA"),
                svd_features=svd,
            )

            self.assertIn("mfcc_shared", pooled.columns)
            self.assertNotIn("hupa_only_feature", pooled.columns)
            self.assertNotIn("svd_only_feature", pooled.columns)
            self.assertTrue(
                pooled["database_speaker_id"]
                .str.startswith(("HUPA::", "SVD::"))
                .all()
            )
            mixed_label_report = pd.read_csv(
                pooled_runner.paths.reports_dir
                / "pooled_mixed_label_groups.csv"
            )
            self.assertEqual(len(mixed_label_report), 1)
            self.assertEqual(
                mixed_label_report.iloc[0][
                    "database_speaker_id"
                ],
                "SVD::speaker-0",
            )
            self.assertEqual(
                mixed_label_report.iloc[0]["n_samples"],
                2,
            )

            output_dir = data_root / "pooled_training"
            training_runner = ModelTrainingRunner(
                features_df=pooled,
                output_dir=output_dir,
                config=pooled_runner.training_config,
                train_dataset_name="HUPA+SVD",
                test_dataset_name="HUPA+SVD",
                feature_scenarios=[
                    FeatureScenario(
                        name="mfcc",
                        include_prefixes=("mfcc",),
                    )
                ],
                model_specs=[
                    ModelSpec(
                        name="dummy",
                        estimator=DummyClassifier(
                            strategy="most_frequent",
                        ),
                        param_grid=[{
                            "imputer__strategy": ["median"],
                            "scaler": ["passthrough"],
                            "selector": ["passthrough"],
                        }],
                    ),
                    ModelSpec(
                        name="logistic_regression",
                        estimator=LogisticRegression(
                            random_state=42,
                            max_iter=1_000,
                        ),
                        param_grid=[{
                            "imputer__strategy": ["median"],
                            "scaler": ["passthrough"],
                            "selector": ["passthrough"],
                            "classifier__C": [1.0],
                        }],
                    ),
                ],
            )
            metrics = training_runner.run()

            self.assertEqual(
                set(metrics["evaluation_scope"]),
                {
                    "overall",
                    "base:HUPA",
                    "base:SVD",
                    "base:macro",
                },
            )
            self.assertEqual(
                set(metrics["model"]),
                {"logistic_regression"},
            )
            self.assertEqual(
                set(metrics["evaluation_mode"]),
                {"pooled_database_holdout"},
            )

            assignments = pd.read_csv(
                output_dir
                / "splits"
                / "holdout_assignments.csv"
            )
            train = assignments[
                assignments["partition"].eq("train")
            ]
            test = assignments[
                assignments["partition"].eq("test")
            ]
            self.assertFalse(
                set(train["database_speaker_id"])
                & set(test["database_speaker_id"])
            )
            self.assertEqual(
                set(zip(train["base"], train["target"])),
                {
                    ("HUPA", 0),
                    ("HUPA", 1),
                    ("SVD", 0),
                    ("SVD", 1),
                },
            )
            self.assertEqual(
                set(zip(test["base"], test["target"])),
                {
                    ("HUPA", 0),
                    ("HUPA", 1),
                    ("SVD", 0),
                    ("SVD", 1),
                },
            )


if __name__ == "__main__":
    unittest.main()

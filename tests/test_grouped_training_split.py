from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from classes.experiment.runners.model_training_runner import (
    ModelTrainingRunner,
)
from classes.experiment.training.training_config import TrainingConfig
from classes.experiment.training.training_plan import (
    FeatureScenario,
    ModelSpec,
)


class GroupedTrainingSplitTests(unittest.TestCase):
    def test_mlp_curves_use_grouped_validation_and_balanced_accuracy(
        self,
    ) -> None:
        rows = []

        for group_index in range(20):
            target = group_index % 2
            label = "pathological" if target else "healthy"

            for session_index in range(2):
                rows.append({
                    "sample_id": (
                        f"curve-{group_index}-{session_index}"
                    ),
                    "base": "TEST",
                    "speaker_id": f"speaker-{group_index}",
                    "label": label,
                    "status": "ok",
                    "mfcc_01_mean": (
                        float(target) + 0.01 * session_index
                    ),
                })

        with TemporaryDirectory() as directory:
            output_dir = Path(directory)
            runner = ModelTrainingRunner(
                features_df=pd.DataFrame(rows),
                output_dir=output_dir,
                config=TrainingConfig(
                    group_col="speaker_id",
                    cv_folds=2,
                    n_jobs=1,
                    bootstrap_iterations=0,
                    save_models=False,
                    save_predictions=False,
                    save_cv_results=False,
                ),
                feature_scenarios=[
                    FeatureScenario(
                        name="mfcc",
                        include_prefixes=("mfcc",),
                    )
                ],
                model_specs=[
                    ModelSpec(
                        name="mlp",
                        estimator=MLPClassifier(
                            hidden_layer_sizes=(4,),
                            max_iter=3,
                            random_state=42,
                        ),
                        param_grid=[{
                            "imputer__strategy": ["median"],
                            "scaler": ["passthrough"],
                            "selector": ["passthrough"],
                        }],
                        use_balanced_sample_weight=True,
                    )
                ],
            )

            runner.run()

            history = pd.read_csv(
                output_dir
                / "figures"
                / "training_curves"
                / "mfcc_mlp_training_curve.csv"
            )
            self.assertEqual(len(history), 3)
            self.assertTrue({
                "train_loss",
                "valid_loss",
                "train_accuracy",
                "valid_accuracy",
                "train_balanced_accuracy",
                "valid_balanced_accuracy",
            }.issubset(history.columns))

            assignments = pd.read_csv(
                output_dir
                / "splits"
                / "mfcc_mlp_curve_assignments.csv"
            )
            fit_groups = set(
                assignments.loc[
                    assignments["partition"] == "curve_fit",
                    "group",
                ]
            )
            validation_groups = set(
                assignments.loc[
                    assignments["partition"]
                    == "curve_validation",
                    "group",
                ]
            )
            self.assertFalse(fit_groups & validation_groups)

    def test_speaker_groups_do_not_cross_holdout_or_inner_cv(self) -> None:
        rows = []

        for group_index in range(40):
            label = (
                "pathological"
                if group_index % 2
                else "healthy"
            )

            for session_index in range(2):
                rows.append({
                    "sample_id": (
                        f"sample-{group_index}-{session_index}"
                    ),
                    "base": "TEST",
                    "speaker_id": f"speaker-{group_index}",
                    "label": label,
                    "status": "ok",
                    "mfcc_01_mean": float(group_index),
                })

        features = pd.DataFrame(rows)

        with TemporaryDirectory() as directory:
            runner = ModelTrainingRunner(
                features_df=features,
                output_dir=Path(directory),
                config=TrainingConfig(
                    group_col="speaker_id",
                    cv_folds=4,
                    n_jobs=1,
                    save_models=False,
                    save_predictions=False,
                    save_cv_results=False,
                    save_split_assignments=False,
                ),
                feature_scenarios=[],
                model_specs=[],
            )

            dataframe = runner._prepare_dataframe()
            split = runner._split_train_test(dataframe)

            train_groups = set(split.meta_train["speaker_id"])
            test_groups = set(split.meta_test["speaker_id"])
            self.assertFalse(train_groups & test_groups)

            inner_splits = runner._build_inner_cv_splits(
                X_train=split.X_train,
                y_train=split.y_train,
                groups_train=split.meta_train["speaker_id"],
            )

            for fit_indices, validation_indices in inner_splits:
                fit_groups = set(
                    split.meta_train["speaker_id"].iloc[
                        fit_indices
                    ]
                )
                validation_groups = set(
                    split.meta_train["speaker_id"].iloc[
                        validation_indices
                    ]
                )
                self.assertFalse(
                    fit_groups & validation_groups
                )

                self.assertEqual(
                    set(
                        split.y_train.iloc[
                            fit_indices
                        ].unique()
                    ),
                    {0, 1},
                )
                self.assertEqual(
                    set(
                        split.y_train.iloc[
                            validation_indices
                        ].unique()
                    ),
                    {0, 1},
                )

    def test_grouped_training_runs_end_to_end(self) -> None:
        rows = []

        for group_index in range(30):
            target = group_index % 2
            label = "pathological" if target else "healthy"

            for session_index in range(2):
                rows.append({
                    "sample_id": (
                        f"sample-{group_index}-{session_index}"
                    ),
                    "base": "TEST",
                    "speaker_id": f"speaker-{group_index}",
                    "label": label,
                    "age": 20 + group_index,
                    "status": "ok",
                    "mfcc_01_mean": (
                        float(target)
                        + 0.01 * session_index
                    ),
                })

        with TemporaryDirectory() as directory:
            output_dir = Path(directory)
            runner = ModelTrainingRunner(
                features_df=pd.DataFrame(rows),
                output_dir=output_dir,
                config=TrainingConfig(
                    group_col="speaker_id",
                    cv_folds=3,
                    n_jobs=1,
                    bootstrap_iterations=20,
                    save_models=False,
                    save_predictions=False,
                    save_cv_results=False,
                    save_split_assignments=True,
                ),
                feature_scenarios=[
                    FeatureScenario(
                        name="mfcc",
                        include_prefixes=("mfcc",),
                    )
                ],
                model_specs=[
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
                    )
                ],
            )

            metrics = runner.run()

            self.assertEqual(len(metrics), 1)
            self.assertEqual(
                metrics.iloc[0]["scenario"],
                "mfcc",
            )
            self.assertEqual(
                metrics.iloc[0]["model"],
                "logistic_regression",
            )
            self.assertNotIn(
                "age",
                runner._get_numeric_feature_columns(
                    runner._prepare_dataframe()
                ),
            )
            self.assertTrue(
                (
                    output_dir
                    / "splits"
                    / "holdout_assignments.csv"
                ).exists()
            )
            self.assertTrue(
                (
                    output_dir
                    / "metrics"
                    / "metrics.parquet"
                ).exists()
            )

    def test_external_database_is_used_only_as_test_data(self) -> None:
        training_rows = []
        test_rows = []

        for group_index in range(18):
            target = group_index % 2
            label = "pathological" if target else "healthy"

            for session_index in range(2):
                training_rows.append({
                    "sample_id": (
                        f"hupa-{group_index}-{session_index}"
                    ),
                    "base": "HUPA",
                    "speaker_id": f"hupa-speaker-{group_index}",
                    "label": label,
                    "status": "ok",
                    "mfcc_01_mean": (
                        float(target) + 0.01 * session_index
                    ),
                    "training_only_feature": 1.0,
                })

        for group_index in range(10):
            target = group_index % 2
            label = "pathological" if target else "healthy"
            test_rows.append({
                "sample_id": f"svd-{group_index}",
                "base": "SVD",
                "speaker_id": f"svd-speaker-{group_index}",
                "label": label,
                "status": "ok",
                "mfcc_01_mean": float(target),
                "test_only_feature": 2.0,
            })

        with TemporaryDirectory() as directory:
            output_dir = Path(directory)
            runner = ModelTrainingRunner(
                features_df=pd.DataFrame(training_rows),
                external_test_features_df=pd.DataFrame(test_rows),
                train_dataset_name="HUPA",
                test_dataset_name="SVD",
                output_dir=output_dir,
                config=TrainingConfig(
                    group_col="speaker_id",
                    cv_folds=3,
                    n_jobs=1,
                    bootstrap_iterations=10,
                    save_models=False,
                    save_predictions=True,
                    save_cv_results=False,
                    save_split_assignments=True,
                ),
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
                    )
                ],
            )

            metrics = runner.run()

            self.assertEqual(len(metrics), 1)
            self.assertEqual(
                metrics.iloc[0]["model"],
                "logistic_regression",
            )
            self.assertEqual(
                metrics.iloc[0]["evaluation_mode"],
                "cross_database",
            )
            self.assertEqual(
                metrics.iloc[0]["train_database"],
                "HUPA",
            )
            self.assertEqual(
                metrics.iloc[0]["test_database"],
                "SVD",
            )
            self.assertEqual(
                metrics.iloc[0]["n_train_samples"],
                len(training_rows),
            )
            self.assertEqual(
                metrics.iloc[0]["n_test_samples"],
                len(test_rows),
            )

            assignments = pd.read_csv(
                output_dir
                / "splits"
                / "holdout_assignments.csv"
            )
            self.assertEqual(
                set(
                    assignments.loc[
                        assignments["partition"] == "train",
                        "base",
                    ]
                ),
                {"HUPA"},
            )
            self.assertEqual(
                set(
                    assignments.loc[
                        assignments["partition"]
                        == "external_test",
                        "base",
                    ]
                ),
                {"SVD"},
            )
            self.assertTrue(
                (
                    output_dir
                    / "splits"
                    / "feature_schema.csv"
                ).exists()
            )
            source_selection = pd.read_csv(
                output_dir
                / "metrics"
                / "source_model_selection.csv"
            )
            self.assertEqual(len(source_selection), 2)
            selected = source_selection[
                source_selection[
                    "selected_for_evaluation"
                ]
            ]
            self.assertEqual(len(selected), 1)
            self.assertEqual(
                selected.iloc[0]["model"],
                "logistic_regression",
            )


if __name__ == "__main__":
    unittest.main()

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from classes.experiment.runners.model_training_runner import (
    ModelTrainingRunner,
)
from classes.experiment.training.training_config import TrainingConfig
from classes.experiment.training.training_plan import (
    FeatureScenario,
    ModelSpec,
)


class GroupedTrainingSplitTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()

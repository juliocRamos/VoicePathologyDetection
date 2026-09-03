import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from classes.experiment.runners.model_training_runner import (
    ModelTrainingRunner,
)
from classes.experiment.training.training_config import TrainingConfig
from classes.experiment.training.grouped_svm_learning_curve_runner import (
    GroupedSVMLearningCurveRunner,
)
from classes.experiment.training.training_plan import (
    FeatureScenario,
    ModelSpec,
)


class GroupedTrainingSplitTests(unittest.TestCase):
    @staticmethod
    def _build_grouped_rows(
        number_of_groups: int,
    ) -> list[dict[str, object]]:
        rows = []

        for group_index in range(number_of_groups):
            target = group_index % 2
            label = "pathological" if target else "healthy"

            for session_index in range(2):
                rows.append({
                    "sample_id": (
                        f"diagnostic-{group_index}-{session_index}"
                    ),
                    "base": "TEST",
                    "speaker_id": f"speaker-{group_index}",
                    "label": label,
                    "status": "ok",
                    "mfcc_01_mean": (
                        float(target)
                        + 0.01 * session_index
                    ),
                })

        return rows

    def test_learning_curve_supports_groups_with_multiple_strata(
        self,
    ) -> None:
        groups = pd.Series([
            "mixed",
            "mixed",
            "healthy-a",
            "healthy-b",
            "pathological-a",
            "pathological-b",
        ])
        strata = pd.Series([0, 1, 0, 0, 1, 1])
        candidate_indices = np.arange(len(groups))

        selected_indices = (
            GroupedSVMLearningCurveRunner._select_fit_indices(
                candidate_indices=candidate_indices,
                groups=groups,
                strata=strata,
                train_fraction=0.5,
                random_state=42,
            )
        )

        selected_groups = set(groups.iloc[selected_indices])
        self.assertEqual(
            set(strata.iloc[selected_indices]),
            {0, 1},
        )
        for group in selected_groups:
            expected_indices = set(
                np.flatnonzero(groups.to_numpy() == group)
            )
            self.assertTrue(
                expected_indices.issubset(set(selected_indices))
            )

    def test_multivowel_holdout_is_grouped_and_reported_by_vowel(
        self,
    ) -> None:
        rows = []

        for speaker_index in range(20):
            target = speaker_index % 2
            label = "pathological" if target else "healthy"

            for vowel_index, vowel in enumerate(("a", "i", "u")):
                rows.append({
                    "sample_id": f"{speaker_index}-{vowel}",
                    "base": "SVD",
                    "speaker_id": f"speaker-{speaker_index}",
                    "label": label,
                    "vowel": vowel,
                    "status": "ok",
                    "mfcc_01_mean": (
                        float(target) + 0.01 * vowel_index
                    ),
                })

        with TemporaryDirectory() as directory:
            output_dir = Path(directory)
            runner = ModelTrainingRunner(
                features_df=pd.DataFrame(rows),
                output_dir=output_dir,
                config=TrainingConfig(
                    group_col="speaker_id",
                    evaluation_subgroup_col="vowel",
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

            self.assertEqual(
                set(metrics["evaluation_scope"]),
                {
                    "overall",
                    "vowel:a",
                    "vowel:i",
                    "vowel:u",
                    "vowel:macro",
                },
            )
            assignments = pd.read_csv(
                output_dir
                / "splits"
                / "holdout_assignments.csv"
            )
            train_groups = set(
                assignments.loc[
                    assignments["partition"].eq("train"),
                    "speaker_id",
                ]
            )
            test_groups = set(
                assignments.loc[
                    assignments["partition"].eq("test"),
                    "speaker_id",
                ]
            )
            self.assertFalse(train_groups & test_groups)

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

    def test_strict_evaluation_separates_primary_and_family_comparison(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            output_dir = Path(directory)
            runner = ModelTrainingRunner(
                features_df=pd.DataFrame(
                    self._build_grouped_rows(20)
                ),
                output_dir=output_dir,
                config=TrainingConfig(
                    group_col="speaker_id",
                    cv_folds=2,
                    n_jobs=1,
                    bootstrap_iterations=0,
                    save_models=False,
                    save_predictions=True,
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
                        name="svm_linear",
                        estimator=SVC(
                            kernel="linear",
                            random_state=42,
                        ),
                        param_grid=[{
                            "imputer__strategy": ["median"],
                            "scaler": ["passthrough"],
                            "selector": ["passthrough"],
                            "classifier__C": [1.0],
                        }],
                    ),
                    ModelSpec(
                        name="mlp",
                        estimator=MLPClassifier(
                            hidden_layer_sizes=(4,),
                            max_iter=2,
                            random_state=42,
                        ),
                        param_grid=[{
                            "imputer__strategy": ["median"],
                            "scaler": ["passthrough"],
                            "selector": ["passthrough"],
                        }],
                        use_balanced_sample_weight=True,
                    ),
                ],
            )

            primary_metrics = runner.run()
            comparison = runner.family_comparison_metrics_df

            self.assertEqual(len(primary_metrics), 1)
            self.assertEqual(
                set(comparison["model_family"]),
                {"svm", "mlp"},
            )
            self.assertEqual(
                set(comparison["comparison_role"]),
                {
                    "primary_global_and_family_champion",
                    "secondary_family_champion",
                },
            )
            self.assertEqual(
                int(
                    comparison[
                        "selected_for_primary_evaluation"
                    ].sum()
                ),
                1,
            )
            self.assertTrue(
                (
                    output_dir
                    / "metrics"
                    / "family_comparison_metrics.csv"
                ).exists()
            )
            self.assertTrue(
                (
                    output_dir
                    / "predictions"
                    / "mfcc_svm_linear_predictions.csv"
                ).exists()
            )
            self.assertTrue(
                (
                    output_dir
                    / "predictions"
                    / "mfcc_mlp_predictions.csv"
                ).exists()
            )

    def test_resume_restores_main_search_without_rerunning_grid(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            output_dir = Path(directory)
            features = pd.DataFrame(
                self._build_grouped_rows(20)
            )
            config = TrainingConfig(
                group_col="speaker_id",
                cv_folds=2,
                n_jobs=1,
                bootstrap_iterations=0,
                run_grouped_svm_learning_curve=False,
                run_repeated_nested_cv=True,
                nested_cv_folds=2,
                nested_cv_repeats=1,
                save_models=False,
                save_predictions=False,
                save_cv_results=True,
            )
            scenarios = [
                FeatureScenario(
                    name="mfcc",
                    include_prefixes=("mfcc",),
                )
            ]
            specs = [
                ModelSpec(
                    name="svm_linear",
                    estimator=SVC(
                        kernel="linear",
                        random_state=42,
                    ),
                    param_grid=[{
                        "imputer__strategy": ["median"],
                        "scaler": ["passthrough"],
                        "selector": ["passthrough"],
                        "classifier__C": [0.1, 1.0],
                    }],
                ),
                ModelSpec(
                    name="mlp",
                    estimator=MLPClassifier(
                        hidden_layer_sizes=(4,),
                        max_iter=2,
                        random_state=42,
                    ),
                    param_grid=[{
                        "imputer__strategy": ["median"],
                        "scaler": ["passthrough"],
                        "selector": ["passthrough"],
                    }],
                    use_balanced_sample_weight=True,
                ),
            ]

            first = ModelTrainingRunner(
                features_df=features,
                output_dir=output_dir,
                config=config,
                feature_scenarios=scenarios,
                model_specs=specs,
            )
            expected = first.run()

            (output_dir / "metrics" / "metrics.csv").unlink()
            (output_dir / "metrics" / "metrics.parquet").unlink()
            for checkpoint in (
                output_dir / "checkpoints"
            ).glob("*.joblib"):
                checkpoint.unlink()

            resumed = ModelTrainingRunner(
                features_df=features,
                output_dir=output_dir,
                config=config,
                feature_scenarios=scenarios,
                model_specs=specs,
                resume=True,
            )
            with patch.object(
                resumed,
                "_run_grid_search",
                side_effect=AssertionError(
                    "resume reran the grid search"
                ),
            ):
                actual = resumed.run()

            self.assertEqual(len(actual), len(expected))
            self.assertTrue(
                np.isclose(
                    actual.iloc[0]["balanced_accuracy"],
                    expected.iloc[0]["balanced_accuracy"],
                )
            )
            self.assertTrue(
                (
                    output_dir / "metrics" / "metrics.csv"
                ).is_file()
            )
            nested = pd.read_csv(
                output_dir
                / "metrics"
                / "repeated_nested_cv_results.csv"
            )
            self.assertEqual(len(nested), 2)

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

    def test_selected_svm_saves_grouped_learning_curve(self) -> None:
        with TemporaryDirectory() as directory:
            output_dir = Path(directory)
            runner = ModelTrainingRunner(
                features_df=pd.DataFrame(
                    self._build_grouped_rows(30)
                ),
                output_dir=output_dir,
                config=TrainingConfig(
                    group_col="speaker_id",
                    cv_folds=3,
                    n_jobs=1,
                    bootstrap_iterations=0,
                    run_grouped_svm_learning_curve=True,
                    learning_curve_train_sizes=(0.5, 1.0),
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
                        name="svm_linear",
                        estimator=SVC(
                            kernel="linear",
                            class_weight="balanced",
                        ),
                        param_grid=[{
                            "imputer__strategy": ["median"],
                            "scaler": ["passthrough"],
                            "selector": ["passthrough"],
                            "classifier__C": [0.1, 1.0],
                        }],
                    )
                ],
            )

            runner.run()

            file_stem = (
                "mfcc_svm_linear_grouped_learning_curve"
            )
            curve = pd.read_csv(
                output_dir
                / "figures"
                / "learning_curves"
                / f"{file_stem}.csv"
            )
            self.assertEqual(len(curve), 6)
            self.assertTrue({
                "train_balanced_accuracy",
                "validation_balanced_accuracy",
                "generalization_gap",
            }.issubset(curve.columns))

            assignments = pd.read_csv(
                output_dir
                / "splits"
                / f"{file_stem}_assignments.csv"
            )

            for (_, _), fold_assignments in assignments.groupby(
                ["fold", "train_fraction"]
            ):
                fit_groups = set(
                    fold_assignments.loc[
                        fold_assignments["partition"]
                        == "curve_train",
                        "group",
                    ]
                )
                validation_groups = set(
                    fold_assignments.loc[
                        fold_assignments["partition"]
                        == "curve_validation",
                        "group",
                    ]
                )
                self.assertFalse(fit_groups & validation_groups)

    def test_resume_completes_svm_family_curve_when_mlp_is_primary(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            output_dir = Path(directory)
            features = pd.DataFrame(
                self._build_grouped_rows(30)
            )
            config = TrainingConfig(
                group_col="speaker_id",
                cv_folds=3,
                n_jobs=1,
                bootstrap_iterations=0,
                run_grouped_svm_learning_curve=True,
                learning_curve_train_sizes=(0.5, 1.0),
                save_models=False,
                save_predictions=False,
                save_cv_results=True,
            )
            scenarios = [
                FeatureScenario(
                    name="mfcc",
                    include_prefixes=("mfcc",),
                )
            ]
            specs = [
                ModelSpec(
                    name="svm_linear",
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
                    name="mlp",
                    estimator=LogisticRegression(
                        random_state=42,
                    ),
                    param_grid=[{
                        "imputer__strategy": ["median"],
                        "scaler": ["passthrough"],
                        "selector": ["passthrough"],
                        "classifier__C": [1.0],
                    }],
                ),
            ]
            curve_stem = (
                "mfcc_svm_linear_grouped_learning_curve"
            )

            runner = ModelTrainingRunner(
                features_df=features,
                output_dir=output_dir,
                config=config,
                feature_scenarios=scenarios,
                model_specs=specs,
            )
            with patch.object(
                runner,
                "_save_training_curves",
            ):
                metrics = runner.run()

            self.assertEqual(metrics.iloc[0]["model"], "mlp")
            curve_dir = output_dir / "figures" / "learning_curves"
            paths = (
                curve_dir / f"{curve_stem}.csv",
                curve_dir / f"{curve_stem}_summary.csv",
                curve_dir / f"{curve_stem}.png",
                output_dir
                / "splits"
                / f"{curve_stem}_assignments.csv",
            )
            for path in paths:
                self.assertTrue(path.is_file())
                path.unlink()

            resumed_runner = ModelTrainingRunner(
                features_df=features,
                output_dir=output_dir,
                config=config,
                feature_scenarios=scenarios,
                model_specs=specs,
                resume=True,
            )
            with (
                patch.object(
                    resumed_runner,
                    "_run_grid_search",
                    side_effect=AssertionError(
                        "Resume reran a completed grid search."
                    ),
                ),
                patch.object(
                    resumed_runner,
                    "_save_training_curves",
                ),
            ):
                resumed_runner.run()

            for path in paths:
                self.assertTrue(path.is_file())

    def test_repeated_nested_cv_is_source_only_and_grouped(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            output_dir = Path(directory)
            runner = ModelTrainingRunner(
                features_df=pd.DataFrame(
                    self._build_grouped_rows(40)
                ),
                output_dir=output_dir,
                config=TrainingConfig(
                    group_col="speaker_id",
                    cv_folds=2,
                    n_jobs=1,
                    bootstrap_iterations=0,
                    run_repeated_nested_cv=True,
                    nested_cv_folds=2,
                    nested_cv_repeats=2,
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
                        name="logistic_regression",
                        estimator=LogisticRegression(
                            random_state=42,
                            max_iter=1_000,
                        ),
                        param_grid=[{
                            "imputer__strategy": ["median"],
                            "scaler": ["passthrough"],
                            "selector": ["passthrough"],
                            "classifier__C": [0.1, 1.0],
                        }],
                    )
                ],
            )

            runner.run()

            nested_results = pd.read_csv(
                output_dir
                / "metrics"
                / "repeated_nested_cv_results.csv"
            )
            self.assertEqual(len(nested_results), 4)
            self.assertTrue({
                "outer_train_balanced_accuracy",
                "balanced_accuracy",
                "outer_generalization_gap",
            }.issubset(nested_results.columns))

            holdout = pd.read_csv(
                output_dir / "splits" / "holdout_assignments.csv"
            )
            holdout_ids = set(
                holdout.loc[
                    holdout["partition"] == "test",
                    "sample_id",
                ]
            )
            assignments = pd.read_csv(
                output_dir
                / "splits"
                / "repeated_nested_cv_assignments.csv"
            )
            self.assertFalse(
                holdout_ids & set(assignments["sample_id"])
            )

            for (_, _), fold_assignments in assignments.groupby(
                ["repeat", "outer_fold"]
            ):
                fit_groups = set(
                    fold_assignments.loc[
                        fold_assignments["partition"]
                        == "outer_train",
                        "group",
                    ]
                )
                validation_groups = set(
                    fold_assignments.loc[
                        fold_assignments["partition"]
                        == "outer_validation",
                        "group",
                    ]
                )
                self.assertFalse(fit_groups & validation_groups)

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
            self.assertTrue({
                "mean_train_score",
                "best_cv_score",
                "train_cv_generalization_gap",
                "global_selection_threshold",
            }.issubset(source_selection.columns))
            self.assertIn(
                "train_cv_generalization_gap",
                metrics.columns,
            )
            protocol = json.loads(
                (
                    output_dir / "experimental_protocol.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(
                selected.iloc[0]["protocol_hash"],
                protocol["protocol_hash"],
            )
            self.assertEqual(
                metrics.iloc[0]["protocol_hash"],
                protocol["protocol_hash"],
            )


if __name__ == "__main__":
    unittest.main()

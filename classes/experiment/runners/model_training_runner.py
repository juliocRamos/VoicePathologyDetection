from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedGroupKFold,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from classes.experiment.training.classification_metrics import (
    ClassificationMetrics,
)
from classes.experiment.training.training_config import TrainingConfig
from classes.experiment.training.training_plan import (
    FeatureScenario,
    ModelSpec,
)


@dataclass(frozen=True)
class TrainingDataSplit:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    meta_train: pd.DataFrame
    meta_test: pd.DataFrame


class ModelTrainingRunner:
    METADATA_COLUMNS = {
        "sample_id",
        "base",
        "filepath",
        "label",
        "speaker_id",
        "speaker_id_source",
        "recording_id",
        "sex",
        "age",
        "pathology",
        "pathology_code",
        "pathology_group",
        "pathology_groups",
        "vowel",
        "condition",
        "pitch",
        "sr",
        "duration",
        "file_sha256",
        "source_count",
        "is_consolidated_duplicate",
        "metadata_conflict_columns",
        "status",
        "error",
    }

    def __init__(
        self,
        features_df: pd.DataFrame,
        output_dir: str | Path,
        config: TrainingConfig,
        feature_scenarios: list[FeatureScenario],
        model_specs: list[ModelSpec],
    ):
        self.features_df = features_df
        self.output_dir = Path(output_dir)
        self.config = config
        self.feature_scenarios = feature_scenarios
        self.model_specs = model_specs

        self.models_dir = self.output_dir / "models"
        self.metrics_dir = self.output_dir / "metrics"
        self.predictions_dir = self.output_dir / "predictions"
        self.splits_dir = self.output_dir / "splits"

        self._create_output_dirs()

    def run(self) -> pd.DataFrame:
        df = self._prepare_dataframe()
        numeric_cols = self._get_numeric_feature_columns(df)

        split = self._split_train_test(df=df)

        if self.config.save_split_assignments:
            self._save_split_assignments(split)

        groups_train = self._get_training_groups(split)

        all_metrics: list[dict[str, Any]] = []

        for scenario in self.feature_scenarios:
            feature_cols = self._select_feature_columns(
                scenario=scenario,
                numeric_cols=numeric_cols,
            )

            if not feature_cols:
                print(f"[WARNING Scenario without features: {scenario.name}]")
                continue

            X_train_scenario = split.X_train[feature_cols]
            X_test_scenario = split.X_test[feature_cols]

            for model_spec in self.model_specs:
                print(
                    f"\nRunning train with scenario: {scenario.name} |"
                    f"model_spec: {model_spec.name} |"
                    f"features={len(feature_cols)}"
                )

                grid = self._run_grid_search(
                    model_spec=model_spec,
                    X_train=X_train_scenario,
                    y_train=split.y_train,
                    groups_train=groups_train,
                )

                best_model = grid.best_estimator_

                metrics = self._evaluate_model(
                    model=best_model,
                    model_name=model_spec.name,
                    scenario_name=scenario.name,
                    feature_cols=feature_cols,
                    X_test=X_test_scenario,
                    y_test=split.y_test,
                    meta_test=split.meta_test,
                    n_train_samples=len(split.y_train),
                    best_cv_score=grid.best_score_,
                    best_params=grid.best_params_,
                )

                all_metrics.append(metrics)

                if self.config.save_cv_results:
                    self._save_cv_results(
                        grid=grid,
                        scenario_name=scenario.name,
                        model_name=model_spec.name,
                    )

                if self.config.save_models:
                    self._save_model(
                        model=best_model,
                        scenario_name=scenario.name,
                        model_name=model_spec.name
                    )

        metrics_df = pd.DataFrame(all_metrics)
        self._save_metrics(metrics_df)

        return metrics_df

    def _create_output_dirs(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.predictions_dir.mkdir(parents=True, exist_ok=True)
        self.splits_dir.mkdir(parents=True, exist_ok=True)

    def _prepare_dataframe(self) -> pd.DataFrame:
        df = self.features_df.copy()

        if "status" in df.columns:
            df = df[df["status"] == "ok"].copy()

        df = df.dropna(subset=[self.config.label_col]).copy()

        labels = df[self.config.label_col].astype("string")
        unexpected_labels = set(labels.unique()) - {
            self.config.positive_label,
            self.config.negative_label,
        }

        if unexpected_labels:
            raise ValueError(
                "Unexpected labels found before training: "
                f"{sorted(str(value) for value in unexpected_labels)}"
            )

        df["target"] = labels.eq(
            self.config.positive_label
        ).astype(int)

        if df["target"].nunique() != 2:
            raise ValueError(
                "Training requires both binary target classes."
            )

        if self.config.group_col is not None:
            group_col = self.config.group_col

            if group_col not in df.columns:
                raise ValueError(
                    f"Configured group column '{group_col}' is missing "
                    "from the features dataframe."
                )

            missing_groups = (
                df[group_col].isna()
                | df[group_col]
                .astype("string")
                .str.strip()
                .eq("")
                .fillna(True)
            )

            if missing_groups.any():
                raise ValueError(
                    f"Configured group column '{group_col}' contains "
                    f"{int(missing_groups.sum())} missing values."
                )

        print("\nClass distribution:")
        print(df["target"].value_counts())

        return df

    def _get_numeric_feature_columns(self, df: pd.DataFrame) -> list[str]:
        metadata_columns = set(self.METADATA_COLUMNS)

        if self.config.group_col is not None:
            metadata_columns.add(self.config.group_col)

        candidates = [
            col for col in df.columns
            if col not in metadata_columns and col != "target"
        ]

        numeric_cols = [
            col for col in candidates
            if pd.api.types.is_numeric_dtype(df[col])
        ]

        return numeric_cols

    def _split_train_test(
        self,
        df: pd.DataFrame,
    ) -> TrainingDataSplit:
        metadata_columns = set(self.METADATA_COLUMNS)

        if self.config.group_col is not None:
            metadata_columns.add(self.config.group_col)

        metadata_cols = sorted(
            column
            for column in metadata_columns
            if column in df.columns
        )

        feature_cols = self._get_numeric_feature_columns(df)

        X = df[feature_cols]
        y = df["target"].copy()
        meta = df[metadata_cols].copy()

        if self.config.group_col is None:
            indices = np.arange(len(df))
            train_indices, test_indices = train_test_split(
                indices,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
                stratify=y,
            )
        else:
            groups = df[self.config.group_col]
            train_indices, test_indices = (
                self._select_grouped_holdout_indices(
                    X=X,
                    y=y,
                    groups=groups,
                )
            )

        split = TrainingDataSplit(
            X_train=X.iloc[train_indices].copy(),
            X_test=X.iloc[test_indices].copy(),
            y_train=y.iloc[train_indices].copy(),
            y_test=y.iloc[test_indices].copy(),
            meta_train=meta.iloc[train_indices].copy(),
            meta_test=meta.iloc[test_indices].copy(),
        )

        self._validate_outer_split(split)
        return split

    def _select_grouped_holdout_indices(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        groups: pd.Series,
    ) -> tuple[np.ndarray, np.ndarray]:
        requested_splits = int(round(1.0 / self.config.test_size))
        n_splits = max(2, requested_splits)
        unique_groups = int(groups.nunique())

        if unique_groups < n_splits:
            raise ValueError(
                "Not enough unique groups for grouped holdout: "
                f"groups={unique_groups}, required={n_splits}."
            )

        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=self.config.random_state,
        )

        overall_prevalence = float(y.mean())
        candidates: list[
            tuple[float, int, np.ndarray, np.ndarray]
        ] = []

        for fold_index, (train_indices, test_indices) in enumerate(
            splitter.split(X, y, groups)
        ):
            y_train = y.iloc[train_indices]
            y_test = y.iloc[test_indices]

            if y_train.nunique() != 2 or y_test.nunique() != 2:
                continue

            train_groups = set(groups.iloc[train_indices])
            test_groups = set(groups.iloc[test_indices])

            if train_groups & test_groups:
                raise RuntimeError(
                    "Grouped holdout produced overlapping groups."
                )

            size_error = abs(
                len(test_indices) / len(y)
                - self.config.test_size
            )
            prevalence_error = abs(
                float(y_test.mean()) - overall_prevalence
            )
            score = size_error + prevalence_error
            candidates.append(
                (
                    score,
                    fold_index,
                    train_indices,
                    test_indices,
                )
            )

        if not candidates:
            raise ValueError(
                "Could not create a grouped holdout containing both "
                "classes in train and test."
            )

        _, selected_fold, train_indices, test_indices = min(
            candidates,
            key=lambda candidate: (
                candidate[0],
                candidate[1],
            ),
        )

        print(
            "\nGrouped holdout:"
            f"\n  selected fold: {selected_fold}"
            f"\n  train samples: {len(train_indices)}"
            f"\n  test samples: {len(test_indices)}"
            f"\n  train groups: "
            f"{groups.iloc[train_indices].nunique()}"
            f"\n  test groups: "
            f"{groups.iloc[test_indices].nunique()}"
        )

        return train_indices, test_indices

    def _validate_outer_split(
        self,
        split: TrainingDataSplit,
    ) -> None:
        if split.y_train.nunique() != 2:
            raise RuntimeError(
                "Training partition does not contain both classes."
            )

        if split.y_test.nunique() != 2:
            raise RuntimeError(
                "Test partition does not contain both classes."
            )

        if self.config.group_col is None:
            return

        group_col = self.config.group_col
        train_groups = set(split.meta_train[group_col])
        test_groups = set(split.meta_test[group_col])
        overlap = train_groups & test_groups

        if overlap:
            preview = sorted(str(value) for value in overlap)[:20]
            raise RuntimeError(
                "Speaker leakage found between train and test: "
                f"{preview}"
            )

    def _get_training_groups(
        self,
        split: TrainingDataSplit,
    ) -> pd.Series | None:
        if self.config.group_col is None:
            return None

        return split.meta_train[self.config.group_col].copy()

    @staticmethod
    def _select_feature_columns(
        scenario: FeatureScenario,
        numeric_cols: list[str],
    ) -> list[str]:
        selected: list[str] = []

        for col in numeric_cols:
            include = any(
                col.startswith(prefix)
                for prefix in scenario.include_prefixes
            )

            exclude = any(
                col.startswith(prefix)
                for prefix in scenario.exclude_prefixes
            )

            if include and not exclude:
                selected.append(col)

        return selected

    @staticmethod
    def _build_pipeline(
        model_spec: ModelSpec,
    ) -> Pipeline:
        return Pipeline([
            ("imputer", SimpleImputer()),
            ("scaler", StandardScaler()),
            ("selector", "passthrough"),
            ("classifier", model_spec.estimator),
        ])

    def _run_grid_search(
        self,
        model_spec: ModelSpec,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        groups_train: pd.Series | None,
    ) -> GridSearchCV:
        pipeline = self._build_pipeline(model_spec)

        cv_splits = self._build_inner_cv_splits(
            X_train=X_train,
            y_train=y_train,
            groups_train=groups_train,
        )

        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=model_spec.param_grid,
            scoring=self.config.scoring,
            cv=cv_splits,
            n_jobs=self.config.n_jobs,
            refit=True,
            return_train_score=True,
        )

        fit_parameters: dict[str, Any] = {}

        if model_spec.use_balanced_sample_weight:
            fit_parameters["classifier__sample_weight"] = (
                compute_sample_weight(
                    class_weight="balanced",
                    y=y_train,
                )
            )

        grid.fit(
            X_train,
            y_train,
            **fit_parameters,
        )

        print(f"\nBest CV score: {grid.best_score_:.4f}")
        print(f"Best parameters: {grid.best_params_}")

        return grid

    def _build_inner_cv_splits(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        groups_train: pd.Series | None,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        if groups_train is None:
            splitter = StratifiedKFold(
                n_splits=self.config.cv_folds,
                shuffle=True,
                random_state=self.config.random_state,
            )
            splits = list(splitter.split(X_train, y_train))
        else:
            if groups_train.nunique() < self.config.cv_folds:
                raise ValueError(
                    "Not enough training groups for grouped cross-"
                    f"validation: groups={groups_train.nunique()}, "
                    f"folds={self.config.cv_folds}."
                )

            splitter = StratifiedGroupKFold(
                n_splits=self.config.cv_folds,
                shuffle=True,
                random_state=self.config.random_state,
            )
            splits = list(
                splitter.split(
                    X_train,
                    y_train,
                    groups_train,
                )
            )

        for fold_index, (fit_indices, validation_indices) in enumerate(
            splits
        ):
            fit_target = y_train.iloc[fit_indices]
            validation_target = y_train.iloc[validation_indices]

            if (
                fit_target.nunique() != 2
                or validation_target.nunique() != 2
            ):
                raise ValueError(
                    f"Inner CV fold {fold_index} does not contain both "
                    "classes in fit and validation partitions."
                )

            if groups_train is not None:
                fit_groups = set(groups_train.iloc[fit_indices])
                validation_groups = set(
                    groups_train.iloc[validation_indices]
                )

                if fit_groups & validation_groups:
                    raise RuntimeError(
                        f"Speaker leakage found in inner CV fold "
                        f"{fold_index}."
                    )

        return splits

    def _evaluate_model(
        self,
        model: Pipeline,
        model_name: str,
        scenario_name: str,
        feature_cols: list[str],
        X_test: pd.DataFrame,
        y_test: pd.Series,
        meta_test: pd.DataFrame,
        n_train_samples: int,
        best_cv_score: float,
        best_params: dict[str, Any],
    ) -> dict[str, Any]:

        y_pred = model.predict(X_test)
        y_score = self._get_model_score(model, X_test)

        metrics = ClassificationMetrics.compute_binary_metrics(
            y_true=y_test.to_numpy(),
            y_pred=y_pred,
            y_score=y_score,
        )

        if (
            self.config.group_col is not None
            and self.config.group_col in meta_test.columns
        ):
            bootstrap_groups = meta_test[
                self.config.group_col
            ].to_numpy()
        else:
            bootstrap_groups = np.arange(len(y_test))

        intervals = (
            ClassificationMetrics.compute_grouped_bootstrap_intervals(
                y_true=y_test.to_numpy(),
                y_pred=y_pred,
                y_score=y_score,
                groups=bootstrap_groups,
                iterations=self.config.bootstrap_iterations,
                confidence_level=self.config.confidence_level,
                random_state=self.config.random_state,
            )
        )
        metrics.update(intervals)

        metrics["scenario"] = scenario_name
        metrics["model"] = model_name
        metrics["n_features"] = len(feature_cols)
        metrics["n_train_samples"] = int(n_train_samples)
        metrics["n_test_samples"] = int(len(y_test))
        metrics["best_cv_score"] = float(best_cv_score)
        metrics["best_params"] = str(best_params)

        if self.config.save_predictions:
            self._save_predictions(
                model_name=model_name,
                scenario_name=scenario_name,
                meta_test=meta_test,
                y_test=y_test,
                y_pred=y_pred,
                y_score=y_score,
            )

        return metrics

    @staticmethod
    def _get_model_score(
        model: Pipeline,
        X_test: pd.DataFrame,
    ) -> np.ndarray | None:
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X_test)
            return probabilities[:, 1]

        if hasattr(model, "decision_function"):
            return model.decision_function(X_test)

        return None

    def _save_split_assignments(
        self,
        split: TrainingDataSplit,
    ) -> None:
        train_assignments = split.meta_train.copy()
        train_assignments["target"] = split.y_train.to_numpy()
        train_assignments["partition"] = "train"

        test_assignments = split.meta_test.copy()
        test_assignments["target"] = split.y_test.to_numpy()
        test_assignments["partition"] = "test"

        assignments = pd.concat(
            [train_assignments, test_assignments],
            ignore_index=True,
        )
        assignments.to_csv(
            self.splits_dir / "holdout_assignments.csv",
            index=False,
        )

    def _save_predictions(
        self,
        model_name: str,
        scenario_name: str,
        meta_test: pd.DataFrame,
        y_test: pd.Series,
        y_pred: np.ndarray,
        y_score: np.ndarray | None,
    ) -> None:
        predictions_df = meta_test.copy()
        predictions_df["y_true"] = y_test.to_numpy()
        predictions_df["y_pred"] = y_pred

        if y_score is not None:
            predictions_df["y_score"] = y_score

        output_path = (
            self.predictions_dir
            / f"{scenario_name}_{model_name}_predictions.csv"
        )

        predictions_df.to_csv(output_path, index=False)

    def _save_cv_results(
        self,
        grid: GridSearchCV,
        scenario_name: str,
        model_name: str,
    ) -> None:
        cv_results_df = pd.DataFrame(grid.cv_results_)

        output_path = (
            self.metrics_dir
            / f"{scenario_name}_{model_name}_cv_results.csv"
        )

        cv_results_df.to_csv(output_path, index=False)

    def _save_model(
        self,
        model: Pipeline,
        scenario_name: str,
        model_name: str,
    ) -> None:
        output_path = (
            self.models_dir
            / f"{scenario_name}_{model_name}.joblib"
        )

        joblib.dump(model, output_path)

    def _save_metrics(
        self,
        metrics_df: pd.DataFrame,
    ) -> None:
        csv_path = self.metrics_dir / "metrics.csv"
        parquet_path = self.metrics_dir / "metrics.parquet"

        metrics_df.to_csv(csv_path, index=False)
        metrics_df.to_parquet(parquet_path, index=False)

        print("\nMetrics saved in:")
        print(csv_path)

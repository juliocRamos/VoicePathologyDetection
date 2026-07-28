from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    log_loss,
)
from sklearn.model_selection import (
    GridSearchCV,
    ParameterGrid,
    StratifiedGroupKFold,
    StratifiedKFold,
    train_test_split,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from classes.experiment.training.classification_metrics import (
    ClassificationMetrics,
)
from classes.experiment.training.compute_backend_runtime import (
    ensure_compute_backend_ready,
)
from classes.experiment.training.host_array_converter import (
    HostArrayConverter,
)
from classes.experiment.training.training_config import TrainingConfig
from classes.experiment.training.training_plan import (
    FeatureScenario,
    ModelSpec,
)
from classes.plot.training_curve_visualizer import (
    TrainingCurveVisualizer,
)


@dataclass(frozen=True)
class TrainingDataSplit:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    meta_train: pd.DataFrame
    meta_test: pd.DataFrame


@dataclass(frozen=True)
class SourceSelectionCandidate:
    scenario: FeatureScenario
    model_spec: ModelSpec
    feature_cols: list[str]
    grid: GridSearchCV
    training_time_seconds: float
    order: int


class ModelTrainingRunner:
    SUBGROUP_MACRO_METRICS = (
        "accuracy",
        "balanced_accuracy",
        "uar",
        "precision",
        "sensitivity",
        "specificity",
        "f1",
        "macro_f1",
        "mcc",
        "auc",
        "pr_auc",
    )
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
        external_test_features_df: pd.DataFrame | None = None,
        train_dataset_name: str | None = None,
        test_dataset_name: str | None = None,
    ):
        ensure_compute_backend_ready(config.compute_backend)

        self.features_df = features_df
        self.output_dir = Path(output_dir)
        self.config = config
        self.feature_scenarios = feature_scenarios
        self.model_specs = model_specs
        self.external_test_features_df = external_test_features_df
        self.train_dataset_name = train_dataset_name
        self.test_dataset_name = test_dataset_name

        self.models_dir = self.output_dir / "models"
        self.metrics_dir = self.output_dir / "metrics"
        self.predictions_dir = self.output_dir / "predictions"
        self.splits_dir = self.output_dir / "splits"
        self.training_curves_dir = (
            self.output_dir / "figures" / "training_curves"
        )

        self._create_output_dirs()

        cache_location = (
            self.output_dir / ".pipeline_cache"
            if self.config.cache_pipeline_transformers
            else None
        )

        self.pipeline_memory = joblib.Memory(
            location=cache_location,
            verbose=0
        )

    def run(self) -> pd.DataFrame:
        train_df = self._prepare_dataframe(
            features_df=self.features_df,
            partition_name="training",
        )

        if self.external_test_features_df is None:
            numeric_cols = self._get_numeric_feature_columns(train_df)
            split = self._split_train_test(df=train_df)
            evaluation_mode = (
                "pooled_database_holdout"
                if (
                    self.config.stratify_col == "base"
                    and self.config.evaluation_subgroup_col == "base"
                )
                else "within_database_holdout"
            )
        else:
            test_df = self._prepare_dataframe(
                features_df=self.external_test_features_df,
                partition_name="external test",
            )
            numeric_cols = self._get_common_numeric_feature_columns(
                train_df=train_df,
                test_df=test_df,
            )
            split = self._build_external_test_split(
                train_df=train_df,
                test_df=test_df,
                feature_cols=numeric_cols,
            )
            evaluation_mode = "cross_database"

        if self.config.save_split_assignments:
            self._save_split_assignments(
                split=split,
                external_test=(
                    self.external_test_features_df is not None
                ),
            )

        groups_train = self._get_training_groups(split)
        strata_train = self._get_training_strata(split)

        if (
            self.external_test_features_df is not None
            or self.config.strict_model_selection
        ):
            metrics_df = self._run_strict_evaluation(
                split=split,
                numeric_cols=numeric_cols,
                groups_train=groups_train,
                strata_train=strata_train,
                evaluation_mode=evaluation_mode,
            )
            self._save_metrics(metrics_df)
            return metrics_df

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

                training_started_at = perf_counter()
                grid = self._run_grid_search(
                    model_spec=model_spec,
                    X_train=X_train_scenario,
                    y_train=split.y_train,
                    groups_train=groups_train,
                    strata_train=strata_train,
                )
                training_time_seconds = (
                    perf_counter() - training_started_at
                )

                best_model = grid.best_estimator_
                self._save_training_curves(
                    model=best_model,
                    scenario_name=scenario.name,
                    model_name=model_spec.name,
                    X_train=X_train_scenario,
                    y_train=split.y_train,
                    groups_train=groups_train,
                    strata_train=strata_train,
                )

                evaluation_metrics = self._evaluate_with_subgroups(
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
                    training_time_seconds=training_time_seconds,
                    evaluation_mode=evaluation_mode,
                )

                all_metrics.extend(evaluation_metrics)

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

    def _run_strict_evaluation(
        self,
        split: TrainingDataSplit,
        numeric_cols: list[str],
        groups_train: pd.Series | None,
        strata_train: pd.Series | None,
        evaluation_mode: str,
    ) -> pd.DataFrame:
        """Select on training CV, then evaluate one model on held-out data."""
        candidates: list[SourceSelectionCandidate] = []
        candidate_order = 0

        for scenario in self.feature_scenarios:
            feature_cols = self._select_feature_columns(
                scenario=scenario,
                numeric_cols=numeric_cols,
            )

            if not feature_cols:
                print(
                    f"[WARNING Scenario without features: "
                    f"{scenario.name}]"
                )
                continue

            X_train_scenario = split.X_train[feature_cols]

            for model_spec in self.model_specs:
                print(
                    "\nSelecting on training partition with scenario: "
                    f"{scenario.name} | model_spec: "
                    f"{model_spec.name} | features={len(feature_cols)}"
                )

                training_started_at = perf_counter()
                grid = self._run_grid_search(
                    model_spec=model_spec,
                    X_train=X_train_scenario,
                    y_train=split.y_train,
                    groups_train=groups_train,
                    strata_train=strata_train,
                )
                training_time_seconds = (
                    perf_counter() - training_started_at
                )

                candidate = SourceSelectionCandidate(
                    scenario=scenario,
                    model_spec=model_spec,
                    feature_cols=feature_cols,
                    grid=grid,
                    training_time_seconds=training_time_seconds,
                    order=candidate_order,
                )
                candidates.append(candidate)
                candidate_order += 1

                if self.config.save_cv_results:
                    self._save_cv_results(
                        grid=grid,
                        scenario_name=scenario.name,
                        model_name=model_spec.name,
                    )

        if not candidates:
            raise RuntimeError(
                "No source-database candidate was available for "
                "held-out evaluation."
            )

        eligible_candidates = [
            candidate
            for candidate in candidates
            if np.isfinite(float(candidate.grid.best_score_))
        ]

        if not eligible_candidates:
            raise RuntimeError(
                "Every source-database candidate produced a "
                "non-finite cross-validation score."
            )

        selected = max(
            eligible_candidates,
            key=lambda candidate: (
                float(candidate.grid.best_score_),
                -candidate.order,
            ),
        )
        self._save_source_selection_results(
            candidates=candidates,
            selected=selected,
        )

        print(
            "\nSelected exclusively by training-partition CV:"
            f"\n  scenario: {selected.scenario.name}"
            f"\n  model: {selected.model_spec.name}"
            f"\n  CV {self.config.scoring}: "
            f"{selected.grid.best_score_:.4f}"
        )

        best_model = selected.grid.best_estimator_
        self._save_training_curves(
            model=best_model,
            scenario_name=selected.scenario.name,
            model_name=selected.model_spec.name,
            X_train=split.X_train[selected.feature_cols],
            y_train=split.y_train,
            groups_train=groups_train,
            strata_train=strata_train,
        )
        mlp_candidates = [
            candidate
            for candidate in eligible_candidates
            if candidate.model_spec.name.lower() == "mlp"
        ]

        if mlp_candidates and selected.model_spec.name.lower() != "mlp":
            best_mlp = max(
                mlp_candidates,
                key=lambda candidate: (
                    float(candidate.grid.best_score_),
                    -candidate.order,
                ),
            )
            self._save_training_curves(
                model=best_mlp.grid.best_estimator_,
                scenario_name=best_mlp.scenario.name,
                model_name=best_mlp.model_spec.name,
                X_train=split.X_train[best_mlp.feature_cols],
                y_train=split.y_train,
                groups_train=groups_train,
                strata_train=strata_train,
            )

        metrics = self._evaluate_with_subgroups(
            model=best_model,
            model_name=selected.model_spec.name,
            scenario_name=selected.scenario.name,
            feature_cols=selected.feature_cols,
            X_test=split.X_test[selected.feature_cols],
            y_test=split.y_test,
            meta_test=split.meta_test,
            n_train_samples=len(split.y_train),
            best_cv_score=selected.grid.best_score_,
            best_params=selected.grid.best_params_,
            training_time_seconds=(
                selected.training_time_seconds
            ),
            evaluation_mode=evaluation_mode,
        )

        if self.config.save_models:
            self._save_model(
                model=best_model,
                scenario_name=selected.scenario.name,
                model_name=selected.model_spec.name,
            )

        return pd.DataFrame(metrics)

    def _save_source_selection_results(
        self,
        candidates: list[SourceSelectionCandidate],
        selected: SourceSelectionCandidate,
    ) -> None:
        rows = []

        for candidate in candidates:
            rows.append({
                "scenario": candidate.scenario.name,
                "model": candidate.model_spec.name,
                "n_features": len(candidate.feature_cols),
                "selection_metric": self.config.scoring,
                "best_cv_score": float(
                    candidate.grid.best_score_
                ),
                "best_params": str(candidate.grid.best_params_),
                "training_time_seconds": float(
                    candidate.training_time_seconds
                ),
                "selected_for_evaluation": (
                    candidate is selected
                ),
                "candidate_order": candidate.order,
            })

        selection_df = pd.DataFrame(rows)
        selection_df["source_cv_rank"] = (
            selection_df["best_cv_score"]
            .rank(
                method="first",
                ascending=False,
                na_option="bottom",
            )
            .astype(int)
        )
        selection_df = selection_df.sort_values(
            by=["source_cv_rank", "candidate_order"],
        )
        selection_df.to_csv(
            self.metrics_dir / "source_model_selection.csv",
            index=False,
        )
        selection_df.to_parquet(
            self.metrics_dir / "source_model_selection.parquet",
            index=False,
        )

    def _create_output_dirs(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.predictions_dir.mkdir(parents=True, exist_ok=True)
        self.splits_dir.mkdir(parents=True, exist_ok=True)
        self.training_curves_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def _prepare_dataframe(
        self,
        features_df: pd.DataFrame | None = None,
        partition_name: str = "training",
    ) -> pd.DataFrame:
        source = (
            self.features_df
            if features_df is None
            else features_df
        )
        df = source.copy()

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

        print(f"\nClass distribution ({partition_name}):")
        print(df["target"].value_counts())

        return df

    def _get_numeric_feature_columns(self, df: pd.DataFrame) -> list[str]:
        metadata_columns = set(self.METADATA_COLUMNS)

        if self.config.group_col is not None:
            metadata_columns.add(self.config.group_col)

        if self.config.stratify_col is not None:
            metadata_columns.add(self.config.stratify_col)

        if self.config.evaluation_subgroup_col is not None:
            metadata_columns.add(
                self.config.evaluation_subgroup_col
            )

        candidates = [
            col for col in df.columns
            if col not in metadata_columns and col != "target"
        ]

        numeric_cols = [
            col for col in candidates
            if pd.api.types.is_numeric_dtype(df[col])
        ]

        return numeric_cols

    def _get_common_numeric_feature_columns(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> list[str]:
        train_columns = self._get_numeric_feature_columns(train_df)
        test_columns = set(
            self._get_numeric_feature_columns(test_df)
        )
        common_columns = [
            column
            for column in train_columns
            if column in test_columns
        ]

        if not common_columns:
            raise ValueError(
                "Training and external-test dataframes do not share "
                "numeric acoustic features."
            )

        missing_in_test = sorted(set(train_columns) - test_columns)
        missing_in_train = sorted(
            set(self._get_numeric_feature_columns(test_df))
            - set(train_columns)
        )
        schema_report = pd.DataFrame([
            {
                "feature": column,
                "available_in_training": column in train_columns,
                "available_in_external_test": column in test_columns,
                "used": (
                    column in train_columns
                    and column in test_columns
                ),
            }
            for column in sorted(
                set(train_columns) | test_columns
            )
        ])
        schema_report.to_csv(
            self.splits_dir / "feature_schema.csv",
            index=False,
        )

        print(
            "\nCross-database feature schema:"
            f"\n  common features: {len(common_columns)}"
            f"\n  training-only features: {len(missing_in_test)}"
            f"\n  external-test-only features: "
            f"{len(missing_in_train)}"
        )
        return common_columns

    def _build_external_test_split(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        feature_cols: list[str],
    ) -> TrainingDataSplit:
        metadata_columns = set(self.METADATA_COLUMNS)

        if self.config.group_col is not None:
            metadata_columns.add(self.config.group_col)

        if self.config.stratify_col is not None:
            metadata_columns.add(self.config.stratify_col)

        if self.config.evaluation_subgroup_col is not None:
            metadata_columns.add(
                self.config.evaluation_subgroup_col
            )

        train_metadata_cols = sorted(
            column
            for column in metadata_columns
            if column in train_df.columns
        )
        test_metadata_cols = sorted(
            column
            for column in metadata_columns
            if column in test_df.columns
        )

        split = TrainingDataSplit(
            X_train=train_df[feature_cols].copy(),
            X_test=test_df[feature_cols].copy(),
            y_train=train_df["target"].copy(),
            y_test=test_df["target"].copy(),
            meta_train=train_df[train_metadata_cols].copy(),
            meta_test=test_df[test_metadata_cols].copy(),
        )

        if split.y_train.nunique() != 2:
            raise RuntimeError(
                "Cross-database training data does not contain both "
                "classes."
            )

        if split.y_test.nunique() != 2:
            raise RuntimeError(
                "Cross-database external test data does not contain "
                "both classes."
            )

        print(
            "\nExternal test split:"
            f"\n  training database: "
            f"{self.train_dataset_name or 'unspecified'}"
            f"\n  test database: "
            f"{self.test_dataset_name or 'unspecified'}"
            f"\n  train samples: {len(split.y_train)}"
            f"\n  test samples: {len(split.y_test)}"
        )
        return split

    def _split_train_test(
        self,
        df: pd.DataFrame,
    ) -> TrainingDataSplit:
        metadata_columns = set(self.METADATA_COLUMNS)

        if self.config.group_col is not None:
            metadata_columns.add(self.config.group_col)

        if self.config.stratify_col is not None:
            metadata_columns.add(self.config.stratify_col)

        if self.config.evaluation_subgroup_col is not None:
            metadata_columns.add(
                self.config.evaluation_subgroup_col
            )

        metadata_cols = sorted(
            column
            for column in metadata_columns
            if column in df.columns
        )

        feature_cols = self._get_numeric_feature_columns(df)

        X = df[feature_cols]
        y = df["target"].copy()
        meta = df[metadata_cols].copy()
        stratification_target = (
            self._build_stratification_target(
                metadata=df,
                target=y,
            )
        )

        if self.config.group_col is None:
            indices = np.arange(len(df))
            train_indices, test_indices = train_test_split(
                indices,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
                stratify=(
                    stratification_target
                    if stratification_target is not None
                    else y
                ),
            )
        else:
            groups = df[self.config.group_col]
            train_indices, test_indices = (
                self._select_grouped_holdout_indices(
                    X=X,
                    y=y,
                    groups=groups,
                    stratification_target=stratification_target,
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
        stratification_target: pd.Series | None = None,
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

        split_target = (
            stratification_target
            if stratification_target is not None
            else y
        )
        overall_distribution = (
            split_target.value_counts(normalize=True)
        )
        required_strata = set(split_target.unique())
        candidates: list[
            tuple[float, int, np.ndarray, np.ndarray]
        ] = []

        for fold_index, (train_indices, test_indices) in enumerate(
            splitter.split(X, split_target, groups)
        ):
            y_train = y.iloc[train_indices]
            y_test = y.iloc[test_indices]

            if y_train.nunique() != 2 or y_test.nunique() != 2:
                continue

            if stratification_target is not None:
                train_strata = set(
                    stratification_target.iloc[train_indices]
                )
                test_strata = set(
                    stratification_target.iloc[test_indices]
                )

                if (
                    train_strata != required_strata
                    or test_strata != required_strata
                ):
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
            test_distribution = (
                split_target.iloc[test_indices]
                .value_counts(normalize=True)
                .reindex(overall_distribution.index, fill_value=0.0)
            )
            distribution_error = float(
                (
                    test_distribution - overall_distribution
                ).abs().sum()
                / 2.0
            )
            score = size_error + distribution_error
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
                "Could not create a grouped holdout containing all "
                "required classes/strata in train and test."
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

        train_strata = self._build_stratification_target(
            metadata=split.meta_train,
            target=split.y_train,
        )
        test_strata = self._build_stratification_target(
            metadata=split.meta_test,
            target=split.y_test,
        )

        if (
            train_strata is not None
            and test_strata is not None
            and set(train_strata.unique())
            != set(test_strata.unique())
        ):
            raise RuntimeError(
                "Train and test partitions do not contain the same "
                "configured database/class strata."
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

    def _get_training_strata(
        self,
        split: TrainingDataSplit,
    ) -> pd.Series | None:
        return self._build_stratification_target(
            metadata=split.meta_train,
            target=split.y_train,
        )

    def _build_stratification_target(
        self,
        metadata: pd.DataFrame,
        target: pd.Series,
    ) -> pd.Series | None:
        stratify_col = self.config.stratify_col

        if stratify_col is None:
            return None

        if stratify_col not in metadata.columns:
            raise ValueError(
                f"Configured stratification column "
                f"'{stratify_col}' is missing."
            )

        values = metadata[stratify_col]
        missing_values = (
            values.isna()
            | values.astype("string").str.strip().eq("").fillna(True)
        )

        if missing_values.any():
            raise ValueError(
                f"Configured stratification column "
                f"'{stratify_col}' contains "
                f"{int(missing_values.sum())} missing values."
            )

        return (
            values.astype("string")
            + "::target="
            + target.astype("string")
        )

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

    def _build_pipeline(
        self,
        model_spec: ModelSpec,
    ) -> Pipeline:
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer()),
                ("scaler", StandardScaler()),
                ("host_array", HostArrayConverter()),
                ("selector", "passthrough"),
                ("classifier", model_spec.estimator),
            ],
            memory=self.pipeline_memory,
        )

    def _run_grid_search(
        self,
        model_spec: ModelSpec,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        groups_train: pd.Series | None,
        strata_train: pd.Series | None = None,
    ) -> GridSearchCV:
        pipeline = self._build_pipeline(model_spec)

        cv_splits = self._build_inner_cv_splits(
            X_train=X_train,
            y_train=y_train,
            groups_train=groups_train,
            strata_train=strata_train,
        )
        candidate_count = len(ParameterGrid(model_spec.param_grid))

        print(
            f"Grid candidates: {candidate_count} | "
            f"CV fits: {candidate_count * len(cv_splits)}"
        )

        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=model_spec.param_grid,
            scoring=self.config.scoring,
            cv=cv_splits,
            n_jobs=self.config.n_jobs,
            refit=True,
            return_train_score=False,
            verbose=self.config.grid_search_verbose,
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
        strata_train: pd.Series | None = None,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        split_target = (
            strata_train
            if strata_train is not None
            else y_train
        )

        if groups_train is None:
            splitter = StratifiedKFold(
                n_splits=self.config.cv_folds,
                shuffle=True,
                random_state=self.config.random_state,
            )
            splits = list(splitter.split(X_train, split_target))
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
                    split_target,
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

            if strata_train is not None:
                required_strata = set(strata_train.unique())
                fit_strata = set(strata_train.iloc[fit_indices])
                validation_strata = set(
                    strata_train.iloc[validation_indices]
                )

                if (
                    fit_strata != required_strata
                    or validation_strata != required_strata
                ):
                    raise ValueError(
                        f"Inner CV fold {fold_index} does not contain "
                        "all configured database/class strata."
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

    def _evaluate_with_subgroups(
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
        training_time_seconds: float,
        evaluation_mode: str,
    ) -> list[dict[str, Any]]:
        overall = self._evaluate_model(
            model=model,
            model_name=model_name,
            scenario_name=scenario_name,
            feature_cols=feature_cols,
            X_test=X_test,
            y_test=y_test,
            meta_test=meta_test,
            n_train_samples=n_train_samples,
            best_cv_score=best_cv_score,
            best_params=best_params,
            training_time_seconds=training_time_seconds,
            evaluation_mode=evaluation_mode,
            save_predictions=True,
        )
        overall["evaluation_scope"] = "overall"
        metrics = [overall]

        subgroup_col = self.config.evaluation_subgroup_col

        if subgroup_col is None:
            return metrics

        if subgroup_col not in meta_test.columns:
            raise ValueError(
                f"Configured evaluation subgroup column "
                f"'{subgroup_col}' is missing from test metadata."
            )

        subgroup_values = (
            meta_test[subgroup_col]
            .dropna()
            .astype("string")
            .drop_duplicates()
            .sort_values()
            .tolist()
        )
        subgroup_rows: list[dict[str, Any]] = []

        for subgroup_value in subgroup_values:
            mask = (
                meta_test[subgroup_col]
                .astype("string")
                .eq(subgroup_value)
                .to_numpy()
            )
            positions = np.flatnonzero(mask)
            subgroup_target = y_test.iloc[positions]

            if subgroup_target.nunique() != 2:
                raise RuntimeError(
                    f"Evaluation subgroup {subgroup_col}="
                    f"{subgroup_value} does not contain both classes."
                )

            subgroup_metrics = self._evaluate_model(
                model=model,
                model_name=model_name,
                scenario_name=scenario_name,
                feature_cols=feature_cols,
                X_test=X_test.iloc[positions],
                y_test=subgroup_target,
                meta_test=meta_test.iloc[positions],
                n_train_samples=n_train_samples,
                best_cv_score=best_cv_score,
                best_params=best_params,
                training_time_seconds=training_time_seconds,
                evaluation_mode=evaluation_mode,
                save_predictions=False,
            )
            subgroup_metrics["evaluation_scope"] = (
                f"{subgroup_col}:{subgroup_value}"
            )

            if subgroup_col == "base":
                subgroup_metrics["test_database"] = str(
                    subgroup_value
                )

            metrics.append(subgroup_metrics)
            subgroup_rows.append(subgroup_metrics)

        if len(subgroup_rows) > 1:
            macro_metrics = overall.copy()

            for metric_name in self.SUBGROUP_MACRO_METRICS:
                values = [
                    row.get(metric_name)
                    for row in subgroup_rows
                    if pd.notna(row.get(metric_name))
                ]
                macro_metrics[metric_name] = (
                    float(np.mean(values))
                    if values
                    else np.nan
                )

            for metric_name in list(macro_metrics):
                if (
                    metric_name.endswith("_ci_lower")
                    or metric_name.endswith("_ci_upper")
                    or metric_name
                    in {
                        "tn",
                        "fp",
                        "fn",
                        "tp",
                        "bootstrap_valid_iterations",
                    }
                ):
                    macro_metrics[metric_name] = np.nan

            macro_metrics["evaluation_scope"] = (
                f"{subgroup_col}:macro"
            )

            if subgroup_col == "base":
                macro_metrics["test_database"] = "DATABASE_MACRO"

            metrics.append(macro_metrics)

        return metrics

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
        training_time_seconds: float,
        evaluation_mode: str,
        save_predictions: bool = True,
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
        metrics["compute_backend"] = (
            self.config.compute_backend.value
        )
        metrics["training_time_seconds"] = float(
            training_time_seconds
        )
        metrics["evaluation_mode"] = evaluation_mode
        inferred_test_database = self._infer_database_name(
            split_metadata=meta_test
        )

        if evaluation_mode == "within_database_holdout":
            metrics["train_database"] = (
                self.train_dataset_name
                or inferred_test_database
            )
            metrics["test_database"] = (
                self.test_dataset_name
                or inferred_test_database
            )
        else:
            metrics["train_database"] = (
                self.train_dataset_name
                or self._infer_database_name(
                    split_metadata=self.features_df
                )
            )
            metrics["test_database"] = (
                self.test_dataset_name
                or inferred_test_database
            )

        if self.config.save_predictions and save_predictions:
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
        external_test: bool = False,
    ) -> None:
        train_assignments = split.meta_train.copy()
        train_assignments["target"] = split.y_train.to_numpy()
        train_assignments["partition"] = "train"

        test_assignments = split.meta_test.copy()
        test_assignments["target"] = split.y_test.to_numpy()
        test_assignments["partition"] = (
            "external_test"
            if external_test
            else "test"
        )

        assignments = pd.concat(
            [train_assignments, test_assignments],
            ignore_index=True,
        )
        assignments.to_csv(
            self.splits_dir / "holdout_assignments.csv",
            index=False,
        )

    @staticmethod
    def _infer_database_name(
        split_metadata: pd.DataFrame,
    ) -> str | None:
        if "base" not in split_metadata.columns:
            return None

        values = (
            split_metadata["base"]
            .dropna()
            .astype("string")
            .drop_duplicates()
            .tolist()
        )
        return str(values[0]) if len(values) == 1 else None

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

    def _save_training_curves(
        self,
        model: Pipeline,
        scenario_name: str,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        groups_train: pd.Series | None,
        strata_train: pd.Series | None,
    ) -> None:
        classifier = model.named_steps["classifier"]
        history = self._build_grouped_curve_history(
            model=model,
            X_train=X_train,
            y_train=y_train,
            groups_train=groups_train,
            strata_train=strata_train,
            scenario_name=scenario_name,
            model_name=model_name,
        )

        if history is None:
            history = self._extract_training_history(classifier)

        if history is None:
            return

        file_stem = f"{scenario_name}_{model_name}_training_curve"
        history.to_csv(
            self.training_curves_dir / f"{file_stem}.csv",
            index=False,
        )
        TrainingCurveVisualizer.save(
            history=history,
            output_path=(
                self.training_curves_dir / f"{file_stem}.png"
            ),
            title=(
                f"Training curves — {scenario_name} / {model_name}"
            ),
        )

    def _build_grouped_curve_history(
        self,
        model: Pipeline,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        groups_train: pd.Series | None,
        strata_train: pd.Series | None,
        scenario_name: str,
        model_name: str,
    ) -> pd.DataFrame | None:
        classifier = model.named_steps["classifier"]

        if not (
            isinstance(classifier, MLPClassifier)
            or hasattr(classifier, "history")
        ):
            return None

        print(
            "\nBuilding diagnostic training curves with a "
            "speaker-grouped validation split..."
        )
        fit_indices, validation_indices = (
            self._select_curve_validation_indices(
                X_train=X_train,
                y_train=y_train,
                groups_train=groups_train,
                strata_train=strata_train,
            )
        )
        self._save_curve_split_assignments(
            y_train=y_train,
            groups_train=groups_train,
            fit_indices=fit_indices,
            validation_indices=validation_indices,
            scenario_name=scenario_name,
            model_name=model_name,
        )

        preprocessing_pipeline = Pipeline(
            steps=[
                (
                    name,
                    step if isinstance(step, str) else clone(step),
                )
                for name, step in model.steps[:-1]
            ],
        )
        X_fit = preprocessing_pipeline.fit_transform(
            X_train.iloc[fit_indices],
            y_train.iloc[fit_indices],
        )
        X_validation = preprocessing_pipeline.transform(
            X_train.iloc[validation_indices]
        )
        y_fit = y_train.iloc[fit_indices].to_numpy(
            dtype=np.int64
        )
        y_validation = y_train.iloc[
            validation_indices
        ].to_numpy(dtype=np.int64)

        if isinstance(classifier, MLPClassifier):
            return self._build_sklearn_mlp_curve_history(
                classifier=classifier,
                X_fit=X_fit,
                y_fit=y_fit,
                X_validation=X_validation,
                y_validation=y_validation,
            )

        return self._build_skorch_curve_history(
            classifier=classifier,
            X_fit=X_fit,
            y_fit=y_fit,
            X_validation=X_validation,
            y_validation=y_validation,
        )

    def _select_curve_validation_indices(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        groups_train: pd.Series | None,
        strata_train: pd.Series | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if groups_train is None:
            indices = np.arange(len(y_train))
            fit_indices, validation_indices = train_test_split(
                indices,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
                stratify=(
                    strata_train
                    if strata_train is not None
                    else y_train
                ),
            )
            return fit_indices, validation_indices

        return self._select_grouped_holdout_indices(
            X=X_train,
            y=y_train,
            groups=groups_train,
            stratification_target=strata_train,
        )

    def _save_curve_split_assignments(
        self,
        y_train: pd.Series,
        groups_train: pd.Series | None,
        fit_indices: np.ndarray,
        validation_indices: np.ndarray,
        scenario_name: str,
        model_name: str,
    ) -> None:
        assignments = pd.DataFrame({
            "source_index": y_train.index,
            "target": y_train.to_numpy(),
            "group": (
                groups_train.to_numpy()
                if groups_train is not None
                else y_train.index.to_numpy()
            ),
            "partition": "curve_fit",
        })
        assignments.iloc[
            validation_indices,
            assignments.columns.get_loc("partition"),
        ] = "curve_validation"

        assignments.to_csv(
            self.splits_dir
            / (
                f"{scenario_name}_{model_name}_"
                "curve_assignments.csv"
            ),
            index=False,
        )

    @staticmethod
    def _build_sklearn_mlp_curve_history(
        classifier: MLPClassifier,
        X_fit: Any,
        y_fit: np.ndarray,
        X_validation: Any,
        y_validation: np.ndarray,
    ) -> pd.DataFrame:
        curve_classifier = clone(classifier)
        sample_weight = compute_sample_weight(
            class_weight="balanced",
            y=y_fit,
        )
        records: list[dict[str, float | int]] = []

        for epoch in range(1, classifier.max_iter + 1):
            fit_kwargs: dict[str, Any] = {
                "sample_weight": sample_weight,
            }

            if epoch == 1:
                fit_kwargs["classes"] = np.array([0, 1])

            curve_classifier.partial_fit(
                X_fit,
                y_fit,
                **fit_kwargs,
            )
            train_probabilities = (
                curve_classifier.predict_proba(X_fit)
            )
            validation_probabilities = (
                curve_classifier.predict_proba(X_validation)
            )
            train_prediction = train_probabilities.argmax(axis=1)
            validation_prediction = (
                validation_probabilities.argmax(axis=1)
            )
            records.append({
                "epoch": epoch,
                "train_loss": float(
                    log_loss(
                        y_fit,
                        train_probabilities,
                        labels=[0, 1],
                    )
                ),
                "valid_loss": float(
                    log_loss(
                        y_validation,
                        validation_probabilities,
                        labels=[0, 1],
                    )
                ),
                "train_accuracy": float(
                    accuracy_score(y_fit, train_prediction)
                ),
                "valid_accuracy": float(
                    accuracy_score(
                        y_validation,
                        validation_prediction,
                    )
                ),
                "train_balanced_accuracy": float(
                    balanced_accuracy_score(
                        y_fit,
                        train_prediction,
                    )
                ),
                "valid_balanced_accuracy": float(
                    balanced_accuracy_score(
                        y_validation,
                        validation_prediction,
                    )
                ),
            })

        return pd.DataFrame(records)

    @staticmethod
    def _build_skorch_curve_history(
        classifier: Any,
        X_fit: Any,
        y_fit: np.ndarray,
        X_validation: Any,
        y_validation: np.ndarray,
    ) -> pd.DataFrame:
        try:
            from skorch.dataset import Dataset
            from skorch.helper import predefined_split
        except ImportError as exc:
            raise RuntimeError(
                "Skorch is required to produce PyTorch validation "
                "curves."
            ) from exc

        curve_classifier = clone(classifier)
        validation_dataset = Dataset(
            np.asarray(X_validation, dtype=np.float32),
            y_validation,
        )
        curve_classifier.set_params(
            train_split=predefined_split(validation_dataset),
            # The project callback records validation accuracy directly
            # from batch logits. Skorch's default scorer attempts to pass
            # a Dataset object through the estimator's NumPy boundary.
            callbacks__valid_acc=None,
        )
        curve_classifier.fit(
            np.asarray(X_fit, dtype=np.float32),
            y_fit,
        )
        return ModelTrainingRunner._extract_training_history(
            curve_classifier
        )

    @staticmethod
    def _extract_training_history(
        classifier: Any,
    ) -> pd.DataFrame | None:
        if hasattr(classifier, "history"):
            records = classifier.history.to_list()
            history = pd.DataFrame(
                {
                    "epoch": [
                        record["epoch"]
                        for record in records
                    ],
                    "train_loss": [
                        record.get("train_loss")
                        for record in records
                    ],
                    "train_accuracy": [
                        record.get("train_accuracy")
                        for record in records
                    ],
                    "valid_loss": [
                        record.get("valid_loss")
                        for record in records
                    ],
                    "valid_accuracy": [
                        record.get("valid_accuracy")
                        for record in records
                    ],
                    "train_balanced_accuracy": [
                        record.get("train_balanced_accuracy")
                        for record in records
                    ],
                    "valid_balanced_accuracy": [
                        record.get("valid_balanced_accuracy")
                        for record in records
                    ],
                }
            )
            return history.dropna(axis="columns", how="all")

        if hasattr(classifier, "loss_curve_"):
            return pd.DataFrame(
                {
                    "epoch": np.arange(
                        1,
                        len(classifier.loss_curve_) + 1,
                    ),
                    "train_loss": classifier.loss_curve_,
                }
            )

        return None

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

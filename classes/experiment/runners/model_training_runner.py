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
from classes.experiment.training.experimental_protocol_writer import (
    ExperimentalProtocolWriter,
)
from classes.experiment.training.grouped_svm_learning_curve_runner import (
    GroupedSVMLearningCurveRunner,
)
from classes.experiment.training.host_array_converter import (
    HostArrayConverter,
)
from classes.experiment.training.model_selection_policy import (
    ModelSelectionPolicy,
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
    grid: Any
    training_time_seconds: float
    order: int


@dataclass
class RestoredGridSearchResult:
    best_estimator_: Pipeline | None
    best_score_: float
    best_score_std_: float
    best_train_score_: float
    generalization_gap_: float
    numerical_best_index_: int
    numerical_best_score_: float
    selection_threshold_: float
    selection_standard_error_: float
    best_params_: dict[str, Any]


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
        resume: bool = False,
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
        self.resume = resume
        self.source_selection_df = pd.DataFrame()
        self.family_comparison_metrics_df = pd.DataFrame()

        self.models_dir = self.output_dir / "models"
        self.metrics_dir = self.output_dir / "metrics"
        self.predictions_dir = self.output_dir / "predictions"
        self.splits_dir = self.output_dir / "splits"
        self.checkpoints_dir = self.output_dir / "checkpoints"
        self.training_curves_dir = (
            self.output_dir / "figures" / "training_curves"
        )
        self.learning_curves_dir = (
            self.output_dir / "figures" / "learning_curves"
        )

        self._create_output_dirs()
        self.protocol_hash = ExperimentalProtocolWriter(
            output_dir=self.output_dir,
        ).write(
            config=self.config,
            feature_scenarios=self.feature_scenarios,
            model_specs=self.model_specs,
        )

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

        if self.resume:
            completed = self._load_completed_metrics()
            if completed is not None:
                if not self._needs_svm_learning_curve_completion():
                    print(
                        "\nResume checkpoint: final metrics and "
                        "required auxiliary artifacts already exist; "
                        "training is complete."
                    )
                    return completed
                print(
                    "\nResume checkpoint: final metrics exist, but the "
                    "SVM family champion learning curve is missing. "
                    "Restoring training artifacts to complete it."
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
        """Evaluate a primary model and a prespecified family comparison."""
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
                if self.resume:
                    restored = self._restore_source_candidate(
                        scenario=scenario,
                        model_spec=model_spec,
                        feature_cols=feature_cols,
                        X_train=X_train_scenario,
                        y_train=split.y_train,
                        groups_train=groups_train,
                        order=candidate_order,
                    )
                    if restored is not None:
                        print(
                            "\nResume checkpoint: restored "
                            f"{scenario.name} | {model_spec.name}"
                        )
                        candidates.append(restored)
                        candidate_order += 1
                        continue

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
                self._save_candidate_checkpoint(candidate)

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

        selected, global_selection_threshold = (
            self._select_source_candidate(
                eligible_candidates
            )
        )
        if self.resume:
            self._validate_persisted_source_selection(selected)
        self.source_selection_df = self._save_source_selection_results(
            candidates=candidates,
            selected=selected,
            global_selection_threshold=(
                global_selection_threshold
            ),
        )

        print(
            "\nSelected exclusively by training-partition CV:"
            f"\n  scenario: {selected.scenario.name}"
            f"\n  model: {selected.model_spec.name}"
            f"\n  CV {self.config.scoring}: "
            f"{selected.grid.best_score_:.4f}"
            f"\n  global one-SE threshold: "
            f"{global_selection_threshold:.4f}"
        )

        best_model = self._ensure_candidate_estimator(
            candidate=selected,
            X_train=split.X_train[selected.feature_cols],
            y_train=split.y_train,
        )

        if self.config.run_repeated_nested_cv:
            self._run_repeated_nested_cv(
                X_train=split.X_train,
                y_train=split.y_train,
                meta_train=split.meta_train,
                numeric_cols=numeric_cols,
                groups_train=groups_train,
                strata_train=strata_train,
            )

        comparison_candidates = (
            self._select_family_comparison_candidates(
                candidates=eligible_candidates,
            )
        )
        curve_candidates = list(comparison_candidates)
        if selected not in curve_candidates:
            curve_candidates.append(selected)

        for candidate in curve_candidates:
            candidate_model = self._ensure_candidate_estimator(
                candidate=candidate,
                X_train=split.X_train[candidate.feature_cols],
                y_train=split.y_train,
            )
            if (
                self.config.run_grouped_svm_learning_curve
                and self._model_family(
                    candidate.model_spec.name
                ) == "svm"
            ):
                self._save_grouped_svm_learning_curve(
                    model=candidate_model,
                    scenario_name=candidate.scenario.name,
                    model_name=candidate.model_spec.name,
                    X_train=split.X_train[
                        candidate.feature_cols
                    ],
                    y_train=split.y_train,
                    groups_train=groups_train,
                    strata_train=strata_train,
                )
            self._save_training_curves(
                model=candidate_model,
                scenario_name=candidate.scenario.name,
                model_name=candidate.model_spec.name,
                X_train=split.X_train[candidate.feature_cols],
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
        for metric_row in metrics:
            metric_row["best_cv_score_std"] = float(
                selected.grid.best_score_std_
            )
            metric_row["mean_train_cv_score"] = float(
                selected.grid.best_train_score_
            )
            metric_row["train_cv_generalization_gap"] = float(
                selected.grid.generalization_gap_
            )
            metric_row["global_selection_threshold"] = float(
                global_selection_threshold
            )

        if self.config.save_models:
            self._save_model(
                model=best_model,
                scenario_name=selected.scenario.name,
                model_name=selected.model_spec.name,
            )

        self.family_comparison_metrics_df = (
            self._evaluate_family_comparison(
                candidates=comparison_candidates,
                primary=selected,
                primary_metrics=metrics,
                split=split,
                evaluation_mode=evaluation_mode,
                global_selection_threshold=(
                    global_selection_threshold
                ),
            )
        )
        self._print_strict_evaluation_summary(
            primary_metrics=pd.DataFrame(metrics),
            family_comparison_metrics=(
                self.family_comparison_metrics_df
            ),
        )

        return pd.DataFrame(metrics)

    @staticmethod
    def _print_strict_evaluation_summary(
        primary_metrics: pd.DataFrame,
        family_comparison_metrics: pd.DataFrame,
    ) -> None:
        display_columns = [
            "comparison_role",
            "scenario",
            "model",
            "balanced_accuracy",
            "macro_f1",
            "mcc",
            "auc",
        ]

        primary_overall = primary_metrics
        if "evaluation_scope" in primary_overall.columns:
            primary_overall = primary_overall[
                primary_overall["evaluation_scope"].eq("overall")
            ]

        primary_columns = [
            column
            for column in display_columns
            if column in primary_overall.columns
        ]
        print("\nPrimary holdout/external evaluation:")
        print(
            primary_overall[primary_columns].to_string(
                index=False
            )
        )

        if family_comparison_metrics.empty:
            return

        comparison_overall = family_comparison_metrics
        if "evaluation_scope" in comparison_overall.columns:
            comparison_overall = comparison_overall[
                comparison_overall[
                    "evaluation_scope"
                ].eq("overall")
            ]

        comparison_columns = [
            column
            for column in display_columns
            if column in comparison_overall.columns
        ]
        print(
            "\nPrespecified secondary SVM vs MLP comparison "
            "(families selected by training CV):"
        )
        print(
            comparison_overall[comparison_columns].to_string(
                index=False
            )
        )

    @staticmethod
    def _model_family(model_name: str) -> str | None:
        normalized = model_name.lower()

        if normalized.startswith("svm"):
            return "svm"

        if "mlp" in normalized:
            return "mlp"

        return None

    def _select_family_comparison_candidates(
        self,
        candidates: list[SourceSelectionCandidate],
    ) -> list[SourceSelectionCandidate]:
        """Choose one SVM and one MLP using training CV only."""
        selected_by_family: dict[str, SourceSelectionCandidate] = {}

        for family in ("svm", "mlp"):
            family_candidates = [
                candidate
                for candidate in candidates
                if self._model_family(candidate.model_spec.name) == family
            ]
            if family_candidates:
                family_best, _ = self._select_source_candidate(
                    family_candidates
                )
                selected_by_family[family] = family_best

        return [
            selected_by_family[family]
            for family in ("svm", "mlp")
            if family in selected_by_family
        ]

    def _evaluate_family_comparison(
        self,
        candidates: list[SourceSelectionCandidate],
        primary: SourceSelectionCandidate,
        primary_metrics: list[dict[str, Any]],
        split: TrainingDataSplit,
        evaluation_mode: str,
        global_selection_threshold: float,
    ) -> pd.DataFrame:
        if len(candidates) < 2:
            return pd.DataFrame()

        comparison_rows: list[dict[str, Any]] = []

        for candidate in candidates:
            if candidate is primary:
                candidate_metrics = [
                    metric.copy()
                    for metric in primary_metrics
                ]
            else:
                candidate_metrics = self._evaluate_with_subgroups(
                    model=candidate.grid.best_estimator_,
                    model_name=candidate.model_spec.name,
                    scenario_name=candidate.scenario.name,
                    feature_cols=candidate.feature_cols,
                    X_test=split.X_test[candidate.feature_cols],
                    y_test=split.y_test,
                    meta_test=split.meta_test,
                    n_train_samples=len(split.y_train),
                    best_cv_score=candidate.grid.best_score_,
                    best_params=candidate.grid.best_params_,
                    training_time_seconds=(
                        candidate.training_time_seconds
                    ),
                    evaluation_mode=evaluation_mode,
                )

                if self.config.save_models:
                    self._save_model(
                        model=candidate.grid.best_estimator_,
                        scenario_name=candidate.scenario.name,
                        model_name=candidate.model_spec.name,
                    )

            for metric_row in candidate_metrics:
                metric_row["model_family"] = self._model_family(
                    candidate.model_spec.name
                )
                metric_row["comparison_role"] = (
                    "primary_global_and_family_champion"
                    if candidate is primary
                    else "secondary_family_champion"
                )
                metric_row["selected_for_primary_evaluation"] = (
                    candidate is primary
                )
                metric_row["best_cv_score_std"] = float(
                    candidate.grid.best_score_std_
                )
                metric_row["mean_train_cv_score"] = float(
                    candidate.grid.best_train_score_
                )
                metric_row["train_cv_generalization_gap"] = float(
                    candidate.grid.generalization_gap_
                )
                metric_row["global_selection_threshold"] = float(
                    global_selection_threshold
                )

            comparison_rows.extend(candidate_metrics)

        comparison_df = pd.DataFrame(comparison_rows)
        comparison_df.to_csv(
            self.metrics_dir / "family_comparison_metrics.csv",
            index=False,
        )
        comparison_df.to_parquet(
            self.metrics_dir / "family_comparison_metrics.parquet",
            index=False,
        )
        return comparison_df

    def _save_source_selection_results(
        self,
        candidates: list[SourceSelectionCandidate],
        selected: SourceSelectionCandidate,
        global_selection_threshold: float,
    ) -> pd.DataFrame:
        rows = []

        for candidate in candidates:
            rows.append({
                "scenario": candidate.scenario.name,
                "model": candidate.model_spec.name,
                "n_features": len(candidate.feature_cols),
                "selection_metric": self.config.scoring,
                "protocol_version": self.config.protocol_version,
                "protocol_hash": self.protocol_hash,
                "eligible_for_final_reporting": (
                    self.config.eligible_for_final_reporting
                ),
                "best_cv_score": float(
                    candidate.grid.best_score_
                ),
                "best_cv_score_std": float(
                    candidate.grid.best_score_std_
                ),
                "numerical_best_cv_score": float(
                    candidate.grid.numerical_best_score_
                ),
                "selection_threshold_within_model": float(
                    candidate.grid.selection_threshold_
                ),
                "global_selection_threshold": float(
                    global_selection_threshold
                ),
                "mean_train_score": float(
                    candidate.grid.best_train_score_
                ),
                "train_cv_generalization_gap": float(
                    candidate.grid.generalization_gap_
                ),
                "best_params": str(candidate.grid.best_params_),
                "training_time_seconds": float(
                    candidate.training_time_seconds
                ),
                "selected_for_evaluation": (
                    candidate is selected
                ),
                "eligible_under_global_one_se": (
                    float(candidate.grid.best_score_)
                    >= global_selection_threshold
                ),
                "complexity_key": str(
                    ModelSelectionPolicy
                    .source_candidate_complexity_key(
                        n_input_features=len(
                            candidate.feature_cols
                        ),
                        best_params=candidate.grid.best_params_,
                    )
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
        return selection_df

    def _select_source_candidate(
        self,
        candidates: list[SourceSelectionCandidate],
    ) -> tuple[SourceSelectionCandidate, float]:
        if not candidates:
            raise ValueError(
                "At least one source candidate is required."
            )

        numerical_best = max(
            candidates,
            key=lambda candidate: (
                float(candidate.grid.numerical_best_score_),
                -candidate.order,
            ),
        )
        allowed_drop = max(
            float(
                numerical_best.grid.selection_standard_error_
            ),
            self.config.selection_score_tolerance,
        )
        threshold = (
            float(numerical_best.grid.numerical_best_score_)
            - allowed_drop
        )
        eligible = [
            candidate
            for candidate in candidates
            if float(candidate.grid.best_score_) >= threshold
        ]

        if not eligible:
            eligible = [numerical_best]

        selected = min(
            eligible,
            key=lambda candidate: (
                float(candidate.grid.best_score_std_),
                abs(float(candidate.grid.generalization_gap_)),
                ModelSelectionPolicy.source_candidate_complexity_key(
                    n_input_features=len(candidate.feature_cols),
                    best_params=candidate.grid.best_params_,
                ),
                -float(candidate.grid.best_score_),
                candidate.order,
            ),
        )
        return selected, float(threshold)

    def _run_repeated_nested_cv(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        meta_train: pd.DataFrame,
        numeric_cols: list[str],
        groups_train: pd.Series | None,
        strata_train: pd.Series | None,
    ) -> None:
        print(
            "\nRunning repeated nested grouped CV exclusively on the "
            "source training partition..."
        )
        X = X_train.reset_index(drop=True)
        y = y_train.reset_index(drop=True)
        metadata = meta_train.reset_index(drop=True)
        groups = (
            groups_train.reset_index(drop=True)
            if groups_train is not None
            else pd.Series(
                np.arange(len(y)),
                name="synthetic_group",
            )
        )
        strata = (
            strata_train.reset_index(drop=True)
            if strata_train is not None
            else y
        )
        result_rows, assignment_rows = (
            self._load_nested_cv_checkpoint()
            if self.resume
            else ([], [])
        )
        completed_folds = {
            (int(row["repeat"]), int(row["outer_fold"]))
            for row in result_rows
        }

        for repeat in range(self.config.nested_cv_repeats):
            outer_seed = self.config.random_state + repeat
            outer_splits = self._build_inner_cv_splits(
                X_train=X,
                y_train=y,
                groups_train=groups,
                strata_train=strata,
                n_splits=self.config.nested_cv_folds,
                random_state=outer_seed,
                split_name="nested outer CV",
            )

            for outer_fold, (
                outer_fit_indices,
                outer_validation_indices,
            ) in enumerate(outer_splits):
                fold_identity = (repeat, outer_fold)
                if fold_identity in completed_folds:
                    print(
                        "\nResume checkpoint: skipping completed "
                        f"nested CV repeat {repeat + 1}/"
                        f"{self.config.nested_cv_repeats}, outer fold "
                        f"{outer_fold + 1}/"
                        f"{self.config.nested_cv_folds}"
                    )
                    continue

                print(
                    "\nNested CV repeat "
                    f"{repeat + 1}/{self.config.nested_cv_repeats}, "
                    "outer fold "
                    f"{outer_fold + 1}/{self.config.nested_cv_folds}"
                )
                candidates: list[SourceSelectionCandidate] = []
                candidate_order = 0

                for scenario in self.feature_scenarios:
                    feature_cols = self._select_feature_columns(
                        scenario=scenario,
                        numeric_cols=numeric_cols,
                    )

                    if not feature_cols:
                        continue

                    for model_spec in self.model_specs:
                        started_at = perf_counter()
                        grid = self._run_grid_search(
                            model_spec=model_spec,
                            X_train=X.iloc[
                                outer_fit_indices
                            ][feature_cols],
                            y_train=y.iloc[outer_fit_indices],
                            groups_train=groups.iloc[
                                outer_fit_indices
                            ],
                            strata_train=strata.iloc[
                                outer_fit_indices
                            ],
                        )
                        candidates.append(
                            SourceSelectionCandidate(
                                scenario=scenario,
                                model_spec=model_spec,
                                feature_cols=feature_cols,
                                grid=grid,
                                training_time_seconds=(
                                    perf_counter() - started_at
                                ),
                                order=candidate_order,
                            )
                        )
                        candidate_order += 1

                eligible_candidates = [
                    candidate
                    for candidate in candidates
                    if np.isfinite(
                        float(candidate.grid.best_score_)
                    )
                ]

                if not eligible_candidates:
                    raise RuntimeError(
                        "Nested CV could not select a finite source "
                        "candidate."
                    )

                selected, selection_threshold = (
                    self._select_source_candidate(
                        eligible_candidates
                    )
                )
                model = selected.grid.best_estimator_
                X_outer_fit = X.iloc[
                    outer_fit_indices
                ][selected.feature_cols]
                X_outer_validation = X.iloc[
                    outer_validation_indices
                ][selected.feature_cols]
                train_prediction = model.predict(X_outer_fit)
                validation_prediction = model.predict(
                    X_outer_validation
                )
                validation_score = self._get_model_score(
                    model,
                    X_outer_validation,
                )
                metrics = (
                    ClassificationMetrics.compute_binary_metrics(
                        y_true=y.iloc[
                            outer_validation_indices
                        ].to_numpy(),
                        y_pred=validation_prediction,
                        y_score=validation_score,
                    )
                )
                train_balanced_accuracy = (
                    balanced_accuracy_score(
                        y.iloc[outer_fit_indices],
                        train_prediction,
                    )
                )
                metrics.update({
                    "protocol_version": self.config.protocol_version,
                    "protocol_hash": self.protocol_hash,
                    "eligible_for_final_reporting": (
                        self.config.eligible_for_final_reporting
                    ),
                    "repeat": repeat,
                    "outer_fold": outer_fold,
                    "outer_seed": outer_seed,
                    "scenario": selected.scenario.name,
                    "model": selected.model_spec.name,
                    "n_input_features": len(
                        selected.feature_cols
                    ),
                    "n_outer_train_samples": len(
                        outer_fit_indices
                    ),
                    "n_outer_validation_samples": len(
                        outer_validation_indices
                    ),
                    "n_outer_train_groups": groups.iloc[
                        outer_fit_indices
                    ].nunique(),
                    "n_outer_validation_groups": groups.iloc[
                        outer_validation_indices
                    ].nunique(),
                    "inner_cv_score": float(
                        selected.grid.best_score_
                    ),
                    "inner_cv_score_std": float(
                        selected.grid.best_score_std_
                    ),
                    "inner_train_score": float(
                        selected.grid.best_train_score_
                    ),
                    "inner_train_cv_gap": float(
                        selected.grid.generalization_gap_
                    ),
                    "outer_train_balanced_accuracy": float(
                        train_balanced_accuracy
                    ),
                    "outer_generalization_gap": float(
                        train_balanced_accuracy
                        - metrics["balanced_accuracy"]
                    ),
                    "selection_threshold": float(
                        selection_threshold
                    ),
                    "best_params": str(
                        selected.grid.best_params_
                    ),
                })
                result_rows.append(metrics)

                for partition, indices in (
                    ("outer_train", outer_fit_indices),
                    (
                        "outer_validation",
                        outer_validation_indices,
                    ),
                ):
                    for index in indices:
                        assignment_rows.append({
                            "repeat": repeat,
                            "outer_fold": outer_fold,
                            "source_index": int(index),
                            "sample_id": (
                                metadata.iloc[index].get(
                                    "sample_id",
                                    index,
                                )
                            ),
                            "group": groups.iloc[index],
                            "target": int(y.iloc[index]),
                            "stratum": strata.iloc[index],
                            "partition": partition,
                        })

                pd.DataFrame(result_rows).to_csv(
                    self.metrics_dir
                    / "repeated_nested_cv_results.csv",
                    index=False,
                )
                pd.DataFrame(assignment_rows).to_csv(
                    self.splits_dir
                    / "repeated_nested_cv_assignments.csv",
                    index=False,
                )
                completed_folds.add(fold_identity)

        results = pd.DataFrame(result_rows)
        results.to_csv(
            self.metrics_dir / "repeated_nested_cv_results.csv",
            index=False,
        )
        results.to_parquet(
            self.metrics_dir / "repeated_nested_cv_results.parquet",
            index=False,
        )
        pd.DataFrame(assignment_rows).to_csv(
            self.splits_dir
            / "repeated_nested_cv_assignments.csv",
            index=False,
        )

        summary_metrics = (
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "mcc",
            "auc",
            "pr_auc",
            "outer_train_balanced_accuracy",
            "outer_generalization_gap",
        )
        summary_rows = []

        for metric_name in summary_metrics:
            values = pd.to_numeric(
                results[metric_name],
                errors="coerce",
            ).dropna()
            summary_rows.append({
                "metric": metric_name,
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)),
                "minimum": float(values.min()),
                "maximum": float(values.max()),
                "n_outer_folds": len(values),
            })

        pd.DataFrame(summary_rows).to_csv(
            self.metrics_dir / "repeated_nested_cv_summary.csv",
            index=False,
        )
        stability = (
            results.groupby(
                ["scenario", "model", "best_params"],
                dropna=False,
            )
            .size()
            .rename("selection_count")
            .reset_index()
            .sort_values(
                "selection_count",
                ascending=False,
            )
        )
        stability["selection_frequency"] = (
            stability["selection_count"] / len(results)
        )
        stability.to_csv(
            self.metrics_dir
            / "repeated_nested_cv_selection_stability.csv",
            index=False,
        )

    def _load_nested_cv_checkpoint(
        self,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        results_path = (
            self.metrics_dir / "repeated_nested_cv_results.csv"
        )
        assignments_path = (
            self.splits_dir
            / "repeated_nested_cv_assignments.csv"
        )

        if not results_path.exists() and not assignments_path.exists():
            return [], []
        if not results_path.is_file() or not assignments_path.is_file():
            raise ValueError(
                "Nested-CV resume requires both results and split "
                "assignment checkpoints."
            )

        results = pd.read_csv(results_path)
        assignments = pd.read_csv(assignments_path)
        required_results = {
            "repeat",
            "outer_fold",
            "protocol_hash",
        }
        required_assignments = {
            "repeat",
            "outer_fold",
            "partition",
            "group",
        }
        missing_results = required_results - set(results.columns)
        missing_assignments = (
            required_assignments - set(assignments.columns)
        )
        if missing_results or missing_assignments:
            raise ValueError(
                "Nested-CV checkpoint schema is incomplete: "
                f"results={sorted(missing_results)}, "
                f"assignments={sorted(missing_assignments)}."
            )
        if set(results["protocol_hash"]) != {self.protocol_hash}:
            raise ValueError(
                "Nested-CV checkpoint protocol hash does not match "
                "the active protocol."
            )
        if results.duplicated(["repeat", "outer_fold"]).any():
            raise ValueError(
                "Nested-CV checkpoint contains duplicated folds."
            )

        result_folds = {
            (int(row.repeat), int(row.outer_fold))
            for row in results.itertuples()
        }
        assignment_folds = {
            (int(row.repeat), int(row.outer_fold))
            for row in assignments[
                ["repeat", "outer_fold"]
            ].drop_duplicates().itertuples(index=False)
        }
        if result_folds != assignment_folds:
            raise ValueError(
                "Nested-CV result and assignment checkpoints disagree "
                "on completed folds."
            )

        valid_folds = {
            (repeat, fold)
            for repeat in range(self.config.nested_cv_repeats)
            for fold in range(self.config.nested_cv_folds)
        }
        if not result_folds.issubset(valid_folds):
            raise ValueError(
                "Nested-CV checkpoint contains folds outside the "
                "configured repeat/fold range."
            )

        for fold_identity, fold_rows in assignments.groupby(
            ["repeat", "outer_fold"]
        ):
            partitions = set(fold_rows["partition"])
            if partitions != {"outer_train", "outer_validation"}:
                raise ValueError(
                    "Nested-CV assignment checkpoint has incomplete "
                    f"partitions for fold {fold_identity}."
                )
            train_groups = set(
                fold_rows.loc[
                    fold_rows["partition"].eq("outer_train"),
                    "group",
                ]
            )
            validation_groups = set(
                fold_rows.loc[
                    fold_rows["partition"].eq(
                        "outer_validation"
                    ),
                    "group",
                ]
            )
            if train_groups & validation_groups:
                raise ValueError(
                    "Nested-CV checkpoint contains group leakage in "
                    f"fold {fold_identity}."
                )

        print(
            "\nResume checkpoint: restored "
            f"{len(result_folds)}/"
            f"{self.config.nested_cv_repeats * self.config.nested_cv_folds} "
            "nested-CV folds."
        )
        return (
            results.to_dict(orient="records"),
            assignments.to_dict(orient="records"),
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
        self.learning_curves_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        self.checkpoints_dir.mkdir(
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

        selection_result = None

        def select_parsimonious_candidate(
            cv_results: dict[str, Any],
        ) -> int:
            nonlocal selection_result
            selection_result = (
                ModelSelectionPolicy.select_grid_candidate(
                    mean_scores=cv_results["mean_test_score"],
                    std_scores=cv_results["std_test_score"],
                    params=cv_results["params"],
                    model_name=model_spec.name,
                    cv_folds=len(cv_splits),
                    minimum_score_tolerance=(
                        self.config.selection_score_tolerance
                    ),
                )
            )
            return selection_result.selected_index

        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=model_spec.param_grid,
            scoring=self.config.scoring,
            cv=cv_splits,
            n_jobs=self.config.n_jobs,
            refit=select_parsimonious_candidate,
            return_train_score=True,
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

        if selection_result is None:
            raise RuntimeError(
                "Grid search did not execute the parsimonious refit "
                "policy."
            )

        selected_index = int(grid.best_index_)
        grid.best_score_ = float(
            grid.cv_results_["mean_test_score"][selected_index]
        )
        grid.best_score_std_ = float(
            grid.cv_results_["std_test_score"][selected_index]
        )
        grid.best_train_score_ = float(
            grid.cv_results_["mean_train_score"][selected_index]
        )
        grid.generalization_gap_ = float(
            grid.best_train_score_ - grid.best_score_
        )
        grid.numerical_best_index_ = int(
            selection_result.numerical_best_index
        )
        grid.numerical_best_score_ = float(
            selection_result.numerical_best_score
        )
        grid.selection_threshold_ = float(
            selection_result.selection_threshold
        )
        grid.selection_standard_error_ = float(
            selection_result.standard_error
        )

        print(
            "\nNumerical best CV score: "
            f"{grid.numerical_best_score_:.4f}"
            "\nParsimonious CV score: "
            f"{grid.best_score_:.4f}"
            "\nSelection threshold: "
            f"{grid.selection_threshold_:.4f}"
            "\nMean training score: "
            f"{grid.best_train_score_:.4f}"
            "\nTraining-CV gap: "
            f"{grid.generalization_gap_:.4f}"
        )
        print(f"Selected parameters: {grid.best_params_}")

        return grid

    def _build_inner_cv_splits(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        groups_train: pd.Series | None,
        strata_train: pd.Series | None = None,
        n_splits: int | None = None,
        random_state: int | None = None,
        split_name: str = "inner CV",
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        effective_n_splits = (
            self.config.cv_folds
            if n_splits is None
            else n_splits
        )
        effective_random_state = (
            self.config.random_state
            if random_state is None
            else random_state
        )
        split_target = (
            strata_train
            if strata_train is not None
            else y_train
        )

        if groups_train is None:
            splitter = StratifiedKFold(
                n_splits=effective_n_splits,
                shuffle=True,
                random_state=effective_random_state,
            )
            splits = list(splitter.split(X_train, split_target))
        else:
            if groups_train.nunique() < effective_n_splits:
                raise ValueError(
                    f"Not enough training groups for {split_name}: "
                    f"validation: groups={groups_train.nunique()}, "
                    f"folds={effective_n_splits}."
                )

            splitter = StratifiedGroupKFold(
                n_splits=effective_n_splits,
                shuffle=True,
                random_state=effective_random_state,
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
                    f"{split_name} fold {fold_index} does not contain "
                    "both "
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
                        f"{split_name} fold {fold_index} does not contain "
                        "all configured database/class strata."
                    )

            if groups_train is not None:
                fit_groups = set(groups_train.iloc[fit_indices])
                validation_groups = set(
                    groups_train.iloc[validation_indices]
                )

                if fit_groups & validation_groups:
                    raise RuntimeError(
                        f"Speaker leakage found in {split_name} fold "
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
        metrics["protocol_version"] = self.config.protocol_version
        metrics["protocol_hash"] = self.protocol_hash
        metrics["eligible_for_final_reporting"] = (
            self.config.eligible_for_final_reporting
        )
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

    def _candidate_checkpoint_path(
        self,
        scenario_name: str,
        model_name: str,
    ) -> Path:
        return (
            self.checkpoints_dir
            / f"{scenario_name}__{model_name}.joblib"
        )

    @staticmethod
    def _snapshot_grid_result(grid: Any) -> RestoredGridSearchResult:
        return RestoredGridSearchResult(
            best_estimator_=grid.best_estimator_,
            best_score_=float(grid.best_score_),
            best_score_std_=float(grid.best_score_std_),
            best_train_score_=float(grid.best_train_score_),
            generalization_gap_=float(grid.generalization_gap_),
            numerical_best_index_=int(
                grid.numerical_best_index_
            ),
            numerical_best_score_=float(
                grid.numerical_best_score_
            ),
            selection_threshold_=float(
                grid.selection_threshold_
            ),
            selection_standard_error_=float(
                grid.selection_standard_error_
            ),
            best_params_=dict(grid.best_params_),
        )

    def _save_candidate_checkpoint(
        self,
        candidate: SourceSelectionCandidate,
    ) -> None:
        path = self._candidate_checkpoint_path(
            scenario_name=candidate.scenario.name,
            model_name=candidate.model_spec.name,
        )
        payload = {
            "protocol_hash": self.protocol_hash,
            "scenario": candidate.scenario.name,
            "model": candidate.model_spec.name,
            "training_time_seconds": float(
                candidate.training_time_seconds
            ),
            "order": int(candidate.order),
            "grid": self._snapshot_grid_result(candidate.grid),
        }
        joblib.dump(payload, path)

    def _restore_source_candidate(
        self,
        scenario: FeatureScenario,
        model_spec: ModelSpec,
        feature_cols: list[str],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        groups_train: pd.Series | None,
        order: int,
    ) -> SourceSelectionCandidate | None:
        del X_train, y_train, groups_train
        checkpoint_path = self._candidate_checkpoint_path(
            scenario_name=scenario.name,
            model_name=model_spec.name,
        )

        if checkpoint_path.is_file():
            payload = joblib.load(checkpoint_path)
            if payload.get("protocol_hash") != self.protocol_hash:
                raise ValueError(
                    "Candidate checkpoint protocol hash does not match "
                    f"the active protocol: {checkpoint_path}"
                )
            if (
                payload.get("scenario") != scenario.name
                or payload.get("model") != model_spec.name
            ):
                raise ValueError(
                    f"Candidate checkpoint identity mismatch: "
                    f"{checkpoint_path}"
                )

            return SourceSelectionCandidate(
                scenario=scenario,
                model_spec=model_spec,
                feature_cols=feature_cols,
                grid=payload["grid"],
                training_time_seconds=float(
                    payload["training_time_seconds"]
                ),
                order=order,
            )

        cv_path = (
            self.metrics_dir
            / f"{scenario.name}_{model_spec.name}_cv_results.csv"
        )
        if not cv_path.is_file():
            return None

        cv_results = pd.read_csv(cv_path)
        required = {
            "params",
            "mean_test_score",
            "std_test_score",
            "mean_train_score",
        }
        missing = required - set(cv_results.columns)
        if missing:
            raise ValueError(
                f"Resume CV checkpoint {cv_path} is missing columns: "
                f"{sorted(missing)}"
            )

        params = [
            self._parse_persisted_params(
                value=value,
                model_spec=model_spec,
            )
            for value in cv_results["params"]
        ]
        selection = ModelSelectionPolicy.select_grid_candidate(
            mean_scores=cv_results["mean_test_score"],
            std_scores=cv_results["std_test_score"],
            params=params,
            model_name=model_spec.name,
            cv_folds=self.config.cv_folds,
            minimum_score_tolerance=(
                self.config.selection_score_tolerance
            ),
        )
        selected_index = selection.selected_index
        best_score = float(
            cv_results.iloc[selected_index]["mean_test_score"]
        )
        best_train_score = float(
            cv_results.iloc[selected_index]["mean_train_score"]
        )
        training_time = float(
            (
                pd.to_numeric(
                    cv_results.get("mean_fit_time"),
                    errors="coerce",
                ).fillna(0.0).sum()
                * self.config.cv_folds
            )
        )

        selection_path = (
            self.metrics_dir / "source_model_selection.csv"
        )
        if selection_path.is_file():
            source_selection = pd.read_csv(selection_path)
            matching = source_selection[
                source_selection["scenario"].eq(scenario.name)
                & source_selection["model"].eq(model_spec.name)
            ]
            if len(matching) != 1:
                raise ValueError(
                    "Persisted source selection must contain exactly "
                    f"one row for {scenario.name}/{model_spec.name}."
                )
            row = matching.iloc[0]
            if row["protocol_hash"] != self.protocol_hash:
                raise ValueError(
                    "Persisted source selection protocol hash does not "
                    "match the active protocol."
                )
            if not np.isclose(
                float(row["best_cv_score"]),
                best_score,
            ):
                raise ValueError(
                    "Persisted source-selection score disagrees with "
                    f"the CV checkpoint for {scenario.name}/"
                    f"{model_spec.name}."
                )
            training_time = float(row["training_time_seconds"])

        grid = RestoredGridSearchResult(
            best_estimator_=None,
            best_score_=best_score,
            best_score_std_=float(
                cv_results.iloc[selected_index]["std_test_score"]
            ),
            best_train_score_=best_train_score,
            generalization_gap_=(
                best_train_score - best_score
            ),
            numerical_best_index_=(
                selection.numerical_best_index
            ),
            numerical_best_score_=(
                selection.numerical_best_score
            ),
            selection_threshold_=selection.selection_threshold,
            selection_standard_error_=selection.standard_error,
            best_params_=params[selected_index],
        )
        return SourceSelectionCandidate(
            scenario=scenario,
            model_spec=model_spec,
            feature_cols=feature_cols,
            grid=grid,
            training_time_seconds=training_time,
            order=order,
        )

    @staticmethod
    def _parse_persisted_params(
        value: Any,
        model_spec: ModelSpec,
    ) -> dict[str, Any]:
        if not isinstance(value, str):
            raise TypeError(
                "Persisted parameter payload must be a string."
            )

        matches = [
            dict(params)
            for params in ParameterGrid(model_spec.param_grid)
            if str(dict(params)) == value
        ]
        if len(matches) != 1:
            raise ValueError(
                "Persisted parameter payload does not uniquely match "
                f"the active {model_spec.name} parameter grid: {value}"
            )
        return matches[0]

    def _ensure_candidate_estimator(
        self,
        candidate: SourceSelectionCandidate,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> Pipeline:
        existing = candidate.grid.best_estimator_
        if existing is not None:
            return existing

        print(
            "\nResume checkpoint: refitting selected estimator "
            f"{candidate.scenario.name} | "
            f"{candidate.model_spec.name}"
        )
        model = self._build_pipeline(candidate.model_spec)
        model.set_params(**candidate.grid.best_params_)
        fit_parameters: dict[str, Any] = {}

        if candidate.model_spec.use_balanced_sample_weight:
            fit_parameters["classifier__sample_weight"] = (
                compute_sample_weight(
                    class_weight="balanced",
                    y=y_train,
                )
            )

        model.fit(X_train, y_train, **fit_parameters)
        candidate.grid.best_estimator_ = model
        self._save_candidate_checkpoint(candidate)
        return model

    def _validate_persisted_source_selection(
        self,
        selected: SourceSelectionCandidate,
    ) -> None:
        path = self.metrics_dir / "source_model_selection.csv"
        if not path.is_file():
            return

        persisted = pd.read_csv(path)
        if (
            "protocol_hash" not in persisted
            or set(persisted["protocol_hash"]) != {self.protocol_hash}
        ):
            raise ValueError(
                "Persisted source selection has an incompatible "
                "protocol hash."
            )
        selected_rows = persisted[
            persisted["selected_for_evaluation"].astype(bool)
        ]
        if len(selected_rows) != 1:
            raise ValueError(
                "Persisted source selection must identify one primary "
                "candidate."
            )

        identity = (
            selected_rows.iloc[0]["scenario"],
            selected_rows.iloc[0]["model"],
        )
        restored_identity = (
            selected.scenario.name,
            selected.model_spec.name,
        )
        if identity != restored_identity:
            raise ValueError(
                "Resume would change the primary model selected before "
                f"interruption: persisted={identity}, "
                f"restored={restored_identity}."
            )

    def _load_completed_metrics(self) -> pd.DataFrame | None:
        path = self.metrics_dir / "metrics.csv"
        if not path.is_file():
            return None

        metrics = pd.read_csv(path)
        if (
            "protocol_hash" not in metrics
            or set(metrics["protocol_hash"]) != {self.protocol_hash}
        ):
            raise ValueError(
                "Completed metrics have an incompatible protocol hash."
            )

        selection_path = (
            self.metrics_dir / "source_model_selection.csv"
        )
        if selection_path.is_file():
            self.source_selection_df = pd.read_csv(selection_path)
        comparison_path = (
            self.metrics_dir / "family_comparison_metrics.csv"
        )
        if comparison_path.is_file():
            self.family_comparison_metrics_df = pd.read_csv(
                comparison_path
            )
        return metrics

    def _needs_svm_learning_curve_completion(self) -> bool:
        if not self.config.run_grouped_svm_learning_curve:
            return False

        identities: set[tuple[str, str]] = set()
        comparison = self.family_comparison_metrics_df
        if not comparison.empty:
            model_family = (
                comparison["model_family"].astype("string")
                if "model_family" in comparison
                else comparison["model"].map(self._model_family)
            )
            svm_rows = comparison[model_family.eq("svm")]
            identities.update(
                zip(
                    svm_rows["scenario"].astype(str),
                    svm_rows["model"].astype(str),
                )
            )

        if not identities and not self.source_selection_df.empty:
            selected = self.source_selection_df[
                self.source_selection_df[
                    "selected_for_evaluation"
                ].astype(bool)
            ]
            selected = selected[
                selected["model"].map(self._model_family).eq("svm")
            ]
            identities.update(
                zip(
                    selected["scenario"].astype(str),
                    selected["model"].astype(str),
                )
            )

        for scenario_name, model_name in identities:
            file_stem = (
                f"{scenario_name}_{model_name}_"
                "grouped_learning_curve"
            )
            required_paths = (
                self.learning_curves_dir / f"{file_stem}.csv",
                self.learning_curves_dir
                / f"{file_stem}_summary.csv",
                self.learning_curves_dir / f"{file_stem}.png",
                self.splits_dir
                / f"{file_stem}_assignments.csv",
            )
            if not all(path.is_file() for path in required_paths):
                return True

        return False

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

    def _save_grouped_svm_learning_curve(
        self,
        model: Pipeline,
        scenario_name: str,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        groups_train: pd.Series | None,
        strata_train: pd.Series | None,
    ) -> None:
        print("\nBuilding speaker-grouped SVM learning curve...")
        effective_groups = (
            groups_train.reset_index(drop=True)
            if groups_train is not None
            else pd.Series(
                np.arange(len(y_train)),
                name="synthetic_group",
            )
        )
        y = y_train.reset_index(drop=True)
        X = X_train.reset_index(drop=True)
        strata = (
            strata_train.reset_index(drop=True)
            if strata_train is not None
            else y
        )
        cv_splits = self._build_inner_cv_splits(
            X_train=X,
            y_train=y,
            groups_train=effective_groups,
            strata_train=strata,
        )
        GroupedSVMLearningCurveRunner(
            config=self.config,
            learning_curves_dir=self.learning_curves_dir,
            splits_dir=self.splits_dir,
            resume=self.resume,
            protocol_hash=self.protocol_hash,
        ).run(
            model=model,
            scenario_name=scenario_name,
            model_name=model_name,
            X=X,
            y=y,
            groups=effective_groups,
            strata=strata,
            cv_splits=cv_splits,
        )

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
        file_stem = f"{scenario_name}_{model_name}_training_curve"
        curve_path = (
            self.training_curves_dir / f"{file_stem}.csv"
        )
        if self.resume and curve_path.is_file():
            print(
                "\nResume checkpoint: skipping completed training "
                f"curve {scenario_name} | {model_name}"
            )
            return

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

        history.to_csv(curve_path, index=False)
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

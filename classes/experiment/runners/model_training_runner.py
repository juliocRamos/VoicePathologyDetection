from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV
)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from classes.experiment.training.classification_metrics import ClassificationMetrics
from classes.experiment.training.training_config import TrainingConfig
from classes.experiment.training.training_plan import FeatureScenario, ModelSpec


class ModelTrainingRunner:
    METADATA_COLUMNS = {
        "sample_id",
        "base",
        "label",
        "speaker_id",
        "sex",
        "age",
        "pathology",
        "pathology_code",
        "vowel",
        "pitch",
        "sr",
        "duration",
        "status",
        "error",
    }

    def __init__(
            self,
            features_df: pd.DataFrame,
            output_dir: str | Path,
            config: TrainingConfig,
            feature_scenarios: list[FeatureScenario],
            model_specs: list[ModelSpec]
    ):
        self.features_df = features_df
        self.output_dir = output_dir
        self.config = config
        self.feature_scenarios = feature_scenarios
        self.model_specs = model_specs

        self.models_dir = self.output_dir / "models"
        self.metrics_dir = self.output_dir / "metrics"
        self.predictions_dir = self.output_dir / "predictions"

        self._create_output_dirs()

    def run(self) -> pd.DataFrame:
        df = self._prepare_dataframe()
        numeric_cols = self._get_numeric_feature_columns(df)

        X_train, X_test, y_train, y_test, meta_train, meta_test = self._split_train_test(
            df = df
        )

        all_metrics: list[dict[str, Any]] = []

        for scenario in self.feature_scenarios:
            feature_cols = self._select_feature_columns(
                scenario = scenario,
                numeric_cols=numeric_cols,
            )

            if not feature_cols:
                print(f"[WARNING Scenario without features: {scenario.name}]")
                continue

            X_train_scenario = X_train[feature_cols]
            X_test_scenario = X_test[feature_cols]

            for model_spec in self.model_specs:
                print(
                    f"\nRunning train with scenario: {scenario.name} |"
                    f"model_spec: {model_spec.name} |"
                    f"features={len(feature_cols)}"
                )

                grid = self._run_grid_search(
                    model_spec=model_spec,
                    X_train=X_train_scenario,
                    y_train=y_train,
                )

                best_model = grid.best_estimator_

                metrics = self._evaluate_model(
                    model=best_model,
                    model_name=model_spec.name,
                    scenario_name=scenario.name,
                    feature_cols=feature_cols,
                    X_test=X_test_scenario,
                    y_test=y_test,
                    meta_test=meta_test,
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

    def _prepare_dataframe(self) -> pd.DataFrame:
        df = self.features_df.copy()

        if "status" in df.columns:
            df = df[df["status"] == "ok"].copy()

        df = df.dropna(subset=[self.config.label_col]).copy()

        df["target"] = (
            df[self.config.label_col].astype(str) == self.config.positive_label
        ).astype(int)

        print("\nClass distribution:")
        print(df["target"].value_counts())

        return df

    def _get_numeric_feature_columns(self, df: pd.DataFrame) -> list[str]:
        candidates = [
            col for col in df.columns
            if col not in self.METADATA_COLUMNS and col != "target"
        ]

        numeric_cols = [
            col for col in candidates
            if pd.api.types.is_numeric_dtype(df[col])
        ]

        return numeric_cols

    def _split_train_test(self,
                          df: pd.DataFrame):
        metadata_cols = [
            col for col in self.METADATA_COLUMNS
            if col in df.columns
        ]

        feature_cols = self._get_numeric_feature_columns(df)

        X = df[feature_cols]
        y = df["target"].copy()
        meta = df[metadata_cols].copy()

        return train_test_split(
            X,
            y,
            meta,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y
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

    @staticmethod
    def _build_pipeline(
        model_spec: ModelSpec,
    ) -> Pipeline:
        return Pipeline([
            ("imputer", SimpleImputer()),
            ("scaler", StandardScaler()),
            ("classifier", model_spec.estimator),
        ])

    def _run_grid_search(
            self,
            model_spec: ModelSpec,
            X_train: pd.DataFrame,
            y_train: pd.DataFrame,
    ) -> GridSearchCV:
        pipeline = self._build_pipeline(model_spec)

        cv = StratifiedKFold(
            n_splits=self.config.cv_folds,
            shuffle=True,
            random_state=self.config.random_state,
        )

        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=model_spec.param_grid,
            scoring=self.config.scoring,
            cv=cv,
            n_jobs=self.config.n_jobs,
            refit=True,
            return_train_score=True,
        )

        grid.fit(X_train, y_train)

        print(f"\nBest CV score: {grid.best_score_:.4f}")
        print(f"Best parameters: {grid.best_params_}")

        return grid

    def _evaluate_model(
        self,
        model: Pipeline,
        model_name: str,
        scenario_name: str,
        feature_cols: list[str],
        X_test: pd.DataFrame,
        y_test: pd.Series,
        meta_test: pd.DataFrame,
        best_cv_score: float,
        best_params: dict[str, Any],
    ) -> dict[str, Any]:

        y_pred = model.predict(X_test)
        y_score = self._get_model_score(model, X_test)

        metrics = ClassificationMetrics.compute_binary_metrics(
            y_true=y_test.to_numpy(),
            y_pred=y_pred,
            y_score=y_score
        )

        metrics["scenario"] = scenario_name
        metrics["model"] = model_name
        metrics["n_features"] = len(feature_cols)
        metrics["n_train_test_samples"] = len(y_test)
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
        try:
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(X_test)
                return probabilities[:, 1]
        except Exception:
            pass

        try:
            if hasattr(model, "decision_function"):
                return model.decision_function(X_test)
        except Exception:
            pass

        return None

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

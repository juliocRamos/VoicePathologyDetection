from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import pandas as pd

from classes.experiment.path_manager.experiment_paths import (
    ExperimentPaths,
)
from classes.experiment.runners.experiment_stage import ExperimentStage
from classes.experiment.runners.hupa_experiment_runner import (
    HUPAExperimentRunner,
)
from classes.experiment.runners.model_training_runner import (
    ModelTrainingRunner,
)
from classes.experiment.runners.svd_experiment_runner import (
    SVDExperimentRunner,
)
from classes.experiment.training.training_config import TrainingConfig
from classes.experiment.training.training_plan import TrainingPlan
from classes.plot.training_metrics_visualizer import (
    TrainingMetricsVisualizer,
)


class CrossDatabaseExperimentRunner:
    """Train on one voice database and evaluate on the other."""

    def __init__(
        self,
        hupa_runner: HUPAExperimentRunner,
        svd_runner: SVDExperimentRunner,
        data_root: str | Path,
        experiment_name: str,
        training_config: TrainingConfig,
    ) -> None:
        self.hupa_runner = hupa_runner
        self.svd_runner = svd_runner
        self.data_root = Path(data_root)
        self.experiment_name = experiment_name
        self.training_config = training_config
        self.paths = ExperimentPaths.create(
            data_root=self.data_root,
            dataset_name="CROSS_DATABASE",
            experiment_name=self.experiment_name,
        )

    def run(
        self,
        stage: ExperimentStage = ExperimentStage.PREPARE,
    ) -> pd.DataFrame | None:
        source_stage = (
            ExperimentStage.FEATURES
            if stage.includes_feature_extraction
            else ExperimentStage.PREPARE
        )

        hupa_features = self.hupa_runner.run(stage=source_stage)
        svd_features = self.svd_runner.run(stage=source_stage)
        self._save_config(stage=stage)

        if hupa_features is not None and svd_features is not None:
            self.save_cohort_comparison(
                hupa_features=hupa_features,
                svd_features=svd_features,
                reports_dir=self.paths.root_dir / "reports",
            )

        if not stage.includes_training:
            print(
                "\nCross-database preparation saved in:\n"
                f"{self.paths.root_dir}"
            )
            return None

        if hupa_features is None or svd_features is None:
            raise RuntimeError(
                "Cross-database training requires features from both "
                "databases."
            )

        results = [
            self._run_direction(
                train_features=hupa_features,
                test_features=svd_features,
                train_database="HUPA",
                test_database="SVD",
            ),
            self._run_direction(
                train_features=svd_features,
                test_features=hupa_features,
                train_database="SVD",
                test_database="HUPA",
            ),
        ]
        metrics = pd.concat(results, ignore_index=True)

        metrics_dir = self.paths.root_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        metrics.to_csv(
            metrics_dir / "cross_database_metrics.csv",
            index=False,
        )
        metrics.to_parquet(
            metrics_dir / "cross_database_metrics.parquet",
            index=False,
        )

        print(
            "\nCross-database experiment saved in:\n"
            f"{self.paths.root_dir}"
        )
        return metrics

    @classmethod
    def save_cohort_comparison(
        cls,
        hupa_features: pd.DataFrame,
        svd_features: pd.DataFrame,
        reports_dir: str | Path,
    ) -> None:
        reports_dir = Path(reports_dir)
        reports_dir.mkdir(parents=True, exist_ok=True)

        cohorts = {
            "HUPA": cls._modeling_rows(hupa_features),
            "SVD": cls._modeling_rows(svd_features),
        }

        for database, cohort in cohorts.items():
            cls._validate_adult_cohort(
                cohort=cohort,
                database=database,
            )

        demographic_rows: list[dict[str, object]] = []
        pathology_frames: list[pd.DataFrame] = []

        for database, cohort in cohorts.items():
            labels = [
                "all",
                *sorted(
                    cohort["label"]
                    .dropna()
                    .astype("string")
                    .unique()
                    .tolist()
                ),
            ]

            for label in labels:
                subset = (
                    cohort
                    if label == "all"
                    else cohort[
                        cohort["label"].astype("string").eq(label)
                    ]
                )
                demographic_rows.append(
                    cls._demographic_summary_row(
                        cohort=subset,
                        database=database,
                        label=label,
                    )
                )

            pathology_column = (
                "pathology"
                if "pathology" in cohort.columns
                else "pathology_group"
            )
            pathology = (
                cohort.assign(
                    pathology_report=(
                        cohort[pathology_column]
                        .astype("string")
                        .fillna("missing")
                    )
                )
                .groupby(
                    ["label", "pathology_report"],
                    dropna=False,
                )
                .agg(
                    n_samples=("label", "size"),
                    n_speakers=("speaker_id", "nunique"),
                )
                .reset_index()
            )
            pathology.insert(0, "database", database)
            pathology_frames.append(pathology)

        demographics = pd.DataFrame(demographic_rows)
        demographics.to_csv(
            reports_dir / "cohort_demographics_by_class.csv",
            index=False,
        )
        demographics.to_parquet(
            reports_dir / "cohort_demographics_by_class.parquet",
            index=False,
        )

        pathology_distribution = pd.concat(
            pathology_frames,
            ignore_index=True,
        )
        pathology_distribution.to_csv(
            reports_dir / "cohort_pathology_distribution.csv",
            index=False,
        )
        pathology_distribution.to_parquet(
            reports_dir / "cohort_pathology_distribution.parquet",
            index=False,
        )

    @staticmethod
    def _modeling_rows(features: pd.DataFrame) -> pd.DataFrame:
        cohort = features.copy()

        if "status" in cohort.columns:
            cohort = cohort[
                cohort["status"].astype("string").eq("ok")
            ].copy()

        return cohort.dropna(subset=["label"]).copy()

    @staticmethod
    def _validate_adult_cohort(
        cohort: pd.DataFrame,
        database: str,
    ) -> None:
        if "age" not in cohort.columns:
            raise ValueError(
                f"{database} cross-database cohort has no age column."
            )

        age = pd.to_numeric(cohort["age"], errors="coerce")

        if age.isna().any():
            raise ValueError(
                f"{database} cross-database cohort contains "
                f"{int(age.isna().sum())} samples without valid age."
            )

        underage = age.lt(18.0)

        if underage.any():
            raise ValueError(
                f"{database} cross-database cohort contains "
                f"{int(underage.sum())} samples younger than 18."
            )

    @staticmethod
    def _demographic_summary_row(
        cohort: pd.DataFrame,
        database: str,
        label: str,
    ) -> dict[str, object]:
        age = pd.to_numeric(cohort["age"], errors="coerce")
        sex = (
            cohort["sex"].astype("string").str.strip().str.lower()
            if "sex" in cohort.columns
            else pd.Series(
                pd.NA,
                index=cohort.index,
                dtype="string",
            )
        )

        return {
            "database": database,
            "label": label,
            "n_samples": int(len(cohort)),
            "n_speakers": int(
                cohort["speaker_id"].nunique()
                if "speaker_id" in cohort.columns
                else len(cohort)
            ),
            "age_available": int(age.notna().sum()),
            "age_missing": int(age.isna().sum()),
            "age_mean": float(age.mean()),
            "age_std": float(age.std()),
            "age_median": float(age.median()),
            "age_min": float(age.min()),
            "age_max": float(age.max()),
            "female_samples": int(sex.eq("female").sum()),
            "male_samples": int(sex.eq("male").sum()),
            "sex_missing_or_other": int(
                (~sex.isin(["female", "male"])).sum()
            ),
        }

    def _run_direction(
        self,
        train_features: pd.DataFrame,
        test_features: pd.DataFrame,
        train_database: str,
        test_database: str,
    ) -> pd.DataFrame:
        direction_name = (
            f"train_{train_database.lower()}_test_"
            f"{test_database.lower()}"
        )
        output_dir = self.paths.root_dir / direction_name
        print(
            "\nCross-database direction:"
            f"\n  train: {train_database}"
            f"\n  test: {test_database}"
        )

        training_runner = ModelTrainingRunner(
            features_df=train_features,
            external_test_features_df=test_features,
            train_dataset_name=train_database,
            test_dataset_name=test_database,
            output_dir=output_dir,
            config=self.training_config,
            feature_scenarios=(
                TrainingPlan.default_feature_scenarios()
            ),
            model_specs=TrainingPlan.default_model_specs(
                random_state=self.training_config.random_state,
                compute_backend=self.training_config.compute_backend,
            ),
        )
        metrics = training_runner.run()

        visualizer = TrainingMetricsVisualizer(
            metrics_df=metrics,
            predictions_dir=output_dir / "predictions",
            output_dir=output_dir / "figures",
        )
        visualizer.generate_best_models_report(
            best_metric="balanced_accuracy"
        )
        return metrics

    def _save_config(self, stage: ExperimentStage) -> None:
        config = {
            "dataset_name": "CROSS_DATABASE",
            "experiment_name": self.experiment_name,
            "stage": stage.value,
            "directions": [
                {
                    "train_database": "HUPA",
                    "test_database": "SVD",
                },
                {
                    "train_database": "SVD",
                    "test_database": "HUPA",
                },
            ],
            "hupa_experiment_root": str(
                self.hupa_runner.paths.root_dir
            ),
            "svd_experiment_root": str(
                self.svd_runner.paths.root_dir
            ),
            "training_config": asdict(self.training_config),
            "cohort_note": (
                "Only adults with valid age are included in both "
                "databases. Sustained vowel /a/ at normal condition is "
                "used in SVD so that its vocal task matches HUPA."
            ),
            "selection_note": (
                "Feature scenario, model family, preprocessing, and "
                "hyperparameters are selected exclusively by grouped "
                "cross-validation on the source database. Only the "
                "selected refitted pipeline is evaluated on the "
                "destination database."
            ),
            "domain_shift_note": (
                "Results include clinical generalization and domain "
                "shift caused by database, language, equipment, and "
                "collection protocol."
            ),
        }

        with open(
            self.paths.config_path,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                config,
                file,
                indent=4,
                ensure_ascii=False,
            )

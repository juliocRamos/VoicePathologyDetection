from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path

import pandas as pd

from classes.experiment.path_manager.experiment_paths import (
    ExperimentPaths,
)
from classes.experiment.runners.cross_database_experiment_runner import (
    CrossDatabaseExperimentRunner,
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


class PooledDatabaseExperimentRunner:
    """Train and test on speaker-disjoint mixtures of HUPA and SVD."""

    POOLED_GROUP_COLUMN = "database_speaker_id"

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
        self.training_config = replace(
            training_config,
            group_col=self.POOLED_GROUP_COLUMN,
            strict_model_selection=True,
            stratify_col="base",
            evaluation_subgroup_col="base",
        )
        self.paths = ExperimentPaths.create(
            data_root=self.data_root,
            dataset_name="POOLED",
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

        if hupa_features is None or svd_features is None:
            if stage.includes_feature_extraction:
                raise RuntimeError(
                    "Pooled feature extraction requires both databases."
                )

            print(
                "\nPooled-database preparation saved in:\n"
                f"{self.paths.root_dir}"
            )
            return None

        CrossDatabaseExperimentRunner.save_cohort_comparison(
            hupa_features=hupa_features,
            svd_features=svd_features,
            reports_dir=self.paths.reports_dir,
        )
        pooled_features = self._build_pooled_features(
            hupa_features=hupa_features,
            svd_features=svd_features,
        )
        pooled_features.to_csv(
            self.paths.features_dir / "pooled_features.csv",
            index=False,
        )
        pooled_features.to_parquet(
            self.paths.features_dir / "pooled_features.parquet",
            index=False,
        )

        if not stage.includes_training:
            print(
                "\nPooled-database features saved in:\n"
                f"{self.paths.root_dir}"
            )
            return pooled_features

        training_dir = self.paths.root_dir / "training"
        training_runner = ModelTrainingRunner(
            features_df=pooled_features,
            output_dir=training_dir,
            config=self.training_config,
            feature_scenarios=(
                TrainingPlan.default_feature_scenarios()
            ),
            model_specs=TrainingPlan.default_model_specs(
                random_state=self.training_config.random_state,
                compute_backend=self.training_config.compute_backend,
            ),
            train_dataset_name="HUPA+SVD",
            test_dataset_name="HUPA+SVD",
        )
        metrics = training_runner.run()

        report_metrics = (
            training_runner.family_comparison_metrics_df
        )
        if report_metrics.empty:
            report_metrics = metrics
        overall_metrics = report_metrics[
            report_metrics["evaluation_scope"].eq("overall")
        ].copy()
        visualizer = TrainingMetricsVisualizer(
            metrics_df=overall_metrics,
            predictions_dir=training_dir / "predictions",
            output_dir=training_dir / "figures",
            ranking_df=training_runner.source_selection_df,
        )
        visualizer.generate_best_models_report(
            best_metric="balanced_accuracy"
        )

        print(
            "\nPooled-database experiment saved in:\n"
            f"{self.paths.root_dir}"
        )
        return metrics

    def _build_pooled_features(
        self,
        hupa_features: pd.DataFrame,
        svd_features: pd.DataFrame,
    ) -> pd.DataFrame:
        hupa = CrossDatabaseExperimentRunner._modeling_rows(
            hupa_features
        )
        svd = CrossDatabaseExperimentRunner._modeling_rows(
            svd_features
        )

        common_features = self._common_numeric_features(
            hupa=hupa,
            svd=svd,
        )
        metadata_columns = sorted(
            (
                set(hupa.columns)
                | set(svd.columns)
            )
            & ModelTrainingRunner.METADATA_COLUMNS
        )
        schema = pd.DataFrame([
            {
                "feature": feature,
                "available_in_hupa": feature in hupa.columns,
                "available_in_svd": feature in svd.columns,
                "used": feature in common_features,
            }
            for feature in sorted(
                set(self._numeric_features(hupa))
                | set(self._numeric_features(svd))
            )
        ])
        schema.to_csv(
            self.paths.reports_dir / "pooled_feature_schema.csv",
            index=False,
        )

        aligned_columns = metadata_columns + common_features
        pooled = pd.concat(
            [
                hupa.reindex(columns=aligned_columns),
                svd.reindex(columns=aligned_columns),
            ],
            ignore_index=True,
        )
        missing_speakers = (
            pooled["speaker_id"].isna()
            | pooled["speaker_id"]
            .astype("string")
            .str.strip()
            .eq("")
            .fillna(True)
        )

        if missing_speakers.any():
            raise ValueError(
                "Pooled cohort contains samples without speaker_id."
            )

        pooled[self.POOLED_GROUP_COLUMN] = (
            pooled["base"].astype("string")
            + "::"
            + pooled["speaker_id"].astype("string")
        )
        label_counts = (
            pooled.groupby(self.POOLED_GROUP_COLUMN)["label"]
            .nunique()
        )

        if label_counts.gt(1).any():
            raise ValueError(
                "A pooled speaker group contains conflicting labels."
            )

        return pooled

    @staticmethod
    def _numeric_features(dataframe: pd.DataFrame) -> list[str]:
        return [
            column
            for column in dataframe.columns
            if (
                column not in ModelTrainingRunner.METADATA_COLUMNS
                and column != "target"
                and pd.api.types.is_numeric_dtype(dataframe[column])
            )
        ]

    def _common_numeric_features(
        self,
        hupa: pd.DataFrame,
        svd: pd.DataFrame,
    ) -> list[str]:
        hupa_features = self._numeric_features(hupa)
        svd_features = set(self._numeric_features(svd))
        common = [
            feature
            for feature in hupa_features
            if feature in svd_features
        ]

        if not common:
            raise ValueError(
                "HUPA and SVD do not share numeric acoustic features."
            )

        return common

    def _save_config(self, stage: ExperimentStage) -> None:
        config = {
            "dataset_name": "POOLED",
            "experiment_name": self.experiment_name,
            "stage": stage.value,
            "hupa_experiment_root": str(
                self.hupa_runner.paths.root_dir
            ),
            "svd_experiment_root": str(
                self.svd_runner.paths.root_dir
            ),
            "training_config": asdict(self.training_config),
            "protocol": (
                "Adult HUPA and adult SVD /a/ normal samples are "
                "pooled. Holdout and CV are grouped by "
                "database::speaker_id and stratified by database/class. "
                "One pipeline is selected by training CV. Test metrics "
                "are reported overall and separately for HUPA and SVD."
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

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
import json

import pandas as pd

from classes.audio_sample.audio_loader.preprocessing.audio_preprocessor import (
    AudioPreprocessor,
)
from classes.audio_sample.audio_loader.preprocessing.audio_preprocess_config import (
    AudioPreprocessConfig,
)
from classes.audio_sample.audio_loader.profilers.ProcessedAudioProfiler import (
    ProcessedAudioProfiler,
)
from classes.dataset.adapters.svd_adapter import SVDAdapter
from classes.dataset.preparation.svd_training_manifest_builder import (
    SVDTrainingManifestBuilder,
    SVDTrainingManifestConfig,
)
from classes.dataset.preparation.training_manifest import (
    TrainingManifestResult,
)
from classes.dataset.preparation.training_manifest_writer import (
    TrainingManifestWriter,
)
from classes.experiment.path_manager.experiment_paths import ExperimentPaths
from classes.experiment.runners.experiment_stage import ExperimentStage
from classes.experiment.runners.model_training_runner import (
    ModelTrainingRunner,
)
from classes.experiment.training.training_config import TrainingConfig
from classes.experiment.training.training_plan import TrainingPlan
from classes.plot.dataset_visualizer import DatasetVisualizer
from classes.plot.training_metrics_visualizer import (
    TrainingMetricsVisualizer,
)
from classes.vpd.feature_extraction_runner import FeatureExtractionRunner
from classes.vpd.vpd_feature_extractor import VPDFeatureExtractor


class SVDExperimentRunner:
    def __init__(
        self,
        dataset_root: str | Path,
        data_root: str | Path,
        experiment_name: str,
        preprocess_config: AudioPreprocessConfig,
        feature_config=None,
        manifest_config: SVDTrainingManifestConfig | None = None,
        training_config: TrainingConfig | None = None,
    ):
        self.dataset_root = Path(dataset_root)
        self.data_root = Path(data_root)
        self.experiment_name = experiment_name
        self.preprocess_config = preprocess_config
        self.feature_config = feature_config
        self.manifest_config = (
            manifest_config
            or SVDTrainingManifestConfig()
        )
        self.training_config = (
            training_config
            or TrainingConfig(group_col="speaker_id")
        )

        self.paths = ExperimentPaths.create(
            data_root=self.data_root,
            dataset_name="SVD",
            experiment_name=self.experiment_name,
        )

    def run(
        self,
        stage: ExperimentStage = ExperimentStage.PREPARE,
    ) -> pd.DataFrame | None:
        self.save_config(stage=stage)

        raw_manifest = self.build_manifest()
        preparation = self.prepare_training_manifest(raw_manifest)
        training_manifest = preparation.training_manifest

        self.profile_preprocessing(training_manifest)
        self.generate_plots(training_manifest)

        features_df: pd.DataFrame | None = None
        if stage.includes_feature_extraction:
            if self.feature_config is None:
                raise ValueError(
                    "feature_config is required for feature extraction."
                )

            features_df = self.extract_features(training_manifest)

        if stage.includes_training:
            if features_df is None:
                raise RuntimeError(
                    "Training requires the feature-extraction stage."
                )
            self.train_models(features_df)

        self.write_summary(
            raw_manifest=raw_manifest,
            preparation=preparation,
        )

        print(f"\nSVD experiment saved in:\n{self.paths.root_dir}")
        return features_df

    def save_config(self, stage: ExperimentStage) -> None:
        config_data = {
            "dataset_name": "SVD",
            "dataset_root": str(self.dataset_root),
            "experiment_name": self.experiment_name,
            "stage": stage.value,
            "preprocess_config": self._to_dict(self.preprocess_config),
            "feature_config": self._to_dict(self.feature_config),
            "manifest_config": self._to_dict(self.manifest_config),
            "training_config": self._to_dict(self.training_config),
        }

        with open(self.paths.config_path, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=4, ensure_ascii=False)

    def build_manifest(self) -> pd.DataFrame:
        print("[1/6] Building SVD raw manifest...")

        adapter = SVDAdapter(
            root_dir=self.dataset_root
        )

        manifest = adapter.build_manifest()
        adapter.validate_manifest(manifest)

        manifest_path = (
            self.paths.manifests_dir
            / "svd_raw_manifest.parquet"
        )
        manifest_csv_path = (
            self.paths.manifests_dir
            / "svd_raw_manifest.csv"
        )

        manifest.to_parquet(manifest_path, index=False)
        manifest.to_csv(manifest_csv_path, index=False)

        print("Manifest saved in:\n", manifest_path)
        return manifest

    def prepare_training_manifest(
        self,
        raw_manifest: pd.DataFrame,
    ) -> TrainingManifestResult:
        print("\n[2/6] Preparing SVD training manifest...")

        builder = SVDTrainingManifestBuilder(
            config=self.manifest_config
        )
        result = builder.build(raw_manifest)

        TrainingManifestWriter(
            manifests_dir=self.paths.manifests_dir,
            reports_dir=self.paths.reports_dir,
            dataset_slug="svd",
        ).write(result)

        print(
            "SVD training manifest rows:",
            len(result.training_manifest),
        )
        return result

    def profile_preprocessing(
        self,
        manifest: pd.DataFrame,
    ) -> pd.DataFrame:
        print("\n[3/6] Evaluating SVD preprocessing...")

        preprocessor = AudioPreprocessor(
            config=self.preprocess_config,
        )

        profiler = ProcessedAudioProfiler(preprocessor)

        profile = profiler.profile_manifest(manifest)

        profile_path = self.paths.profiles_dir / "svd_processed_audio_profile.parquet"
        profile_csv_path = self.paths.profiles_dir / "svd_processed_audio_profile.csv"

        profile.to_parquet(profile_path, index=False)
        profile.to_csv(profile_csv_path, index=False)

        print("\nPreprocessing status:")
        print(profile["status"].value_counts(dropna=False))

        if "processed_duration" in profile.columns:
            print("\nProcessed duration:")
            print(profile["processed_duration"].describe())

        if "processed_rms" in profile.columns:
            print("\nProcessed RMS:")
            print(profile["processed_rms"].describe())

        if "processed_peak" in profile.columns:
            print("\nProcessed peak:")
            print(profile["processed_peak"].describe())

        errors = profile[profile["status"] == "error"]

        if not errors.empty:
            print("\nErrors found in preprocessing:")
            print(errors[["sample_id", "filepath", "error"]].head(30))

        print(f"Profile saved in: \n{profile_path}")

        return profile

    def profile_peprocessing(
        self,
        manifest: pd.DataFrame,
    ) -> pd.DataFrame:
        """Backward-compatible alias for the previous misspelling."""
        return self.profile_preprocessing(manifest)

    def generate_plots(self, manifest: pd.DataFrame) -> None:
        print("\n[4/6] Generating SVD plots...")

        plot_dir = self.paths.figures_dir / "svd_training_manifest"

        visualizer = DatasetVisualizer(
            manifest=manifest,
            output_dir=plot_dir
        )

        visualizer.generate_svd_report()
        print("\nGenerated plots saved in:\n", plot_dir)

    def extract_features(
        self,
        manifest: pd.DataFrame,
    ) -> pd.DataFrame:
        print("\n[5/6] Extracting SVD features...")

        preprocessor = AudioPreprocessor(
            config=self.preprocess_config
        )
        feature_extractor = VPDFeatureExtractor(
            config=self.feature_config
        )
        runner = FeatureExtractionRunner(
            preprocessor=preprocessor,
            extractor=feature_extractor,
        )
        features_df = runner.extract_from_manifest(manifest)

        features_path = (
            self.paths.features_dir
            / "svd_features_v1.parquet"
        )
        features_csv_path = (
            self.paths.features_dir
            / "svd_features_v1.csv"
        )
        features_df.to_parquet(features_path, index=False)
        features_df.to_csv(features_csv_path, index=False)

        print("\nFeature extraction status:")
        print(features_df["status"].value_counts(dropna=False))
        print("\nFeatures saved in:\n", features_path)
        return features_df

    def train_models(
        self,
        features_df: pd.DataFrame,
    ) -> pd.DataFrame:
        print("\n[6/6] Training SVD models...")

        feature_scenarios = TrainingPlan.default_feature_scenarios()
        model_specs = TrainingPlan.default_model_specs(
            random_state=self.training_config.random_state,
            compute_backend=self.training_config.compute_backend,
        )
        training_runner = ModelTrainingRunner(
            features_df=features_df,
            output_dir=self.paths.root_dir / "training",
            config=self.training_config,
            feature_scenarios=feature_scenarios,
            model_specs=model_specs,
        )
        metrics_df = training_runner.run()

        print("\nTop 5 candidates by training CV balanced accuracy:")
        if not training_runner.source_selection_df.empty:
            ranking_columns = [
                column
                for column in (
                    "source_cv_rank",
                    "scenario",
                    "model",
                    "best_cv_score",
                    "best_cv_score_std",
                )
                if column
                in training_runner.source_selection_df.columns
            ]
            print(
                training_runner.source_selection_df[
                    ranking_columns
                ].head(5).to_string(index=False)
            )
        else:
            print(metrics_df.head(5))

        report_metrics = (
            training_runner.family_comparison_metrics_df
        )
        if report_metrics.empty:
            report_metrics = metrics_df
        elif "evaluation_scope" in report_metrics.columns:
            report_metrics = report_metrics[
                report_metrics["evaluation_scope"].eq("overall")
            ].copy()

        visualizer = TrainingMetricsVisualizer(
            metrics_df=report_metrics,
            predictions_dir=(
                self.paths.root_dir
                / "training"
                / "predictions"
            ),
            output_dir=(
                self.paths.root_dir
                / "training"
                / "figures"
            ),
            ranking_df=training_runner.source_selection_df,
        )
        visualizer.generate_best_models_report(
            best_metric="balanced_accuracy"
        )
        return metrics_df

    def write_summary(
        self,
        raw_manifest: pd.DataFrame,
        preparation: TrainingManifestResult,
    ) -> None:
        print("\nWriting SVD summary...")

        manifest = preparation.training_manifest

        summary_path = self.paths.reports_dir / "summary.txt"

        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("SVD Experiment Summary\n")
            f.write("======================\n\n")

            f.write(f"Experiment name: {self.experiment_name}\n")
            f.write(f"Dataset root: {self.dataset_root}\n")
            f.write(f"Output root: {self.paths.root_dir}\n\n")

            f.write("Manifest\n")
            f.write("--------\n")
            f.write(f"Raw samples: {len(raw_manifest)}\n")
            f.write(f"Training samples: {len(manifest)}\n")
            f.write(
                f"Excluded rows: "
                f"{len(preparation.excluded_samples)}\n"
            )
            f.write(
                f"Duplicate groups: "
                f"{preparation.summary['duplicate_groups']}\n"
            )

            if "recording_id" in manifest.columns:
                f.write(f"Unique recordings: {manifest['recording_id'].nunique()}\n")

            if "speaker_id" in manifest.columns:
                f.write(f"Unique speakers: {manifest['speaker_id'].nunique()}\n")

            f.write("\n")

            self._write_value_counts(f, manifest, "label", "Class distribution")
            self._write_value_counts(f, manifest, "sex", "Sex distribution")
            self._write_value_counts(f, manifest, "vowel", "Vowel distribution")
            self._write_value_counts(f, manifest, "condition", "Condition distribution")
            self._write_value_counts(
                f,
                manifest,
                "pathology_group",
                "Pathology group distribution",
                top_n=30,
            )
            self._write_value_counts(f, manifest, "pathology", "Pathology distribution", top_n=30)

            if {"vowel", "condition"}.issubset(manifest.columns):
                f.write("Vowel x condition coverage:\n")
                f.write(str(pd.crosstab(manifest["vowel"], manifest["condition"])))
                f.write("\n\n")

            if "speaker_id" in manifest.columns and "recording_id" in manifest.columns:
                sessions_per_speaker = (
                    manifest[["speaker_id", "recording_id"]]
                    .dropna()
                    .drop_duplicates()
                    .groupby("speaker_id")
                    .size()
                )

                f.write("Sessions per speaker:\n")
                f.write(str(sessions_per_speaker.describe()))
                f.write("\n\n")

            if "age" in manifest.columns:
                f.write("Age distribution:\n")
                f.write(str(manifest["age"].describe()))
                f.write("\n\n")

            f.write("Preprocessing config:\n")
            f.write(json.dumps(self._to_dict(self.preprocess_config), indent=4, ensure_ascii=False))
            f.write("\n\n")

            f.write("Manifest config:\n")
            f.write(
                json.dumps(
                    self._to_dict(self.manifest_config),
                    indent=4,
                    ensure_ascii=False,
                )
            )
            f.write("\n")

        print(f"Summary saved in: \n{summary_path}")


    @staticmethod
    def _write_value_counts(
            file,
            manifest: pd.DataFrame,
            column: str,
            title: str,
            top_n: int | None = None
    ) -> None:
        if column not in manifest.columns:
            return

        file.write(f"{title}:\n")

        counts = manifest[column].value_counts(dropna=False)

        if top_n is not None:
            counts = counts.head(top_n)

        file.write(str(counts))
        file.write("\n\n")

    @staticmethod
    def _to_dict(value):
        if value is None:
            return None

        if is_dataclass(value):
            return asdict(value)

        if isinstance(value, dict):
            return value

        return str(value)

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
import json

import pandas as pd

from classes.audio_sample.audio_loader.preprocessing.audio_preprocessor import AudioPreprocessor
from classes.audio_sample.audio_loader.preprocessing.audio_preprocess_config import AudioPreprocessConfig
from classes.audio_sample.audio_loader.profilers.ProcessedAudioProfiler import ProcessedAudioProfiler
from classes.dataset.adapters.hupa_adapter import HUPADatasetAdapter
from classes.experiment.training.training_config import TrainingConfig
from classes.plot.dataset_visualizer import DatasetVisualizer
from classes.experiment.path_manager.experiment_paths import ExperimentPaths
from classes.vpd.feature_extraction_runner import FeatureExtractionRunner
from classes.vpd.vpd_feature_extractor import VPDFeatureExtractor


class HUPAExperimentRunner:
    def __init__(
            self,
            dataset_root: str | Path,
            data_root: str | Path,
            experiment_name: str,
            preprocess_config: AudioPreprocessConfig,
            feature_config=None
    ):
        self.dataset_root = Path(dataset_root)
        self.data_root = Path(data_root)

        self.experiment_name = experiment_name
        self.preprocess_config = preprocess_config
        self.feature_config = feature_config

        self.paths = ExperimentPaths.create(
            data_root=self.data_root,
            dataset_name="HUPA",
            experiment_name=self.experiment_name,
        )


    def run(self) -> None:
        self.save_config()

        manifest = self.build_manifest()
        self.profile_preprocessing(manifest)
        self.generate_plots(manifest)

        if self.feature_config is not None:
            self.extract_features(manifest)

        self.write_summary(manifest)

        print(f"\nExperiment saved in:\n{self.paths.root_dir}")

    def save_config(self) -> None:
        config_data = {
            "dataset_name": "HUPA",
            "dataset_root": str(self.dataset_root),
            "experiment_name": self.experiment_name,
            "preprocess_config": self._to_dict(self.preprocess_config),
            "feature_config": self._to_dict(self.feature_config),
        }

        with open(self.paths.config_path, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=4, ensure_ascii=False)

    def build_manifest(self) -> pd.DataFrame:
        print("\n[1/4] Building HUPA manifest...")


        adapter = HUPADatasetAdapter(
            root_dir=self.dataset_root,
        )

        manifest = adapter.build_manifest()
        adapter.validate_manifest(manifest)

        manifest_path = self.paths.manifests_dir / "hupa_manifest.parquet"
        manifest_csv_path = self.paths.manifests_dir / "hupa_manifest.csv"

        manifest.to_parquet(manifest_path, index=False)
        manifest.to_csv(manifest_csv_path, index=False)

        print("Manifest saved in:\n", manifest_path)
        return manifest


    def profile_preprocessing(self, manifest: pd.DataFrame) -> pd.DataFrame:
        print("\n[2/4] Evaluating preprocessing...")

        preprocessor = AudioPreprocessor(config=self.preprocess_config)
        profiler = ProcessedAudioProfiler(preprocessor)

        profile = profiler.profile_manifest(manifest)

        profile_path = self.paths.profiles_dir / "hupa_processed_audio_profile.parquet"
        profile_csv_path = self.paths.profiles_dir / "hupa_processed_audio_profile.csv"

        profile.to_parquet(profile_path, index=False)
        profile.to_csv(profile_csv_path, index=False)

        print("\nPreprocessing status:")
        print(profile["status"].value_counts(dropna=False))

        print("\nProcessed duration:")
        print(profile["processed_duration"].describe())

        print("\nProcessed RMS:")
        print(profile["processed_rms"].describe())

        print("\nProcessed peak:")
        print(profile["processed_peak"].describe())

        errors = profile[profile["status"] == "error"]

        if not errors.empty:
            print("\nErrors found in preprocessing:")
            print(errors[["sample_id", "filepath", "error"]].head())

        print(f"Profile saved in: {profile_path}")
        return profile

    def generate_plots(self, manifest: pd.DataFrame) -> None:
        print("\n[3/4] Generating plots...")

        plot_dir = self.paths.figures_dir / "hupa_manifest"

        visualizer = DatasetVisualizer(
            manifest=manifest,
            output_dir=plot_dir)

        visualizer.generate_basic_report()
        print("\nGenerated plots saved in:\n", plot_dir)

    def extract_features(self, manifest: pd.DataFrame) -> pd.DataFrame:
        print("\n[4/4] Extracting features...")

        preprocessor = AudioPreprocessor(config=self.preprocess_config)

        feature_extractor = VPDFeatureExtractor(config=self.feature_config)

        runner = FeatureExtractionRunner(
            preprocessor=preprocessor,
            extractor=feature_extractor
        )

        features_df = runner.extract_from_manifest(manifest)

        features_path = self.paths.features_dir / "hupa_features_v1.parquet"
        features_csv_path = self.paths.features_dir / "hupa_features_v1.csv"

        features_df.to_parquet(features_path, index=False)
        features_df.to_csv(features_csv_path, index=False)

        print("\nFeature extraction status:")
        print(features_df["status"].value_counts(dropna=False))

        print("\nFeatures saved in:\n", features_path)
        return features_df

    def write_summary(self, manifest: pd.DataFrame) -> None:
        summary_path = self.paths.reports_dir / "summary.txt"

        with open(summary_path, "w", encoding="utf-8") as file:
            file.write("HUPA Experiment Summary\n")
            file.write("=======================\n\n")

            file.write(f"Experiment name: {self.experiment_name}\n")
            file.write(f"Dataset root: {self.dataset_root}\n")
            file.write(f"Output root: {self.paths.root_dir}\n\n")

            file.write("Manifest\n")
            file.write("--------\n")
            file.write(f"Total samples: {len(manifest)}\n\n")

            if "label" in manifest.columns:
                file.write("Class distribution:\n")
                file.write(str(manifest["label"].value_counts(dropna=False)))
                file.write("\n\n")

            if "sex" in manifest.columns:
                file.write("Sex distribution:\n")
                file.write(str(manifest["sex"].value_counts(dropna=False)))
                file.write("\n\n")

            if "pathology" in manifest.columns:
                file.write("Top pathologies:\n")
                file.write(str(manifest["pathology"].value_counts(dropna=False).head(20)))
                file.write("\n\n")

            file.write("Preprocess config:\n")
            file.write(json.dumps(self._to_dict(self.preprocess_config), indent=4, ensure_ascii=False))
            file.write("\n\n")

            file.write("Feature config:\n")
            file.write(json.dumps(self._to_dict(self.feature_config), indent=4, ensure_ascii=False))
            file.write("\n")

    def train_models(self, features_df: pd.DataFrame) -> pd.DataFrame:
        print("\n[5/5] Training models...")

        training_config = TrainingConfig(
            label_col="label",
            positive_label="pathological",
            test_size=0.15,
            validation_size=0.15,
            random_state=42,
            balance_train=False
        )

        training_runner = ModelTrainingRunner(
            features_df=features_df,
            output_dif=self.paths.root_dir / "training",
            config=training_config
        )

        metrics_df = training_runner.run_all()

        print("\nFinal metrics:")
        print(
            metrics_df.sort_values(
                by=["f1", "auc", "accuracy"],
                ascending=False
            )
        )

        return metrics_df

    @staticmethod
    def _to_dict(value):
        if value is None:
            return None

        if is_dataclass(value):
            return asdict(value)

        if isinstance(value, dict):
            return value

        return str(value)











from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
import json

import pandas as pd

from classes.audio_sample.audio_loader.preprocessing.audio_preprocessor import AudioPreprocessor
from classes.audio_sample.audio_loader.preprocessing.audio_preprocess_config import AudioPreprocessConfig
from classes.audio_sample.audio_loader.profilers.ProcessedAudioProfiler import ProcessedAudioProfiler
from classes.dataset.adapters.svd_adapter import SVDAdapter
from classes.experiment.path_manager.experiment_paths import ExperimentPaths
from classes.plot.dataset_visualizer import DatasetVisualizer


class SVDExperimentRunner:
    def __init__(
            self,
            dataset_root: str | Path,
            data_root: str | Path,
            experiment_name: str,
            preprocess_config: AudioPreprocessConfig,
            feature_config=None,
    ):
        self.dataset_root = Path(dataset_root)
        self.data_root = Path(data_root)
        self.experiment_name = experiment_name
        self.preprocess_config = preprocess_config

        self.paths = ExperimentPaths.create(
            data_root=self.data_root,
            dataset_name="SVD",
            experiment_name=self.experiment_name,
        )

    def run(self) -> None:
        self.save_config()

        manifest = self.build_manifest()
        self.profile_peprocessing(manifest)
        self.generate_plots(manifest)
        self.write_summary(manifest)

        print(f"\nSVD exploration saved in: \n{self.paths.root_dir}")

    def save_config(self) -> None:
        config_data = {
            "dataset_name": "SVD",
            "dataset_root": str(self.dataset_root),
            "experiment_name": self.experiment_name,
            "preprocess_config": self._to_dict(self.preprocess_config),
        }

        with open(self.paths.config_path, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=4, ensure_ascii=False)

    def build_manifest(self) -> pd.DataFrame:
        print(f"[1/4] Building SVD manifest...")

        adapter = SVDAdapter(
            root_dir=self.dataset_root
        )

        manifest = adapter.build_manifest()
        adapter.validate_manifest(manifest)

        manifest_path = self.paths.manifests_dir / "svd_manifest.parquet"
        manifest_csv_path = self.paths.manifests_dir / "svd_manifest.csv"

        manifest.to_parquet(manifest_path, index=False)
        manifest.to_csv(manifest_csv_path, index=False)

        print("Manifest saved in:\n", manifest_path)
        return manifest

    def profile_peprocessing(self, manifest: pd.DataFrame) -> pd.DataFrame:
        print(f"\n[2/4] Evaluating SVD preprocessing...")

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

        if "processed_peaks" in profile.columns:
            print("\nProcessed peak:")
            print(profile["processed_peak"].describe())

        errors = profile[profile["status"] == "error"]

        if not errors.empty:
            print("\nErrors found in preprocessing:")
            print(errors[["sample_id", "filepath", "error"]].head(30))

        print(f"Profile saved in: \n{profile_path}")

        return profile

    def generate_plots(self, manifest: pd.DataFrame) -> None:
        print(f"\n[3/4] Generating SVD plots...")

        plot_dir = self.paths.figures_dir / "svd_manifest"

        visualizer = DatasetVisualizer(
            manifest=manifest,
            output_dir=plot_dir
        )

        visualizer.generate_svd_report()
        print("\nGenerated plots saved in:\n", plot_dir)

    def write_summary(self, manifest: pd.DataFrame) -> None:
        print(f"\n[4/4] Writing SVD summary...")

        summary_path = self.paths.reports_dir / "summary.txt"

        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("SVD Exploration Summary\n")
            f.write("========================\n\n")

            f.write(f"Experiment name: {self.experiment_name}\n")
            f.write(f"Dataset root: {self.dataset_root}\n")
            f.write(f"Output root: {self.paths.root_dir}\n\n")

            f.write("Manifest\n")
            f.write("--------\n")
            f.write(f"Total samples: {len(manifest)}\n")

            if "recording_id" in manifest.columns:
                f.write(f"Unique recordings: {manifest['recording_id'].nunique()}\n")

            if "speaker_id" in manifest.columns:
                f.write(f"Unique speakers: {manifest['speaker_id'].nunique()}\n")

            f.write("\n")

            self._write_value_counts(f, manifest, "label", "Class distribution")
            self._write_value_counts(f, manifest, "sex", "Sex distribution")
            self._write_value_counts(f, manifest, "vowel", "Vowel distribution")
            self._write_value_counts(f, manifest, "condition", "Condition distribution")
            self._write_value_counts(f, manifest, "pathology_group", "Pathology group distribution", top_n=30)
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

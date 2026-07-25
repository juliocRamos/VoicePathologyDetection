from __future__ import annotations

import argparse
from pathlib import Path

from classes.audio_sample.audio_loader.preprocessing.audio_preprocess_config import (
    AudioPreprocessConfig,
)
from classes.dataset.preparation.hupa_training_manifest_builder import (
    HUPATrainingManifestConfig,
)
from classes.dataset.preparation.svd_training_manifest_builder import (
    SVDTrainingManifestConfig,
)
from classes.experiment.runners.hupa_experiment_runner import (
    HUPAExperimentRunner,
)
from classes.experiment.runners.experiment_stage import ExperimentStage
from classes.experiment.runners.svd_experiment_runner import (
    SVDExperimentRunner,
)
from classes.experiment.training.training_config import TrainingConfig
from classes.vpd.vpd_feature_config import VPDFeatureConfig


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data"

HUPA_ROOT = Path(
    "/mnt/d/masters_degree/datasets/hupa/BDAtos HUPA Segmentada"
)

SVD_ROOT = Path(
    "/mnt/d/masters_degree/datasets/svd"
)


def build_preprocess_config() -> AudioPreprocessConfig:
    return AudioPreprocessConfig(
        target_sr=16_000,
        convert_to_mono=True,
        remove_dc=True,
        normalize_rms=True,
        target_dbfs=-20.0,
        peak_limit=0.99,
        center_crop=False,
    )


def build_feature_config() -> VPDFeatureConfig:
    return VPDFeatureConfig(
        n_mfcc=30,
        n_fft=1024,
        hop_length=128,
        top_n_harmonics=30,
        harmonic_min_freq=50.0,
        harmonic_max_freq=None,
        entropy_bins=64,
        zcr_percent=20,
        energy_percent_steps=10,
        include_mfcc_delta=True,
        include_glottal_features=True,
    )


def build_hupa_manifest_config() -> HUPATrainingManifestConfig:
    return HUPATrainingManifestConfig(
        adults_only=False,
        minimum_duration_sec=0.5,
        require_audio_status_ok=True,
        require_speaker_id=True,
    )


def build_svd_manifest_config() -> SVDTrainingManifestConfig:
    return SVDTrainingManifestConfig(
        vowels=("a",),
        conditions=("n",),
        adults_only=True,
        minimum_age=18.0,
        minimum_duration_sec=0.5,
        require_audio_status_ok=True,
        require_speaker_id=True,
    )


def build_training_config() -> TrainingConfig:
    return TrainingConfig(
        label_col="label",
        positive_label="pathological",
        negative_label="healthy",
        group_col="speaker_id",
        test_size=0.20,
        random_state=42,
        cv_folds=5,
        scoring="balanced_accuracy",
        n_jobs=-1,
        bootstrap_iterations=1_000,
        confidence_level=0.95,
        save_models=True,
        save_predictions=True,
        save_cv_results=True,
        save_split_assignments=True,
    )


def run_hupa_experiment(
    preprocess_config: AudioPreprocessConfig,
    feature_config: VPDFeatureConfig,
    manifest_config: HUPATrainingManifestConfig,
    training_config: TrainingConfig,
    experiment_name: str,
    stage: ExperimentStage,
) -> None:
    runner = HUPAExperimentRunner(
        dataset_root=HUPA_ROOT,
        data_root=DATA_ROOT,
        experiment_name=experiment_name,
        preprocess_config=preprocess_config,
        feature_config=feature_config,
        manifest_config=manifest_config,
        training_config=training_config,
    )

    runner.run(stage=stage)


def run_svd_experiment(
    preprocess_config: AudioPreprocessConfig,
    feature_config: VPDFeatureConfig,
    manifest_config: SVDTrainingManifestConfig,
    training_config: TrainingConfig,
    experiment_name: str,
    stage: ExperimentStage,
) -> None:
    runner = SVDExperimentRunner(
        dataset_root=SVD_ROOT,
        data_root=DATA_ROOT,
        experiment_name=experiment_name,
        preprocess_config=preprocess_config,
        feature_config=feature_config,
        manifest_config=manifest_config,
        training_config=training_config,
    )

    runner.run(stage=stage)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute voice pathology detection experiments."
    )

    parser.add_argument(
        "--dataset",
        choices=["hupa", "svd"],
        required=True,
        help="Dataset whose independent experiment will be executed.",
    )

    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Optional experiment name.",
    )

    parser.add_argument(
        "--stage",
        choices=[
            stage.value
            for stage in ExperimentStage
        ],
        default=ExperimentStage.PREPARE.value,
        help=(
            "Last pipeline stage to execute. 'prepare' is the safe "
            "default; 'features' also extracts attributes; 'train' "
            "runs the complete experiment."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    preprocess_config = build_preprocess_config()
    feature_config = build_feature_config()
    training_config = build_training_config()
    stage = ExperimentStage(args.stage)

    experiment_name = (
        args.experiment_name
        or f"{args.dataset}_pre16k_rms20_fullsignal_features_v1"
    )

    if args.dataset == "hupa":
        run_hupa_experiment(
            preprocess_config=preprocess_config,
            feature_config=feature_config,
            manifest_config=build_hupa_manifest_config(),
            training_config=training_config,
            experiment_name=experiment_name,
            stage=stage,
        )

    elif args.dataset == "svd":
        run_svd_experiment(
            preprocess_config=preprocess_config,
            feature_config=feature_config,
            manifest_config=build_svd_manifest_config(),
            training_config=training_config,
            experiment_name=experiment_name,
            stage=stage,
        )

    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path

from classes.audio_sample.audio_loader.preprocessing.audio_preprocess_config import (
    AudioPreprocessConfig,
)
from classes.experiment.runners.hupa_experiment_runner import (
    HUPAExperimentRunner,
)
from classes.experiment.runners.svd_experiment_runner import (
    SVDExperimentRunner,
)
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
        min_duration_sec=0.5,
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


def run_hupa_experiment(
    preprocess_config: AudioPreprocessConfig,
    feature_config: VPDFeatureConfig,
    experiment_name: str,
) -> None:
    runner = HUPAExperimentRunner(
        dataset_root=HUPA_ROOT,
        data_root=DATA_ROOT,
        experiment_name=experiment_name,
        preprocess_config=preprocess_config,
        feature_config=feature_config,
    )

    runner.run()


def run_svd_experiment(
    preprocess_config: AudioPreprocessConfig,
    feature_config: VPDFeatureConfig,
    experiment_name: str,
) -> None:
    runner = SVDExperimentRunner(
        dataset_root=SVD_ROOT,
        data_root=DATA_ROOT,
        experiment_name=experiment_name,
        preprocess_config=preprocess_config,
        feature_config=feature_config,
    )

    runner.run()


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

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    preprocess_config = build_preprocess_config()
    feature_config = build_feature_config()

    experiment_name = (
        args.experiment_name
        or f"{args.dataset}_pre16k_rms20_fullsignal_features_v1"
    )

    if args.dataset == "hupa":
        run_hupa_experiment(
            preprocess_config=preprocess_config,
            feature_config=feature_config,
            experiment_name=experiment_name,
        )

    elif args.dataset == "svd":
        run_svd_experiment(
            preprocess_config=preprocess_config,
            feature_config=feature_config,
            experiment_name=experiment_name,
        )

    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")


if __name__ == "__main__":
    main()

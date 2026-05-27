from pathlib import Path

from classes.audio_sample.audio_loader.preprocessing.audio_preprocess_config import AudioPreprocessConfig
from classes.experiment.runners.hupa_experiment_runner import HUPAExperimentRunner
from classes.vpd.vpd_feature_config import VPDFeatureConfig


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data"

HUPA_ROOT = Path("/mnt/d/masters_degree/datasets/hupa/BDAtos HUPA Segmentada")


def main() -> None:
    preprocess_config = AudioPreprocessConfig(
        target_sr=16_000,
        convert_to_mono=True,
        remove_dc=True,
        normalize_rms=True,
        target_dbfs=-20.0,
        peak_limit=0.99,
        center_crop=False,
        min_duration_sec=0.5,
    )

    feature_config = VPDFeatureConfig(
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
        include_glottal_features=False,
    )

    runner = HUPAExperimentRunner(
        dataset_root=HUPA_ROOT,
        data_root=DATA_ROOT,
        experiment_name="pre16k_rms20_fullsignal_features_v1",
        preprocess_config=preprocess_config,
        feature_config=feature_config,
    )

    runner.run()


if __name__ == "__main__":
    main()
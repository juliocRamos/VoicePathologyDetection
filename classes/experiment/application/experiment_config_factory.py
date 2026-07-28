from __future__ import annotations

from dataclasses import dataclass

from classes.audio_sample.audio_loader.preprocessing.audio_preprocess_config import (
    AudioPreprocessConfig,
)
from classes.dataset.preparation.hupa_training_manifest_builder import (
    HUPATrainingManifestConfig,
)
from classes.dataset.preparation.svd_training_manifest_builder import (
    SVDTrainingManifestConfig,
)
from classes.experiment.training.compute_backend import ComputeBackend
from classes.experiment.training.training_config import TrainingConfig
from classes.vpd.vpd_feature_config import VPDFeatureConfig


@dataclass(frozen=True)
class ExperimentConfigBundle:
    preprocess: AudioPreprocessConfig
    features: VPDFeatureConfig
    hupa_manifest: HUPATrainingManifestConfig
    svd_manifest: SVDTrainingManifestConfig
    training: TrainingConfig


class ExperimentConfigFactory:
    @classmethod
    def build(
        cls,
        compute_backend: ComputeBackend,
    ) -> ExperimentConfigBundle:
        return ExperimentConfigBundle(
            preprocess=cls.build_preprocess_config(),
            features=cls.build_feature_config(),
            hupa_manifest=cls.build_hupa_manifest_config(),
            svd_manifest=cls.build_svd_manifest_config(),
            training=cls.build_training_config(
                compute_backend=compute_backend,
            ),
        )

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def build_hupa_manifest_config() -> HUPATrainingManifestConfig:
        return HUPATrainingManifestConfig(
            adults_only=True,
            minimum_age=18.0,
            minimum_duration_sec=0.5,
            require_audio_status_ok=True,
            require_speaker_id=True,
        )

    @classmethod
    def build_cross_hupa_manifest_config(
        cls,
    ) -> HUPATrainingManifestConfig:
        return cls.build_hupa_manifest_config()

    @staticmethod
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

    @classmethod
    def build_cross_svd_manifest_config(
        cls,
    ) -> SVDTrainingManifestConfig:
        return cls.build_svd_manifest_config()

    @staticmethod
    def build_training_config(
        compute_backend: ComputeBackend,
    ) -> TrainingConfig:
        return TrainingConfig(
            protocol_version=(
                "gpu_confirmatory_v2"
                if compute_backend.uses_cuda
                else "cpu_development_fallback_v1"
            ),
            eligible_for_final_reporting=(
                compute_backend.uses_cuda
            ),
            label_col="label",
            positive_label="pathological",
            negative_label="healthy",
            group_col="speaker_id",
            test_size=0.20,
            random_state=42,
            cv_folds=5,
            selection_score_tolerance=0.005,
            compute_backend=compute_backend,
            scoring="balanced_accuracy",
            grid_search_verbose=2,
            n_jobs=1 if compute_backend.uses_cuda else -1,
            bootstrap_iterations=1_000,
            confidence_level=0.95,
            run_grouped_svm_learning_curve=True,
            learning_curve_train_sizes=(
                0.25,
                0.50,
                0.75,
                1.00,
            ),
            run_repeated_nested_cv=True,
            nested_cv_folds=3,
            nested_cv_repeats=2,
            save_models=True,
            save_predictions=True,
            save_cv_results=True,
            save_split_assignments=True,
            cache_pipeline_transformers=True,
        )

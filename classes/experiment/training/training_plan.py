from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sklearn.base import BaseEstimator
from sklearn.feature_selection import (
    SelectPercentile,
    f_classif,
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVC

from classes.experiment.training.compute_backend import ComputeBackend


@dataclass(frozen=True)
class FeatureScenario:
    name: str
    include_prefixes: tuple[str, ...]
    exclude_prefixes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator: BaseEstimator
    param_grid: list[dict[str, list[Any]]]
    use_balanced_sample_weight: bool = False


class TrainingPlan:
    @staticmethod
    def default_feature_scenarios() -> list[FeatureScenario]:
        return [
            FeatureScenario(
                name="mfcc",
                include_prefixes=("mfcc",),
            ),
            FeatureScenario(
                name="harmonics",
                include_prefixes=("harmonic_",),
            ),
            FeatureScenario(
                name="energy_entropy_zcr",
                include_prefixes=(
                    "energy_area_",
                    "entropy_c2_",
                    "zcr_",
                ),
            ),
            FeatureScenario(
                name="all_without_glottal",
                include_prefixes=(
                    "harmonic_",
                    "energy_area_",
                    "entropy_c2_",
                    "zcr_",
                    "mfcc",
                ),
                exclude_prefixes=("glottal_",),
            ),
            FeatureScenario(
                name="glottal",
                include_prefixes=("glottal_",),
            ),
            FeatureScenario(
                name="all_with_glottal",
                include_prefixes=(
                    "harmonic_",
                    "energy_area_",
                    "entropy_c2_",
                    "zcr_",
                    "mfcc",
                    "glottal_",
                ),
            ),
        ]

    @staticmethod
    def default_model_specs(
        random_state: int = 42,
        compute_backend: ComputeBackend = ComputeBackend.CPU,
    ) -> list[ModelSpec]:
        selectors = TrainingPlan.fast_feature_selectors()

        model_specs = [
            ModelSpec(
                name="svm_linear",
                estimator=SVC(
                    kernel="linear",
                    class_weight="balanced",
                    probability=False,
                    max_iter=20_000,
                    random_state=random_state,
                ),
                param_grid=[
                    {
                        "imputer__strategy": ["median"],
                        "scaler": [
                            StandardScaler(),
                            RobustScaler(),
                        ],
                        "selector": selectors,
                        "classifier__C": [0.1, 1.0, 10.0],
                    }
                ],
            ),
            ModelSpec(
                name="svm_rbf",
                estimator=SVC(
                    kernel="rbf",
                    class_weight="balanced",
                    probability=False,
                    max_iter=20_000,
                    random_state=random_state,
                ),
                param_grid=[
                    {
                        "imputer__strategy": ["median"],
                        "scaler": [
                            StandardScaler(),
                            RobustScaler(),
                        ],
                        "selector": selectors,
                        "classifier__C": [1.0, 10.0, 100.0],
                        "classifier__gamma": ["scale", 0.01, 0.1],
                    }
                ],
            ),
        ]

        if compute_backend.uses_cuda:
            model_specs.append(
                TrainingPlan._cuda_mlp_spec(
                    random_state=random_state,
                    selectors=selectors,
                )
            )
        else:
            model_specs.append(
                ModelSpec(
                    name="mlp",
                    estimator=MLPClassifier(
                        activation="relu",
                        solver="adam",
                        # sklearn's internal early-stopping split is not
                        # group-aware and could mix sessions from one speaker.
                        early_stopping=False,
                        max_iter=400,
                        random_state=random_state,
                    ),
                    param_grid=[
                        {
                            "imputer__strategy": ["median"],
                            "scaler": [
                                StandardScaler(),
                                RobustScaler(),
                            ],
                            "classifier__hidden_layer_sizes": [
                                (64,),
                                (128,),
                                (128, 64),
                            ],
                            "selector": selectors,
                            "classifier__alpha": [0.0001, 0.001],
                            "classifier__learning_rate_init": [0.001],
                        }
                    ],
                    use_balanced_sample_weight=True,
                )
            )

        return model_specs

    @staticmethod
    def fast_feature_selectors() -> list[str | SelectPercentile]:
        return [
            "passthrough",
            SelectPercentile(score_func=f_classif, percentile=50),
            SelectPercentile(score_func=f_classif, percentile=75),
        ]

    @staticmethod
    def _cuda_mlp_spec(
        random_state: int,
        selectors: list[str | SelectPercentile],
    ) -> ModelSpec:
        try:
            from torch import nn, optim

            from classes.experiment.training.torch_mlp_classifier import (
                BalancedTorchMLPClassifier,
                TorchMLPModule,
            )
            from classes.experiment.training.training_accuracy_callback import (
                TrainingAccuracyCallback,
            )
        except ImportError as exc:
            raise RuntimeError(
                "CUDA MLP requires torch and skorch. Install "
                "requirements-gpu.txt."
            ) from exc

        return ModelSpec(
            name="mlp",
            estimator=BalancedTorchMLPClassifier(
                module=TorchMLPModule,
                criterion=nn.CrossEntropyLoss,
                optimizer=optim.Adam,
                lr=0.001,
                max_epochs=150,
                batch_size=-1,
                train_split=None,
                callbacks=[
                    (
                        "training_accuracy",
                        TrainingAccuracyCallback(),
                    ),
                ],
                verbose=0,
                device="cuda",
                iterator_train__shuffle=True,
                random_state=random_state,
            ),
            param_grid=[
                {
                    "imputer__strategy": ["median"],
                    "scaler": [StandardScaler()],
                    "selector": selectors,
                    "classifier__module__hidden_layer_sizes": [
                        (64,),
                        (128, 64),
                    ],
                    "classifier__optimizer__weight_decay": [
                        0.0001,
                        0.001,
                    ],
                    "classifier__lr": [0.001],
                }
            ],
        )

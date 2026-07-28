from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sklearn.base import BaseEstimator
from sklearn.feature_selection import (
    SelectPercentile,
    f_classif,
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
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
    rationale: str = ""


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
        selectors = TrainingPlan.shared_feature_selectors()

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
                        "scaler": [StandardScaler()],
                        "selector": selectors,
                        "classifier__C": [
                            0.01,
                            0.1,
                            1.0,
                            10.0,
                        ],
                    }
                ],
                rationale=(
                    "Lower-capacity linear margin baseline. C controls "
                    "regularization and is selected on grouped source CV."
                ),
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
                        "scaler": [StandardScaler()],
                        "selector": selectors,
                        "classifier__C": [
                            0.01,
                            0.1,
                            1.0,
                            10.0,
                        ],
                        "classifier__gamma": [
                            "scale",
                            0.0001,
                            0.001,
                            0.01,
                        ],
                    }
                ],
                rationale=(
                    "Nonlinear margin model for interactions not captured "
                    "by the linear SVM. Low C and gamma values are included "
                    "to favor smooth decision boundaries."
                ),
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
                        max_iter=30,
                        batch_size=32,
                        tol=0.0,
                        random_state=random_state,
                    ),
                    param_grid=[
                        {
                            "imputer__strategy": ["median"],
                            "scaler": [StandardScaler()],
                            "classifier__hidden_layer_sizes": [
                                (8,),
                                (16,),
                            ],
                            "selector": selectors,
                            "classifier__alpha": [0.001, 0.01],
                            "classifier__learning_rate_init": [0.001],
                            # Epoch count is selected only by the
                            # speaker-grouped outer GridSearchCV.
                            "classifier__max_iter": [
                                5,
                                10,
                                15,
                                20,
                                30,
                            ],
                            "classifier__batch_size": [32],
                        }
                    ],
                    use_balanced_sample_weight=True,
                    rationale=(
                        "CPU-only development fallback. It is not eligible "
                        "for the confirmatory GPU results."
                    ),
                )
            )

        return model_specs

    @staticmethod
    def shared_feature_selectors() -> list[SelectPercentile]:
        return [
            SelectPercentile(score_func=f_classif, percentile=10),
            SelectPercentile(score_func=f_classif, percentile=25),
            SelectPercentile(score_func=f_classif, percentile=50),
        ]

    @staticmethod
    def fast_feature_selectors() -> list[SelectPercentile]:
        return TrainingPlan.shared_feature_selectors()

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
                max_epochs=30,
                batch_size=32,
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
                        (8,),
                        (16,),
                    ],
                    "classifier__module__dropout": [0.2],
                    "classifier__optimizer__weight_decay": [0.001],
                    "classifier__criterion__label_smoothing": [0.05],
                    "classifier__lr": [0.001],
                    "classifier__max_epochs": [
                        5,
                        10,
                        15,
                        20,
                    ],
                    "classifier__batch_size": [32],
                },
                {
                    "imputer__strategy": ["median"],
                    "scaler": [StandardScaler()],
                    "selector": selectors,
                    "classifier__module__hidden_layer_sizes": [
                        (8,),
                        (16,),
                        (16, 8),
                    ],
                    "classifier__module__dropout": [0.4],
                    "classifier__optimizer__weight_decay": [0.01],
                    "classifier__criterion__label_smoothing": [0.10],
                    "classifier__lr": [0.001],
                    "classifier__max_epochs": [
                        5,
                        10,
                        15,
                        20,
                    ],
                    "classifier__batch_size": [32],
                },
            ],
            rationale=(
                "Canonical confirmatory neural network. Compact hidden "
                "layers and short epoch budgets limit capacity. The tapered "
                "(16, 8) architecture is evaluated only under the strong "
                "regularization profile. Bundled moderate and strong "
                "dropout, weight-decay, and label-smoothing profiles control "
                "overconfidence without an internal group-unaware validation "
                "split. Removing the 30-epoch candidate keeps the total CUDA "
                "search budget unchanged."
            ),
        )

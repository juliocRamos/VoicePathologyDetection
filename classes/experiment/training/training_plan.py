from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sklearn.base import BaseEstimator
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVC


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
        ]

    @staticmethod
    def default_model_specs(random_state: int = 42) -> list[ModelSpec]:
        return [
            ModelSpec(
                name="svm_linear",
                estimator=SVC(
                    kernel="linear",
                    class_weight="balanced",
                    probability=False,
                    random_state=random_state,
                ),
                param_grid=[
                    {
                        "imputer__strategy": ["median"],
                        "scaler": [
                            StandardScaler(),
                            RobustScaler(),
                            "passthrough",
                        ],
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
                    random_state=random_state,
                ),
                param_grid=[
                    {
                        "imputer__strategy": ["median"],
                        "scaler": [
                            StandardScaler(),
                            RobustScaler(),
                        ],
                        "classifier__C": [1.0, 10.0, 100.0],
                        "classifier__gamma": ["scale", 0.01, 0.1],
                    }
                ],
            ),

            ModelSpec(
                name="mlp",
                estimator=MLPClassifier(
                    activation="relu",
                    solver="adam",
                    early_stopping=True,
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
                        "classifier__alpha": [0.0001, 0.001],
                        "classifier__learning_rate_init": [0.001],
                    }
                ],
            ),
        ]
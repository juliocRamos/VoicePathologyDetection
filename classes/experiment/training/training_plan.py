from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any

from sklearn.base import BaseEstimator
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import (
    SelectFromModel,
    SelectPercentile,
    f_classif,
    mutual_info_classif,
)
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
                        "selector": (
                            TrainingPlan.nonlinear_feature_selectors(
                                random_state=random_state
                            )
                        ),
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
                        "selector": (
                            TrainingPlan.nonlinear_feature_selectors(
                                random_state=random_state
                            )
                        ),
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
                        "selector": (
                            TrainingPlan.nonlinear_feature_selectors(
                                random_state=random_state
                            )
                        ),
                        "classifier__alpha": [0.0001, 0.001],
                        "classifier__learning_rate_init": [0.001],
                    }
                ],
                use_balanced_sample_weight=True,
            ),
        ]

    @staticmethod
    def default_feature_selectors(
        random_state: int = 42,
    ) -> list[str | SelectPercentile]:
        return [
            "passthrough",
            SelectPercentile(score_func=f_classif, percentile=50),
            SelectPercentile(score_func=f_classif, percentile=75),
            SelectPercentile(
                score_func=partial(
                    mutual_info_classif,
                    random_state=random_state,
                ),
                percentile=50,
            ),
            SelectPercentile(
                score_func=partial(
                    mutual_info_classif,
                    random_state=random_state,
                ),
                percentile=75,
            ),
        ]

    @staticmethod
    def nonlinear_feature_selectors(
        random_state: int = 42,
    ) -> list[str | SelectFromModel]:
        return [
            "passthrough",

            SelectFromModel(
                estimator=ExtraTreesClassifier(
                    n_estimators=300,
                    class_weight="balanced",
                    random_state=random_state,
                    n_jobs=1,
                ),
                threshold="median",
            ),

            SelectFromModel(
                estimator=ExtraTreesClassifier(
                    n_estimators=300,
                    class_weight="balanced",
                    random_state=random_state,
                    n_jobs=1,
                ),
                threshold="mean",
            ),

            SelectFromModel(
                estimator=RandomForestClassifier(
                    n_estimators=300,
                    class_weight="balanced",
                    random_state=random_state,
                    n_jobs=1,
                ),
                threshold="median",
            ),
        ]

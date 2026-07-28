from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any, Sequence

import numpy as np

from sklearn.feature_selection import SelectPercentile


@dataclass(frozen=True)
class ParsimoniousSelection:
    selected_index: int
    numerical_best_index: int
    numerical_best_score: float
    selection_threshold: float
    standard_error: float


class ModelSelectionPolicy:
    MODEL_COMPLEXITY_ORDER = {
        "svm_linear": 0,
        "svm_rbf": 1,
        "mlp": 2,
    }

    @classmethod
    def select_grid_candidate(
        cls,
        mean_scores: Sequence[float],
        std_scores: Sequence[float],
        params: Sequence[dict[str, Any]],
        model_name: str,
        cv_folds: int,
        minimum_score_tolerance: float,
    ) -> ParsimoniousSelection:
        means = np.asarray(mean_scores, dtype=float)
        standard_deviations = np.asarray(std_scores, dtype=float)

        finite_indices = np.flatnonzero(np.isfinite(means))

        if len(finite_indices) == 0:
            raise RuntimeError(
                "No finite cross-validation score is available for "
                "parsimonious selection."
            )

        numerical_best_index = int(
            finite_indices[np.argmax(means[finite_indices])]
        )
        numerical_best_score = float(means[numerical_best_index])
        best_standard_deviation = float(
            standard_deviations[numerical_best_index]
        )

        if not np.isfinite(best_standard_deviation):
            best_standard_deviation = 0.0

        standard_error = best_standard_deviation / sqrt(cv_folds)
        allowed_drop = max(
            standard_error,
            minimum_score_tolerance,
        )
        selection_threshold = numerical_best_score - allowed_drop
        eligible_indices = [
            int(index)
            for index in finite_indices
            if means[index] >= selection_threshold
        ]
        selected_index = min(
            eligible_indices,
            key=lambda index: (
                cls.parameter_complexity_key(
                    model_name=model_name,
                    params=params[index],
                ),
                -float(means[index]),
                index,
            ),
        )

        return ParsimoniousSelection(
            selected_index=selected_index,
            numerical_best_index=numerical_best_index,
            numerical_best_score=numerical_best_score,
            selection_threshold=selection_threshold,
            standard_error=standard_error,
        )

    @classmethod
    def source_candidate_complexity_key(
        cls,
        n_input_features: int,
        best_params: dict[str, Any],
    ) -> tuple[Any, ...]:
        selector_fraction = cls._selector_fraction(
            best_params.get("selector")
        )
        estimated_selected_features = max(
            1,
            round(n_input_features * selector_fraction),
        )

        return (
            estimated_selected_features,
        )

    @classmethod
    def parameter_complexity_key(
        cls,
        model_name: str,
        params: dict[str, Any],
    ) -> tuple[Any, ...]:
        selector_fraction = cls._selector_fraction(
            params.get("selector")
        )
        hidden_layers = cls._hidden_layers(params)
        hidden_neurons = sum(hidden_layers)
        hidden_depth = len(hidden_layers)
        epochs = cls._numeric_param(
            params,
            "classifier__max_epochs",
            "classifier__max_iter",
            default=0.0,
        )
        c_value = cls._numeric_param(
            params,
            "classifier__C",
            default=0.0,
        )
        gamma = cls._gamma_complexity(
            params.get("classifier__gamma")
        )
        regularization = cls._numeric_param(
            params,
            "classifier__optimizer__weight_decay",
            "classifier__alpha",
            default=0.0,
        )
        dropout = cls._numeric_param(
            params,
            "classifier__module__dropout",
            default=0.0,
        )

        return (
            cls.MODEL_COMPLEXITY_ORDER.get(model_name, 10),
            selector_fraction,
            hidden_neurons,
            hidden_depth,
            epochs,
            c_value,
            gamma,
            -regularization,
            -dropout,
        )

    @staticmethod
    def _selector_fraction(selector: Any) -> float:
        if isinstance(selector, SelectPercentile):
            return float(selector.percentile) / 100.0

        return 1.0

    @staticmethod
    def _hidden_layers(
        params: dict[str, Any],
    ) -> tuple[int, ...]:
        value = params.get(
            "classifier__module__hidden_layer_sizes",
            params.get(
                "classifier__hidden_layer_sizes",
                (),
            ),
        )

        if isinstance(value, int):
            return (value,)

        return tuple(value)

    @staticmethod
    def _numeric_param(
        params: dict[str, Any],
        *names: str,
        default: float,
    ) -> float:
        for name in names:
            if name in params:
                return float(params[name])

        return default

    @staticmethod
    def _gamma_complexity(value: Any) -> float:
        if value is None:
            return 0.0

        if value in {"scale", "auto"}:
            return float("inf")

        return float(value)

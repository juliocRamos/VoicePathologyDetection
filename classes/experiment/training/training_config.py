from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingConfig:
    label_col: str = "label"
    positive_label: str = "pathological"
    negative_label: str = "healthy"
    group_col: str | None = None

    test_size: float = 0.20
    random_state: int = 42

    cv_folds: int = 5
    scoring: str = "balanced_accuracy"
    n_jobs: int = -1
    bootstrap_iterations: int = 1_000
    confidence_level: float = 0.95

    save_models: bool = True
    save_predictions: bool = True
    save_cv_results: bool = True
    save_split_assignments: bool = True

    def __post_init__(self) -> None:
        if self.positive_label == self.negative_label:
            raise ValueError(
                "positive_label and negative_label must be different."
            )

        if not 0 < self.test_size < 1:
            raise ValueError("test_size must be between 0 and 1.")

        if self.cv_folds < 2:
            raise ValueError("cv_folds must be at least 2.")

        if self.bootstrap_iterations < 0:
            raise ValueError(
                "bootstrap_iterations must be non-negative."
            )

        if not 0 < self.confidence_level < 1:
            raise ValueError(
                "confidence_level must be between 0 and 1."
            )

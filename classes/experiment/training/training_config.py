from __future__ import annotations

from dataclasses import dataclass

from classes.experiment.training.compute_backend import ComputeBackend


@dataclass(frozen=True)
class TrainingConfig:
    protocol_version: str = "development_v1"
    eligible_for_final_reporting: bool = False
    label_col: str = "label"
    positive_label: str = "pathological"
    negative_label: str = "healthy"
    group_col: str | None = None

    test_size: float = 0.20
    random_state: int = 42

    cv_folds: int = 5
    selection_score_tolerance: float = 0.005
    cache_pipeline_transformers: bool = True
    scoring: str = "balanced_accuracy"
    strict_model_selection: bool = True
    stratify_col: str | None = None
    evaluation_subgroup_col: str | None = None
    grid_search_verbose: int = 2
    compute_backend: ComputeBackend = ComputeBackend.CPU
    n_jobs: int = -1
    bootstrap_iterations: int = 1_000
    confidence_level: float = 0.95

    run_grouped_svm_learning_curve: bool = False
    learning_curve_train_sizes: tuple[float, ...] = (
        0.25,
        0.50,
        0.75,
        1.00,
    )
    run_repeated_nested_cv: bool = False
    nested_cv_folds: int = 3
    nested_cv_repeats: int = 2

    save_models: bool = True
    save_predictions: bool = True
    save_cv_results: bool = True
    save_split_assignments: bool = True

    def __post_init__(self) -> None:
        if not self.protocol_version.strip():
            raise ValueError(
                "protocol_version cannot be empty."
            )

        if self.positive_label == self.negative_label:
            raise ValueError(
                "positive_label and negative_label must be different."
            )

        if not 0 < self.test_size < 1:
            raise ValueError("test_size must be between 0 and 1.")

        if self.cv_folds < 2:
            raise ValueError("cv_folds must be at least 2.")

        if self.selection_score_tolerance < 0:
            raise ValueError(
                "selection_score_tolerance must be non-negative."
            )

        if self.bootstrap_iterations < 0:
            raise ValueError(
                "bootstrap_iterations must be non-negative."
            )

        if not 0 < self.confidence_level < 1:
            raise ValueError(
                "confidence_level must be between 0 and 1."
            )

        if not isinstance(self.compute_backend, ComputeBackend):
            raise TypeError(
                "compute_backend must be a ComputeBackend instance."
            )

        if (
            self.eligible_for_final_reporting
            and not self.compute_backend.uses_cuda
        ):
            raise ValueError(
                "Only CUDA experiments are eligible for final "
                "reporting."
            )

        if self.n_jobs == 0:
            raise ValueError("n_jobs cannot be zero.")

        if self.grid_search_verbose < 0:
            raise ValueError(
                "grid_search_verbose cannot be negative."
            )

        if self.nested_cv_folds < 2:
            raise ValueError(
                "nested_cv_folds must be at least 2."
            )

        if self.nested_cv_repeats < 1:
            raise ValueError(
                "nested_cv_repeats must be at least 1."
            )

        if (
            not self.learning_curve_train_sizes
            or any(
                not 0 < size <= 1
                for size in self.learning_curve_train_sizes
            )
        ):
            raise ValueError(
                "learning_curve_train_sizes must contain values in "
                "(0, 1]."
            )

        if tuple(sorted(set(
            self.learning_curve_train_sizes
        ))) != self.learning_curve_train_sizes:
            raise ValueError(
                "learning_curve_train_sizes must be unique and sorted."
            )

        for field_name, column_name in (
            ("stratify_col", self.stratify_col),
            (
                "evaluation_subgroup_col",
                self.evaluation_subgroup_col,
            ),
        ):
            if (
                column_name is not None
                and not column_name.strip()
            ):
                raise ValueError(
                    f"{field_name} cannot be an empty string."
                )

        if (
            self.compute_backend is ComputeBackend.CUDA
            and self.n_jobs != 1
        ):
            raise ValueError(
                "CUDA training requires n_jobs=1. "
                "Parallelism is handled by the GPU."
            )

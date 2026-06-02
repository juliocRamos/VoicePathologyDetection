from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainingConfig:
    label_col: str = "label"
    positive_label: str = "pathological"

    test_size: float = 0.20
    random_state: int = 42

    cv_folds: int = 5
    scoring: str = "f1"
    n_jobs: int = -1

    save_models: bool = True
    save_predictions: bool = True
    save_cv_results: bool = True
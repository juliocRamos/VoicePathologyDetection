from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import Pipeline

from classes.experiment.training.training_config import TrainingConfig
from classes.plot.grouped_learning_curve_visualizer import (
    GroupedLearningCurveVisualizer,
)


class GroupedSVMLearningCurveRunner:
    def __init__(
        self,
        config: TrainingConfig,
        learning_curves_dir: str | Path,
        splits_dir: str | Path,
    ) -> None:
        self.config = config
        self.learning_curves_dir = Path(learning_curves_dir)
        self.splits_dir = Path(splits_dir)

    def run(
        self,
        model: Pipeline,
        scenario_name: str,
        model_name: str,
        X: pd.DataFrame,
        y: pd.Series,
        groups: pd.Series,
        strata: pd.Series,
        cv_splits: list[tuple[np.ndarray, np.ndarray]],
    ) -> None:
        records: list[dict[str, Any]] = []
        assignment_records: list[dict[str, Any]] = []

        for fold_index, (
            fold_fit_indices,
            validation_indices,
        ) in enumerate(cv_splits):
            validation_groups = set(
                groups.iloc[validation_indices]
            )

            for train_fraction in (
                self.config.learning_curve_train_sizes
            ):
                selected_fit_indices = self._select_fit_indices(
                    candidate_indices=fold_fit_indices,
                    groups=groups,
                    strata=strata,
                    train_fraction=train_fraction,
                    random_state=(
                        self.config.random_state
                        + fold_index * 1_000
                        + round(train_fraction * 100)
                    ),
                )
                selected_groups = set(
                    groups.iloc[selected_fit_indices]
                )

                if selected_groups & validation_groups:
                    raise RuntimeError(
                        "Speaker leakage found in the grouped SVM "
                        "learning curve."
                    )

                curve_model = clone(model)
                curve_model.fit(
                    X.iloc[selected_fit_indices],
                    y.iloc[selected_fit_indices],
                )
                train_prediction = curve_model.predict(
                    X.iloc[selected_fit_indices]
                )
                validation_prediction = curve_model.predict(
                    X.iloc[validation_indices]
                )
                train_score = balanced_accuracy_score(
                    y.iloc[selected_fit_indices],
                    train_prediction,
                )
                validation_score = balanced_accuracy_score(
                    y.iloc[validation_indices],
                    validation_prediction,
                )
                records.append({
                    "fold": fold_index,
                    "train_fraction": float(train_fraction),
                    "n_train_samples": len(selected_fit_indices),
                    "n_train_groups": len(selected_groups),
                    "n_validation_samples": len(
                        validation_indices
                    ),
                    "n_validation_groups": len(
                        validation_groups
                    ),
                    "train_balanced_accuracy": float(
                        train_score
                    ),
                    "validation_balanced_accuracy": float(
                        validation_score
                    ),
                    "generalization_gap": float(
                        train_score - validation_score
                    ),
                })

                assignment_records.extend(
                    self._assignment_rows(
                        fold_index=fold_index,
                        train_fraction=train_fraction,
                        groups=selected_groups,
                        partition="curve_train",
                    )
                )
                assignment_records.extend(
                    self._assignment_rows(
                        fold_index=fold_index,
                        train_fraction=train_fraction,
                        groups=validation_groups,
                        partition="curve_validation",
                    )
                )

        curve_df = pd.DataFrame(records)
        summary = (
            curve_df.groupby("train_fraction", as_index=False)
            .agg(
                mean_train_samples=("n_train_samples", "mean"),
                mean_train_groups=("n_train_groups", "mean"),
                mean_train_balanced_accuracy=(
                    "train_balanced_accuracy",
                    "mean",
                ),
                std_train_balanced_accuracy=(
                    "train_balanced_accuracy",
                    "std",
                ),
                mean_validation_balanced_accuracy=(
                    "validation_balanced_accuracy",
                    "mean",
                ),
                std_validation_balanced_accuracy=(
                    "validation_balanced_accuracy",
                    "std",
                ),
                mean_generalization_gap=(
                    "generalization_gap",
                    "mean",
                ),
            )
        )
        file_stem = (
            f"{scenario_name}_{model_name}_grouped_learning_curve"
        )
        curve_df.to_csv(
            self.learning_curves_dir / f"{file_stem}.csv",
            index=False,
        )
        summary.to_csv(
            self.learning_curves_dir
            / f"{file_stem}_summary.csv",
            index=False,
        )
        pd.DataFrame(assignment_records).to_csv(
            self.splits_dir / f"{file_stem}_assignments.csv",
            index=False,
        )
        GroupedLearningCurveVisualizer.save(
            summary=summary,
            output_path=(
                self.learning_curves_dir / f"{file_stem}.png"
            ),
            title=(
                "Speaker-grouped learning curve — "
                f"{scenario_name} / {model_name}"
            ),
        )

    @staticmethod
    def _assignment_rows(
        fold_index: int,
        train_fraction: float,
        groups: set[Any],
        partition: str,
    ) -> list[dict[str, Any]]:
        return [
            {
                "fold": fold_index,
                "train_fraction": float(train_fraction),
                "group": group,
                "partition": partition,
            }
            for group in groups
        ]

    @staticmethod
    def _select_fit_indices(
        candidate_indices: np.ndarray,
        groups: pd.Series,
        strata: pd.Series,
        train_fraction: float,
        random_state: int,
    ) -> np.ndarray:
        candidate_frame = pd.DataFrame({
            "index": candidate_indices,
            "group": groups.iloc[candidate_indices].to_numpy(),
            "stratum": strata.iloc[candidate_indices].to_numpy(),
        })
        group_strata_counts = (
            candidate_frame.groupby("group")["stratum"].nunique()
        )

        if (group_strata_counts > 1).any():
            raise ValueError(
                "Each speaker group must belong to one stratum when "
                "building a learning curve."
            )

        group_frame = (
            candidate_frame[["group", "stratum"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )

        if train_fraction >= 1.0:
            return np.sort(candidate_indices)

        number_of_strata = group_frame["stratum"].nunique()
        requested_groups = max(
            number_of_strata,
            round(len(group_frame) * train_fraction),
        )

        if requested_groups >= len(group_frame):
            return np.sort(candidate_indices)

        stratum_counts = group_frame["stratum"].value_counts()
        target_counts = {
            stratum: min(
                count,
                max(
                    1,
                    int(np.floor(count * train_fraction)),
                ),
            )
            for stratum, count in stratum_counts.items()
        }

        while sum(target_counts.values()) < requested_groups:
            expandable = [
                stratum
                for stratum, count in stratum_counts.items()
                if target_counts[stratum] < count
            ]

            if not expandable:
                break

            stratum = max(
                expandable,
                key=lambda value: (
                    stratum_counts[value] * train_fraction
                    - target_counts[value],
                    str(value),
                ),
            )
            target_counts[stratum] += 1

        generator = np.random.default_rng(random_state)
        selected_group_set = set()

        for stratum, number_to_select in target_counts.items():
            available_groups = group_frame.loc[
                group_frame["stratum"] == stratum,
                "group",
            ].to_numpy()
            chosen_groups = generator.choice(
                available_groups,
                size=number_to_select,
                replace=False,
            )
            selected_group_set.update(chosen_groups.tolist())

        selected_indices = candidate_frame.loc[
            candidate_frame["group"].isin(selected_group_set),
            "index",
        ].to_numpy(dtype=int)

        if strata.iloc[selected_indices].nunique() != number_of_strata:
            raise RuntimeError(
                "Learning-curve subset lost a required stratum."
            )

        return np.sort(selected_indices)

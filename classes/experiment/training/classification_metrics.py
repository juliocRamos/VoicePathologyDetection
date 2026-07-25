from __future__ import annotations

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


class ClassificationMetrics:
    @staticmethod
    def compute_binary_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_score: np.ndarray | None = None,
    ) -> dict[str, float]:
        tn, fp, fn, tp = confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1],
        ).ravel()

        specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan

        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(
                balanced_accuracy_score(y_true, y_pred)
            ),
            "uar": float(
                recall_score(
                    y_true,
                    y_pred,
                    average="macro",
                    zero_division=0,
                )
            ),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
            "specificity": float(specificity),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "macro_f1": float(
                f1_score(
                    y_true,
                    y_pred,
                    average="macro",
                    zero_division=0,
                )
            ),
            "mcc": float(matthews_corrcoef(y_true, y_pred)),
            "auc": np.nan,
            "pr_auc": np.nan,
            "tn": float(tn),
            "fp": float(fp),
            "fn": float(fn),
            "tp": float(tp),
        }

        if y_score is not None and len(np.unique(y_true)) == 2:
            try:
                metrics["auc"] = float(roc_auc_score(y_true, y_score))
            except ValueError:
                metrics["auc"] = np.nan

            try:
                metrics["pr_auc"] = float(
                    average_precision_score(y_true, y_score)
                )
            except ValueError:
                metrics["pr_auc"] = np.nan

        return metrics

    @staticmethod
    def compute_grouped_bootstrap_intervals(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        groups: np.ndarray,
        y_score: np.ndarray | None = None,
        iterations: int = 1_000,
        confidence_level: float = 0.95,
        random_state: int = 42,
    ) -> dict[str, float]:
        if iterations == 0:
            return {}

        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        groups = np.asarray(groups)
        y_score_array = (
            np.asarray(y_score)
            if y_score is not None
            else None
        )

        if not (
            len(y_true) == len(y_pred) == len(groups)
        ):
            raise ValueError(
                "y_true, y_pred, and groups must have the same length."
            )

        unique_groups = np.unique(groups)

        if len(unique_groups) < 2:
            return {}

        group_indices = {
            group: np.flatnonzero(groups == group)
            for group in unique_groups
        }
        metric_names = [
            "balanced_accuracy",
            "uar",
            "macro_f1",
            "mcc",
            "sensitivity",
            "specificity",
            "auc",
            "pr_auc",
        ]
        samples: dict[str, list[float]] = {
            metric_name: []
            for metric_name in metric_names
        }
        generator = np.random.default_rng(random_state)

        for _ in range(iterations):
            sampled_groups = generator.choice(
                unique_groups,
                size=len(unique_groups),
                replace=True,
            )
            sampled_indices = np.concatenate([
                group_indices[group]
                for group in sampled_groups
            ])
            sampled_true = y_true[sampled_indices]

            if len(np.unique(sampled_true)) != 2:
                continue

            sampled_score = (
                y_score_array[sampled_indices]
                if y_score_array is not None
                else None
            )
            metrics = (
                ClassificationMetrics.compute_binary_metrics(
                    y_true=sampled_true,
                    y_pred=y_pred[sampled_indices],
                    y_score=sampled_score,
                )
            )

            for metric_name in metric_names:
                value = metrics[metric_name]

                if np.isfinite(value):
                    samples[metric_name].append(float(value))

        alpha = (1.0 - confidence_level) / 2.0
        intervals: dict[str, float] = {}

        for metric_name, values in samples.items():
            if not values:
                intervals[f"{metric_name}_ci_lower"] = np.nan
                intervals[f"{metric_name}_ci_upper"] = np.nan
                continue

            intervals[f"{metric_name}_ci_lower"] = float(
                np.quantile(values, alpha)
            )
            intervals[f"{metric_name}_ci_upper"] = float(
                np.quantile(values, 1.0 - alpha)
            )

        intervals["bootstrap_valid_iterations"] = float(
            len(samples["balanced_accuracy"])
        )
        return intervals

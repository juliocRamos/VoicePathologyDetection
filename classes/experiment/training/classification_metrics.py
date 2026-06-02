from __future__ import annotations

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
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
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
            "specificity": float(specificity),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "auc": np.nan,
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

        return metrics
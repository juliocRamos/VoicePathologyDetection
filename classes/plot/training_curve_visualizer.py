from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


class TrainingCurveVisualizer:
    @staticmethod
    def save(
        history: pd.DataFrame,
        output_path: str | Path,
        title: str,
    ) -> None:
        metric_groups = [
            (
                "Cross-entropy loss",
                ("train_loss", "valid_loss"),
                None,
            ),
            (
                "Accuracy",
                ("train_accuracy", "valid_accuracy"),
                (0.0, 1.0),
            ),
            (
                "Balanced accuracy",
                (
                    "train_balanced_accuracy",
                    "valid_balanced_accuracy",
                ),
                (0.0, 1.0),
            ),
        ]
        available_groups = [
            (
                axis_title,
                [
                    metric
                    for metric in metrics
                    if metric in history.columns
                ],
                y_limits,
            )
            for axis_title, metrics, y_limits in metric_groups
            if any(metric in history.columns for metric in metrics)
        ]

        if not available_groups:
            return

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        figure, axes = plt.subplots(
            1,
            len(available_groups),
            figsize=(7 * len(available_groups), 5),
            squeeze=False,
        )

        for axis, (
            axis_title,
            metrics,
            y_limits,
        ) in zip(axes[0], available_groups):
            for metric in metrics:
                axis.plot(
                    history["epoch"],
                    history[metric],
                    linewidth=2,
                    label=metric.replace("_", " ").title(),
                )

            axis.set_xlabel("Epoch")
            axis.set_title(axis_title)
            axis.set_ylabel(axis_title)
            axis.grid(alpha=0.3)

            if y_limits is not None:
                axis.set_ylim(*y_limits)

            if len(metrics) > 1:
                axis.legend()

        figure.suptitle(title)
        figure.tight_layout()
        figure.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(figure)

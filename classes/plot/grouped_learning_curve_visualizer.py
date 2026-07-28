from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


class GroupedLearningCurveVisualizer:
    @staticmethod
    def save(
        summary: pd.DataFrame,
        output_path: str | Path,
        title: str,
    ) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        figure, axis = plt.subplots(figsize=(9, 6))
        x_values = summary["mean_train_samples"]

        for prefix, label, color in (
            ("train", "Training", "#1f77b4"),
            ("validation", "Grouped validation", "#d62728"),
        ):
            means = summary[
                f"mean_{prefix}_balanced_accuracy"
            ]
            standard_deviations = summary[
                f"std_{prefix}_balanced_accuracy"
            ].fillna(0.0)
            axis.plot(
                x_values,
                means,
                marker="o",
                label=label,
                color=color,
            )
            axis.fill_between(
                x_values,
                means - standard_deviations,
                means + standard_deviations,
                alpha=0.15,
                color=color,
            )

        axis.set_title(title)
        axis.set_xlabel("Training samples")
        axis.set_ylabel("Balanced accuracy")
        axis.set_ylim(0.0, 1.05)
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(output_path, dpi=160)
        plt.close(figure)

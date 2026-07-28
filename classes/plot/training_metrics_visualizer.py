from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    confusion_matrix,
    roc_curve,
    auc,
)


class TrainingMetricsVisualizer:
    METRIC_COLUMNS = [
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "macro_f1",
        "mcc",
        "auc",
        "pr_auc",
    ]

    def __init__(
        self,
        metrics_df: pd.DataFrame,
        predictions_dir: str | Path,
        output_dir: str | Path,
        ranking_df: pd.DataFrame | None = None,
    ):
        self.metrics_df = metrics_df.copy()
        self.ranking_df = (
            ranking_df.copy()
            if ranking_df is not None
            else pd.DataFrame()
        )
        self.predictions_dir = Path(predictions_dir)
        self.output_dir = Path(output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _add_bar_labels(ax, fmt: str = "{:.3f}") -> None:
        for container in ax.containers:
            ax.bar_label(
                container,
                labels=[
                    fmt.format(value.get_height())
                    for value in container
                ],
                padding=3,
                fontsize=9,
            )

    @staticmethod
    def _add_horizontal_bar_labels(ax, fmt: str = "{:.3f}") -> None:
        for container in ax.containers:
            ax.bar_label(
                container,
                labels=[
                    fmt.format(value.get_width())
                    for value in container
                ],
                padding=3,
                fontsize=9,
            )

    def generate_best_models_report(
        self,
        best_metric: str = "f1",
    ) -> None:
        family_champions_df = self._validate_family_champions()

        if family_champions_df.empty:
            print("[WARNING] No SVM/MLP family champion found to plot.")
            return

        self.plot_family_champions_metrics(
            family_champions_df=family_champions_df,
            filename=(
                "training_cv_family_champions_holdout_metrics.png"
            ),
        )

        self.plot_top_models(
            best_metric=best_metric,
            filename=(
                "top_5_training_cv_pipelines_by_"
                f"{best_metric}.png"
            ),
            top_n=5,
        )

        self.plot_confusion_matrices_for_best_models(
            best_models_df=family_champions_df,
        )

        self.plot_roc_curves_for_best_models(
            best_models_df=family_champions_df,
            filename=(
                "training_cv_family_champions_holdout_roc.png"
            ),
        )

    def _validate_family_champions(self) -> pd.DataFrame:
        """Return prespecified champions without ranking by holdout data."""
        df = self.metrics_df.copy()

        champion_rows = []

        for family, pattern in (
            ("SVM", "svm"),
            ("MLP", "mlp"),
        ):
            family_df = df[
                df["model"].astype(str).str.contains(
                    pattern,
                    case=False,
                    na=False,
                )
            ]

            if len(family_df) > 1:
                raise ValueError(
                    f"Expected at most one prespecified {family} "
                    "champion. Family champions must be selected by "
                    "training CV before holdout visualization."
                )

            if len(family_df) == 1:
                champion_rows.append(family_df.iloc[0])

        if not champion_rows:
            return pd.DataFrame()

        return pd.DataFrame(champion_rows)

    def plot_family_champions_metrics(
        self,
        family_champions_df: pd.DataFrame,
        filename: str,
    ) -> None:
        metric_cols = [
            col for col in self.METRIC_COLUMNS
            if col in family_champions_df.columns
        ]

        labels = [
            f"{row['model']}\n{row['scenario']}"
            for _, row in family_champions_df.iterrows()
        ]

        x = np.arange(len(metric_cols))
        width = 0.8 / len(family_champions_df)
        minimum_value = float(
            family_champions_df[metric_cols]
            .astype(float)
            .min()
            .min()
        )

        fig, ax = plt.subplots(figsize=(12, 6))

        for i, (_, row) in enumerate(
            family_champions_df.iterrows()
        ):
            values = row[metric_cols].astype(float).to_numpy()
            offset = (
                i - (len(family_champions_df) - 1) / 2
            ) * width

            ax.bar(
                x + offset,
                values,
                width,
                label=labels[i],
            )

        ax.set_xticks(x)
        ax.set_xticklabels(metric_cols, rotation=30, ha="right")
        ax.set_ylim(
            min(0.0, minimum_value - 0.1),
            1.12,
        )
        ax.set_ylabel("Metric value")
        ax.set_title(
            "Holdout performance of training-CV family champions"
        )
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        self._add_bar_labels(ax, fmt="{:.3f}")

        self._save(filename)

    def plot_top_models(
        self,
        best_metric: str = "f1",
        filename: str = "top_models_by_f1.png",
        top_n: int = 5,
    ) -> None:
        if (
            not self.ranking_df.empty
            and "best_cv_score" in self.ranking_df.columns
        ):
            df = self.ranking_df.copy()
            ranking_column = "best_cv_score"
            axis_label = f"{best_metric} (training CV)"
            title = (
                f"Top {top_n} pipelines by training-CV "
                f"{best_metric}"
            )
        else:
            df = self.metrics_df.copy()
            ranking_column = best_metric
            axis_label = best_metric
            title = f"Top {top_n} models by {best_metric}"

        if ranking_column not in df.columns:
            raise ValueError(
                f"Metric '{ranking_column}' not found in ranking data."
            )

        df = df.dropna(subset=[ranking_column]).copy()

        df["model_label"] = (
            df["model"].astype(str)
            + " | "
            + df["scenario"].astype(str)
        )

        top_df = df.sort_values(
            by=ranking_column,
            ascending=False,
        ).head(top_n)

        fig, ax = plt.subplots(figsize=(12, 7))

        ax.barh(
            top_df["model_label"][::-1],
            top_df[ranking_column][::-1],
        )

        ax.set_xlim(0.0, 1.12)
        ax.set_xlabel(axis_label)
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.3)

        self._add_horizontal_bar_labels(ax, fmt="{:.3f}")

        self._save(filename)

    def plot_confusion_matrices_for_best_models(
        self,
        best_models_df: pd.DataFrame,
    ) -> None:
        for _, row in best_models_df.iterrows():
            scenario = str(row["scenario"])
            model = str(row["model"])

            predictions_df = self._load_predictions(
                scenario=scenario,
                model=model,
            )

            if predictions_df is None:
                print(f"[WARNING] No predictions found for {scenario} | {model}")
                continue

            self.plot_confusion_matrix(
                predictions_df=predictions_df,
                scenario=scenario,
                model=model,
                filename=f"{scenario}_{model}_confusion_matrix.png",
            )

    def plot_confusion_matrix(
        self,
        predictions_df: pd.DataFrame,
        scenario: str,
        model: str,
        filename: str,
    ) -> None:
        y_true = predictions_df["y_true"].to_numpy()
        y_pred = predictions_df["y_pred"].to_numpy()

        cm = confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1],
        )

        display = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["normal", "pathological"],
        )

        fig, ax = plt.subplots(figsize=(6, 5))

        display.plot(
            ax=ax,
            values_format="d",
            colorbar=False,
        )

        ax.set_title(f"Confusion matrix\n{model} | {scenario}")

        self._save(filename)

    def plot_roc_curves_for_best_models(
        self,
        best_models_df: pd.DataFrame,
        filename: str,
    ) -> None:
        fig, ax = plt.subplots(figsize=(7, 6))

        plotted_any = False

        for _, row in best_models_df.iterrows():
            scenario = str(row["scenario"])
            model = str(row["model"])

            predictions_df = self._load_predictions(
                scenario=scenario,
                model=model,
            )

            if predictions_df is None:
                continue

            if "y_score" not in predictions_df.columns:
                print(f"[WARNING] missing y_score for {scenario} | {model}")
                continue

            y_true = predictions_df["y_true"].to_numpy()
            y_score = predictions_df["y_score"].to_numpy()

            if len(np.unique(y_true)) < 2:
                continue

            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)

            ax.plot(
                fpr,
                tpr,
                label=f"{model} | {scenario} | AUC={roc_auc:.3f}",
            )

            plotted_any = True

        if not plotted_any:
            print("[WARNING] No ROC curve was created.")
            plt.close()
            return

        ax.plot([0, 1], [0, 1], linestyle="--", label="Random")

        ax.set_xlabel("False positive rate (FPR)")
        ax.set_ylabel("True positive rate (TPR)")
        ax.set_title(
            "ROC curves of training-CV family champions"
        )
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)

        self._save(filename)

    def _load_predictions(
        self,
        scenario: str,
        model: str,
    ) -> pd.DataFrame | None:
        expected_path = self.predictions_dir / f"{scenario}_{model}_predictions.csv"

        if expected_path.exists():
            return pd.read_csv(expected_path)

        candidates = list(self.predictions_dir.glob("*_predictions.csv"))

        for path in candidates:
            name = path.name

            if scenario in name and model in name:
                return pd.read_csv(path)

        return None

    def _save(
        self,
        filename: str,
    ) -> None:
        output_path = self.output_dir / filename

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Plot saved in: {output_path}")

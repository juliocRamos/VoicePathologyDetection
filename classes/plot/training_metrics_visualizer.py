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
        "accuracy",
        "precision",
        "sensitivity",
        "specificity",
        "f1",
        "auc",
    ]

    def __init__(
        self,
        metrics_df: pd.DataFrame,
        predictions_dir: str | Path,
        output_dir: str | Path,
    ):
        self.metrics_df = metrics_df.copy()
        self.predictions_dir = Path(predictions_dir)
        self.output_dir = Path(output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_best_models_report(
        self,
        best_metric: str = "f1",
    ) -> None:
        best_models_df = self._select_best_svm_and_mlp(
            best_metric=best_metric,
        )

        if best_models_df.empty:
            print("[WARNING] No SVM/MLP model found to use plot.")
            return

        self.plot_best_svm_vs_mlp_metrics(
            best_models_df=best_models_df,
            filename="best_svm_vs_mlp_metrics.png",
        )

        self.plot_top_models(
            best_metric=best_metric,
            filename=f"top_models_by_{best_metric}.png",
        )

        self.plot_confusion_matrices_for_best_models(
            best_models_df=best_models_df,
        )

        self.plot_roc_curves_for_best_models(
            best_models_df=best_models_df,
            filename="best_svm_mlp_roc_curve.png",
        )

    def _select_best_svm_and_mlp(
        self,
        best_metric: str = "f1",
    ) -> pd.DataFrame:
        df = self.metrics_df.copy()

        if best_metric not in df.columns:
            raise ValueError(f"Metric '{best_metric}' not found in metrics_df.")

        df = df.dropna(subset=[best_metric]).copy()

        best_rows = []

        svm_df = df[df["model"].astype(str).str.contains("svm", case=False, na=False)]
        mlp_df = df[df["model"].astype(str).str.contains("mlp", case=False, na=False)]

        if not svm_df.empty:
            best_rows.append(
                svm_df.sort_values(
                    by=[best_metric, "accuracy"],
                    ascending=False,
                ).iloc[0]
            )

        if not mlp_df.empty:
            best_rows.append(
                mlp_df.sort_values(
                    by=[best_metric, "accuracy"],
                    ascending=False,
                ).iloc[0]
            )

        if not best_rows:
            return pd.DataFrame()

        return pd.DataFrame(best_rows)

    def plot_best_svm_vs_mlp_metrics(
        self,
        best_models_df: pd.DataFrame,
        filename: str,
    ) -> None:
        metric_cols = [
            col for col in self.METRIC_COLUMNS
            if col in best_models_df.columns
        ]

        labels = [
            f"{row['model']}\n{row['scenario']}"
            for _, row in best_models_df.iterrows()
        ]

        x = np.arange(len(metric_cols))
        width = 0.8 / len(best_models_df)

        plt.figure(figsize=(12, 6))

        for i, (_, row) in enumerate(best_models_df.iterrows()):
            values = row[metric_cols].astype(float).to_numpy()
            offset = (i - (len(best_models_df) - 1) / 2) * width

            plt.bar(
                x + offset,
                values,
                width,
                label=labels[i],
            )

        plt.xticks(x, metric_cols, rotation=30, ha="right")
        plt.ylim(0.0, 1.05)
        plt.ylabel("Score")
        plt.title("Best SVM and MLP models comparison")
        plt.legend()
        plt.grid(axis="y", alpha=0.3)

        self._save(filename)

    def plot_top_models(
        self,
        best_metric: str = "f1",
        filename: str = "top_models_by_f1.png",
        top_n: int = 10,
    ) -> None:
        df = self.metrics_df.copy()

        if best_metric not in df.columns:
            raise ValueError(f"Metric '{best_metric}' not found in metrics_df.")

        df = df.dropna(subset=[best_metric]).copy()

        df["model_label"] = (
            df["model"].astype(str)
            + " | "
            + df["scenario"].astype(str)
        )

        top_df = df.sort_values(
            by=[best_metric, "accuracy"],
            ascending=False,
        ).head(top_n)

        plt.figure(figsize=(12, 7))
        plt.barh(
            top_df["model_label"][::-1],
            top_df[best_metric][::-1],
        )

        plt.xlim(0.0, 1.05)
        plt.xlabel(best_metric)
        plt.title(f"Top {top_n} modelos por {best_metric}")
        plt.grid(axis="x", alpha=0.3)

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
        display.plot(ax=ax, values_format="d")

        ax.set_title(f"Confusion Matrix\n{model} | {scenario}")

        self._save(filename)

    def plot_roc_curves_for_best_models(
        self,
        best_models_df: pd.DataFrame,
        filename: str,
    ) -> None:
        plt.figure(figsize=(7, 6))

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
                print(f"[WARNING] y_score ausente para {scenario} | {model}")
                continue

            y_true = predictions_df["y_true"].to_numpy()
            y_score = predictions_df["y_score"].to_numpy()

            if len(np.unique(y_true)) < 2:
                continue

            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)

            plt.plot(
                fpr,
                tpr,
                label=f"{model} | {scenario} | AUC={roc_auc:.3f}",
            )

            plotted_any = True

        if not plotted_any:
            print("[WARNING] No ROC curve was created.")
            plt.close()
            return

        plt.plot([0, 1], [0, 1], linestyle="--", label="Random")

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC curve of best SVM and MLP models")
        plt.legend()
        plt.grid(alpha=0.3)

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
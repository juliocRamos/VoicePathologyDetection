from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd

from classes.plot.training_metrics_visualizer import (
    TrainingMetricsVisualizer,
)


class TrainingMetricsVisualizerTests(unittest.TestCase):
    def test_report_compares_svm_and_mlp_and_limits_ranking_to_five(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            predictions_dir = root / "predictions"
            figures_dir = root / "figures"
            predictions_dir.mkdir()

            metrics = pd.DataFrame([
                {
                    "scenario": "mfcc",
                    "model": "svm_rbf",
                    "accuracy": 0.80,
                    "balanced_accuracy": 0.78,
                    "f1": 0.77,
                    "auc": 0.84,
                },
                {
                    "scenario": "all_with_glottal",
                    "model": "mlp",
                    "accuracy": 0.76,
                    "balanced_accuracy": 0.75,
                    "f1": 0.74,
                    "auc": 0.81,
                },
            ])
            ranking = pd.DataFrame([
                {
                    "scenario": f"scenario_{index}",
                    "model": (
                        "svm_rbf"
                        if index % 2 == 0
                        else "mlp"
                    ),
                    "best_cv_score": 0.90 - index * 0.01,
                }
                for index in range(6)
            ])
            predictions = pd.DataFrame({
                "y_true": [0, 0, 1, 1],
                "y_pred": [0, 1, 1, 1],
                "y_score": [0.1, 0.4, 0.7, 0.9],
            })
            predictions.to_csv(
                predictions_dir
                / "mfcc_svm_rbf_predictions.csv",
                index=False,
            )
            predictions.to_csv(
                predictions_dir
                / "all_with_glottal_mlp_predictions.csv",
                index=False,
            )

            visualizer = TrainingMetricsVisualizer(
                metrics_df=metrics,
                predictions_dir=predictions_dir,
                output_dir=figures_dir,
                ranking_df=ranking,
            )

            with patch.object(
                visualizer,
                "plot_top_models",
                wraps=visualizer.plot_top_models,
            ) as plot_top_models:
                visualizer.generate_best_models_report(
                    best_metric="balanced_accuracy"
                )

            plot_top_models.assert_called_once_with(
                best_metric="balanced_accuracy",
                filename=(
                    "top_5_training_cv_pipelines_by_"
                    "balanced_accuracy.png"
                ),
                top_n=5,
            )
            self.assertEqual(
                len(visualizer._validate_family_champions()),
                2,
            )
            for filename in (
                "training_cv_family_champions_holdout_metrics.png",
                "training_cv_family_champions_holdout_roc.png",
                (
                    "top_5_training_cv_pipelines_by_"
                    "balanced_accuracy.png"
                ),
                "mfcc_svm_rbf_confusion_matrix.png",
                "all_with_glottal_mlp_confusion_matrix.png",
            ):
                self.assertTrue(
                    (figures_dir / filename).exists()
                )

    def test_report_rejects_multiple_holdout_rows_per_family(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            metrics = pd.DataFrame([
                {
                    "scenario": "mfcc",
                    "model": "svm_linear",
                },
                {
                    "scenario": "harmonics",
                    "model": "svm_rbf",
                },
            ])
            visualizer = TrainingMetricsVisualizer(
                metrics_df=metrics,
                predictions_dir=Path(directory) / "predictions",
                output_dir=Path(directory) / "figures",
            )

            with self.assertRaisesRegex(
                ValueError,
                "selected by training CV",
            ):
                visualizer._validate_family_champions()


if __name__ == "__main__":
    unittest.main()

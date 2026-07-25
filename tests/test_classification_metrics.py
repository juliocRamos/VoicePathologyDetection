import unittest

import numpy as np

from classes.experiment.training.classification_metrics import (
    ClassificationMetrics,
)


class ClassificationMetricsTests(unittest.TestCase):
    def test_grouped_bootstrap_resamples_complete_speakers(self) -> None:
        y_true = np.array([0, 0, 1, 1] * 5)
        y_pred = y_true.copy()
        groups = np.repeat(
            [f"speaker-{index}" for index in range(10)],
            2,
        )

        intervals = (
            ClassificationMetrics.compute_grouped_bootstrap_intervals(
                y_true=y_true,
                y_pred=y_pred,
                y_score=y_true.astype(float),
                groups=groups,
                iterations=50,
                confidence_level=0.95,
                random_state=42,
            )
        )

        self.assertEqual(
            intervals["balanced_accuracy_ci_lower"],
            1.0,
        )
        self.assertEqual(
            intervals["balanced_accuracy_ci_upper"],
            1.0,
        )
        self.assertGreater(
            intervals["bootstrap_valid_iterations"],
            0,
        )


if __name__ == "__main__":
    unittest.main()

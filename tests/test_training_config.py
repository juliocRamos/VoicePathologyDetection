import unittest

from classes.experiment.training.compute_backend import ComputeBackend
from classes.experiment.training.training_config import TrainingConfig


class TrainingConfigTests(unittest.TestCase):

    def test_cpu_allows_all_available_workers(self) -> None:
        config = TrainingConfig(
            compute_backend=ComputeBackend.CPU,
            n_jobs=-1,
        )

        self.assertEqual(config.n_jobs, -1)

    def test_cuda_requires_single_grid_search_worker(self) -> None:
        with self.assertRaisesRegex(ValueError, "n_jobs=1"):
            TrainingConfig(
                compute_backend=ComputeBackend.CUDA,
                n_jobs=-1,
            )

    def test_cuda_accepts_single_grid_search_worker(self) -> None:
        config = TrainingConfig(
            compute_backend=ComputeBackend.CUDA,
            n_jobs=1,
        )

        self.assertTrue(config.compute_backend.uses_cuda)

    def test_grid_search_verbose_cannot_be_negative(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "grid_search_verbose cannot be negative",
        ):
            TrainingConfig(grid_search_verbose=-1)

    def test_learning_curve_sizes_must_be_sorted(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique and sorted"):
            TrainingConfig(
                learning_curve_train_sizes=(0.5, 0.25, 1.0),
            )

    def test_nested_cv_requires_multiple_folds(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 2"):
            TrainingConfig(nested_cv_folds=1)

    def test_protocol_version_cannot_be_empty(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            TrainingConfig(protocol_version=" ")

    def test_cpu_is_not_eligible_for_final_reporting(self) -> None:
        with self.assertRaisesRegex(ValueError, "Only CUDA"):
            TrainingConfig(
                compute_backend=ComputeBackend.CPU,
                eligible_for_final_reporting=True,
            )


if __name__ == "__main__":
    unittest.main()

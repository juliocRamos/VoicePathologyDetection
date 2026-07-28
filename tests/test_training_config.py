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


if __name__ == "__main__":
    unittest.main()

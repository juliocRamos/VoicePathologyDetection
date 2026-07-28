import importlib.util
import unittest

from sklearn.feature_selection import SelectPercentile
from sklearn.model_selection import ParameterGrid
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from classes.experiment.training.compute_backend import ComputeBackend
from classes.experiment.training.training_plan import TrainingPlan


class TrainingPlanTests(unittest.TestCase):
    def test_fast_selectors_do_not_fit_nested_forests(self) -> None:
        selectors = TrainingPlan.fast_feature_selectors()

        self.assertTrue(
            all(
                isinstance(selector, SelectPercentile)
                for selector in selectors
            )
        )
        self.assertEqual(
            [
                selector.percentile
                for selector in selectors
            ],
            [10, 25, 50],
        )

    def test_cpu_plan_keeps_sklearn_mlp(self) -> None:
        specs = TrainingPlan.default_model_specs(
            compute_backend=ComputeBackend.CPU,
        )
        mlp_spec = next(
            spec
            for spec in specs
            if spec.name == "mlp"
        )

        self.assertIsInstance(
            mlp_spec.estimator,
            MLPClassifier,
        )
        self.assertTrue(mlp_spec.use_balanced_sample_weight)
        self.assertFalse(mlp_spec.estimator.early_stopping)
        self.assertEqual(mlp_spec.estimator.batch_size, 32)
        self.assertEqual(mlp_spec.estimator.tol, 0.0)

        grid = mlp_spec.param_grid[0]
        self.assertEqual(
            grid["classifier__hidden_layer_sizes"],
            [(8,), (16,)],
        )
        self.assertEqual(
            grid["classifier__max_iter"],
            [5, 10, 15, 20, 30],
        )
        self.assertEqual(
            grid["classifier__alpha"],
            [0.001, 0.01],
        )
        self.assertEqual(
            [selector.percentile for selector in grid["selector"]],
            [10, 25, 50],
        )

    @unittest.skipUnless(
        importlib.util.find_spec("torch")
        and importlib.util.find_spec("skorch"),
        "GPU training dependencies are not installed.",
    )
    def test_cuda_plan_uses_torch_mlp(self) -> None:
        from classes.experiment.training.torch_mlp_classifier import (
            BalancedTorchMLPClassifier,
        )

        specs = TrainingPlan.default_model_specs(
            compute_backend=ComputeBackend.CUDA,
        )
        mlp_spec = next(
            spec
            for spec in specs
            if spec.name == "mlp"
        )

        self.assertIsInstance(
            mlp_spec.estimator,
            BalancedTorchMLPClassifier,
        )
        self.assertEqual(mlp_spec.estimator.device, "cuda")
        self.assertFalse(mlp_spec.use_balanced_sample_weight)
        self.assertEqual(mlp_spec.estimator.batch_size, 32)

        self.assertEqual(
            {
                value
                for grid in mlp_spec.param_grid
                for value in grid[
                    "classifier__module__dropout"
                ]
            },
            {0.2, 0.4},
        )
        self.assertEqual(
            {
                value
                for grid in mlp_spec.param_grid
                for value in grid[
                    "classifier__optimizer__weight_decay"
                ]
            },
            {0.001, 0.01},
        )
        self.assertEqual(
            {
                value
                for grid in mlp_spec.param_grid
                for value in grid[
                    "classifier__criterion__label_smoothing"
                ]
            },
            {0.05, 0.10},
        )

        for grid in mlp_spec.param_grid:
            self.assertEqual(
                grid["classifier__max_epochs"],
                [5, 10, 15, 20],
            )
            self.assertEqual(
                [
                    selector.percentile
                    for selector in grid["selector"]
                ],
                [10, 25, 50],
            )

        moderate_grid, strong_grid = mlp_spec.param_grid
        self.assertEqual(
            moderate_grid[
                "classifier__module__hidden_layer_sizes"
            ],
            [(8,), (16,)],
        )
        self.assertEqual(
            strong_grid[
                "classifier__module__hidden_layer_sizes"
            ],
            [(8,), (16,), (16, 8)],
        )

    def test_optimized_plan_reduces_candidate_count(self) -> None:
        cpu_specs = TrainingPlan.default_model_specs(
            compute_backend=ComputeBackend.CPU,
        )

        cpu_candidate_count = sum(
            len(ParameterGrid(spec.param_grid))
            for spec in cpu_specs
        )

        self.assertEqual(cpu_candidate_count, 120)

        if (
            importlib.util.find_spec("torch")
            and importlib.util.find_spec("skorch")
        ):
            cuda_specs = TrainingPlan.default_model_specs(
                compute_backend=ComputeBackend.CUDA,
            )
            cuda_candidate_count = sum(
                len(ParameterGrid(spec.param_grid))
                for spec in cuda_specs
            )

            self.assertEqual(cuda_candidate_count, 120)

    def test_svm_specs_are_bounded_and_always_scaled(self) -> None:
        specs = TrainingPlan.default_model_specs(
            compute_backend=ComputeBackend.CPU,
        )
        svm_specs = [
            spec
            for spec in specs
            if spec.name.startswith("svm_")
        ]

        self.assertEqual(len(svm_specs), 2)

        for spec in svm_specs:
            self.assertEqual(spec.estimator.max_iter, 20_000)
            self.assertTrue(
                all(
                    isinstance(
                        scaler,
                        StandardScaler,
                    )
                    for scaler in spec.param_grid[0]["scaler"]
                )
            )

    @unittest.skipUnless(
        importlib.util.find_spec("torch")
        and importlib.util.find_spec("skorch"),
        "GPU training dependencies are not installed.",
    )
    def test_cuda_models_share_preprocessing_search_space(
        self,
    ) -> None:
        specs = TrainingPlan.default_model_specs(
            compute_backend=ComputeBackend.CUDA,
        )

        for spec in specs:
            for grid in spec.param_grid:
                self.assertEqual(len(grid["scaler"]), 1)
                self.assertIsInstance(
                    grid["scaler"][0],
                    StandardScaler,
                )
                self.assertEqual(
                    [
                        selector.percentile
                        for selector in grid["selector"]
                    ],
                    [10, 25, 50],
                )


if __name__ == "__main__":
    unittest.main()

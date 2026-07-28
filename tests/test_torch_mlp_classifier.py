import importlib.util
import unittest

import numpy as np


GPU_DEPENDENCIES_AVAILABLE = bool(
    importlib.util.find_spec("torch")
    and importlib.util.find_spec("skorch")
)


@unittest.skipUnless(
    GPU_DEPENDENCIES_AVAILABLE,
    "GPU training dependencies are not installed.",
)
class TorchMLPClassifierTests(unittest.TestCase):
    def test_fit_balances_each_binary_training_fold(self) -> None:
        import torch
        from torch import nn, optim
        from skorch.dataset import Dataset
        from skorch.helper import predefined_split

        from classes.experiment.training.torch_mlp_classifier import (
            BalancedTorchMLPClassifier,
            TorchMLPModule,
        )
        from classes.experiment.training.training_accuracy_callback import (
            TrainingAccuracyCallback,
        )

        features = np.arange(
            32,
            dtype=np.float64,
        ).reshape(8, 4)
        target = np.array(
            [0, 0, 0, 0, 0, 0, 1, 1],
            dtype=np.int64,
        )

        estimator = BalancedTorchMLPClassifier(
            module=TorchMLPModule,
            module__hidden_layer_sizes=(4,),
            criterion=nn.CrossEntropyLoss,
            criterion__label_smoothing=0.1,
            optimizer=optim.Adam,
            max_epochs=1,
            batch_size=-1,
            train_split=predefined_split(
                Dataset(
                    features[[0, 1, 6, 7]].astype(
                        np.float32
                    ),
                    target[[0, 1, 6, 7]],
                )
            ),
            callbacks=[
                (
                    "training_accuracy",
                    TrainingAccuracyCallback(),
                ),
            ],
            callbacks__valid_acc=None,
            verbose=0,
            device="cpu",
            random_state=42,
        )

        estimator.fit(features, target)

        np.testing.assert_allclose(
            estimator.criterion_.weight.detach().numpy(),
            np.array([2.0 / 3.0, 2.0]),
            rtol=1e-6,
        )
        self.assertEqual(
            estimator.criterion_.label_smoothing,
            0.1,
        )
        self.assertEqual(
            estimator.predict_proba(features).shape,
            (8, 2),
        )
        self.assertIsInstance(
            estimator.module_,
            torch.nn.Module,
        )
        self.assertIn(
            "train_accuracy",
            estimator.history[-1],
        )
        self.assertGreaterEqual(
            estimator.history[-1, "train_accuracy"],
            0.0,
        )
        self.assertLessEqual(
            estimator.history[-1, "train_accuracy"],
            1.0,
        )
        self.assertIn(
            "train_balanced_accuracy",
            estimator.history[-1],
        )
        self.assertIn(
            "valid_loss",
            estimator.history[-1],
        )
        self.assertIn(
            "valid_accuracy",
            estimator.history[-1],
        )
        self.assertIn(
            "valid_balanced_accuracy",
            estimator.history[-1],
        )


if __name__ == "__main__":
    unittest.main()

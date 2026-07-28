from __future__ import annotations

from typing import Any

import numpy as np
import torch

from skorch import NeuralNetClassifier
from torch import nn

from classes.experiment.training.host_array_converter import (
    to_numpy_array,
)


class TorchMLPModule(nn.Module):
    def __init__(
        self,
        hidden_layer_sizes: tuple[int, ...] = (64,),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if not hidden_layer_sizes:
            raise ValueError(
                "hidden_layer_sizes must contain at least one layer."
            )

        if any(size <= 0 for size in hidden_layer_sizes):
            raise ValueError(
                "hidden layer sizes must be positive."
            )

        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")

        layers: list[nn.Module] = []

        layers.extend([
            nn.LazyLinear(hidden_layer_sizes[0]),
            nn.ReLU(),
        ])

        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        for input_size, output_size in zip(
            hidden_layer_sizes,
            hidden_layer_sizes[1:],
        ):
            layers.extend([
                nn.Linear(input_size, output_size),
                nn.ReLU(),
            ])

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_layer_sizes[-1], 2))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


class BalancedTorchMLPClassifier(NeuralNetClassifier):
    def __init__(
        self,
        module: type[nn.Module],
        *args: Any,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        self.random_state = random_state
        super().__init__(module, *args, **kwargs)

    def fit(
        self,
        X: Any,
        y: Any,
        **fit_params: Any,
    ) -> BalancedTorchMLPClassifier:
        features = self._prepare_features(X)
        target = np.asarray(y, dtype=np.int64)

        classes, counts = np.unique(
            target,
            return_counts=True,
        )

        if not np.array_equal(classes, np.array([0, 1])):
            raise ValueError(
                "BalancedTorchMLPClassifier requires binary targets "
                "encoded as 0 and 1."
            )

        class_weights = (
            len(target)
            / (len(classes) * counts.astype(np.float64))
        )

        self.set_params(
            criterion__weight=torch.as_tensor(
                class_weights,
                dtype=torch.float32,
            )
        )

        torch.manual_seed(self.random_state)
        torch.cuda.manual_seed_all(self.random_state)

        return super().fit(
            features,
            target,
            **fit_params,
        )

    def predict(self, X: Any) -> np.ndarray:
        return super().predict(self._prepare_features(X))

    def predict_proba(self, X: Any) -> np.ndarray:
        return super().predict_proba(
            self._prepare_features(X)
        )

    @staticmethod
    def _prepare_features(X: Any) -> np.ndarray:
        return np.ascontiguousarray(
            to_numpy_array(X),
            dtype=np.float32,
        )

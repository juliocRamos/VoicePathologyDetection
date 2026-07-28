from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


def to_numpy_array(X: Any) -> np.ndarray:
    """Explicitly transfer CUDA arrays to host-backed NumPy arrays."""
    if hasattr(X, "get"):
        X = X.get()
    elif hasattr(X, "to_numpy"):
        X = X.to_numpy()

    return np.asarray(X)


class HostArrayConverter(
    TransformerMixin,
    BaseEstimator,
):
    """Establish the GPU-to-CPU boundary inside an sklearn pipeline."""

    def fit(
        self,
        X: Any,
        y: Any = None,
    ) -> HostArrayConverter:
        if hasattr(X, "shape") and len(X.shape) >= 2:
            self.n_features_in_ = X.shape[1]

        return self

    def transform(self, X: Any) -> np.ndarray:
        return to_numpy_array(X)

import importlib.util
import unittest

import numpy as np

from classes.experiment.training.host_array_converter import (
    HostArrayConverter,
    to_numpy_array,
)


def cupy_runtime_available() -> bool:
    if importlib.util.find_spec("cupy") is None:
        return False

    try:
        import cupy as cp

        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


class HostArrayConverterTests(unittest.TestCase):
    def test_numpy_array_remains_host_backed(self) -> None:
        features = np.arange(12).reshape(3, 4)

        converter = HostArrayConverter()
        converted = converter.fit_transform(features)

        self.assertIsInstance(converted, np.ndarray)
        self.assertEqual(converter.n_features_in_, 4)
        np.testing.assert_array_equal(converted, features)

    @unittest.skipUnless(
        cupy_runtime_available(),
        "A working CuPy CUDA runtime is not available.",
    )
    def test_cupy_array_is_explicitly_transferred_to_host(self) -> None:
        import cupy as cp

        features = cp.arange(12).reshape(3, 4)
        converted = to_numpy_array(features)

        self.assertIsInstance(converted, np.ndarray)
        np.testing.assert_array_equal(
            converted,
            np.arange(12).reshape(3, 4),
        )


if __name__ == "__main__":
    unittest.main()

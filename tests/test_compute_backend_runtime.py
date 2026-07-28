import sys
import unittest
from types import ModuleType
from unittest.mock import Mock, patch

from classes.experiment.training.compute_backend import ComputeBackend
from classes.experiment.training.compute_backend_runtime import (
    configure_torch_determinism,
    ensure_compute_backend_ready,
)


class ComputeBackendRuntimeTests(unittest.TestCase):
    @staticmethod
    def _fake_accel_module(enabled: bool) -> ModuleType:
        module = ModuleType("cuml.accel")
        module.enabled = lambda: enabled
        return module

    def test_cpu_rejects_active_cuda_accelerator(self) -> None:
        fake_accel = self._fake_accel_module(enabled=True)

        with patch.dict(
            sys.modules,
            {"cuml.accel": fake_accel},
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "CPU backend",
            ):
                ensure_compute_backend_ready(
                    ComputeBackend.CPU
                )

    def test_cuda_rejects_inactive_accelerator(self) -> None:
        fake_accel = self._fake_accel_module(enabled=False)

        with patch.dict(
            sys.modules,
            {"cuml.accel": fake_accel},
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "without activating",
            ):
                ensure_compute_backend_ready(
                    ComputeBackend.CUDA
                )

    def test_torch_determinism_disables_cudnn_benchmark(
        self,
    ) -> None:
        torch_module = Mock()
        torch_module.backends.cudnn.benchmark = True
        torch_module.backends.cudnn.deterministic = False

        configure_torch_determinism(torch_module)

        torch_module.use_deterministic_algorithms.assert_called_once_with(
            True,
            warn_only=True,
        )
        self.assertFalse(
            torch_module.backends.cudnn.benchmark
        )
        self.assertTrue(
            torch_module.backends.cudnn.deterministic
        )


if __name__ == "__main__":
    unittest.main()

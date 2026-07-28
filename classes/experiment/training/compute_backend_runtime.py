from __future__ import annotations

import sys

from classes.experiment.training.compute_backend import ComputeBackend


def activate_compute_backend(
    backend: ComputeBackend,
) -> None:
    if not backend.uses_cuda:
        return

    if "sklearn" in sys.modules:
        raise RuntimeError(
            "CUDA must be activated before importing scikit-learn."
        )

    try:
        from cuml.accel import enabled, install
    except ImportError as exc:
        raise RuntimeError(
            "CUDA backend requires the optional GPU dependencies. "
            "Install them with requirements-gpu.txt."
        ) from exc

    install()

    if not enabled():
        raise RuntimeError(
            "cuML was imported, but its accelerator was not enabled."
        )

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "CUDA MLP requires torch. Install requirements-gpu.txt."
        ) from exc

    if not torch.cuda.is_available():
        raise RuntimeError(
            "PyTorch could not access a CUDA device."
        )


def ensure_compute_backend_ready(
    backend: ComputeBackend,
) -> None:
    if not backend.uses_cuda:
        accel_module = sys.modules.get("cuml.accel")

        if (
            accel_module is not None
            and accel_module.enabled()
        ):
            raise RuntimeError(
                "CPU backend was requested, but cuml.accel is active."
            )

        return

    try:
        from cuml.accel import enabled
    except ImportError as exc:
        raise RuntimeError(
            "CUDA backend requires the optional GPU dependencies. "
            "Install them with requirements-gpu.txt."
        ) from exc

    if not enabled():
        raise RuntimeError(
            "CUDA backend was requested without activating "
            "cuml.accel before importing scikit-learn."
        )

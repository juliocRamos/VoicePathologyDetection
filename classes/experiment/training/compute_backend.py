from enum import Enum


class ComputeBackend(str, Enum):
    CPU = "cpu"
    CUDA = "cuda"

    @property
    def uses_cuda(self) -> bool:
        return self is ComputeBackend.CUDA

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


class FusionBackend:
    def __init__(self):
        self.backend = None
        try:
            import fusion_math
            self.backend = fusion_math
        except ImportError:
            print("Fusion backend not found. Please install the fusion package.")

    def add(self, v1: "np.ndarray", v2: "np.ndarray") -> "np.ndarray":
        return self.backend.add(v1, v2)

    def subtract(self, v1: "np.ndarray", v2: "np.ndarray") -> "np.ndarray":
        return self.backend.subtract(v1, v2)

    def multiply(self, v1: "np.ndarray", v2: "np.ndarray") -> "np.ndarray":
        return self.backend.multiply(v1, v2)

    def divide(self, v1: "np.ndarray", v2: "np.ndarray") -> "np.ndarray":
        return self.backend.divide(v1, v2)

    @staticmethod
    def matmul(self, v1: "np.ndarray", v2: "np.ndarray") -> "np.ndarray":
        return self.backend.matmul(v1, v2)

    def sum(self, v1: "np.ndarray") -> "np.ndarray":
        return self.backend.sum(v1)

    def maximum(self, v1: "np.ndarray", v2: "np.ndarray") -> "np.ndarray":
        return self.backend.maximum(v1, v2)

    def transpose(self, v1: "np.ndarray") -> "np.ndarray":
        return self.backend.transpose(v1)

    def power(self, v1: "np.ndarray", v2: "np.ndarray") -> "np.ndarray":
        return self.backend.power(v1, v2)

    def exp(self, v1: "np.ndarray") -> "np.ndarray":
        return self.backend.exp(v1)

    def log(self, v1: "np.ndarray") -> "np.ndarray":
        return self.backend.log(v1)

    def sqrt(self, v1: "np.ndarray") -> "np.ndarray":
        return self.backend.sqrt(v1)

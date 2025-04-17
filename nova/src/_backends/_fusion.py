import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import fusion_math
    import numpy as np

sys.path.append(
    "/Users/tompope/Documents/Documents - Tom’s MacBook Air/toms_personal_devs/deep_learning/Nova/fusion/build"
)


class FusionBackend:
    def __init__(self):
        self.backend = None
        try:
            import fusion_math

            self.backend = fusion_math
        except ImportError:
            print("Fusion backend not found. Please install the fusion package.")

    def _convert_numpy_to_tensor(self, array: "np.ndarray") -> "fusion_math.Tensor":
        """Convert numpy arrays to fusion tensors."""
        shape = list(array.shape)
        if len(shape) == 1:
            shape = [shape[0], 1]
        data = list(array.flatten().astype(float))
        fusion_tensor = self.backend.Tensor(shape, data)
        return fusion_tensor

    def _convert_scalar_to_tensor(self, scalar: "np.float32") -> "fusion_math.Tensor":
        """Convert numpy arrays to fusion tensors."""
        fusion_tensor = self.backend.Tensor(scalar)
        return fusion_tensor

    @staticmethod
    def _convert_tensor_to_numpy(tensor: "fusion_math.Tensor") -> "np.ndarray":
        return (
            tensor.to_numpy()
        )  # TODO: this is causing shape mismatch issues - squeezing in tests for now

    def add(self, v1: "np.ndarray", v2: "np.ndarray") -> "np.ndarray":
        tensor1 = self._convert_numpy_to_tensor(v1)
        tensor2 = self._convert_numpy_to_tensor(v2)
        tensor3 = tensor1 + tensor2
        return self._convert_tensor_to_numpy(tensor3)

    def subtract(self, v1: "np.ndarray", v2: "np.ndarray") -> "np.ndarray":
        tensor1 = self._convert_numpy_to_tensor(v1)
        tensor2 = self._convert_numpy_to_tensor(v2)
        tensor3 = tensor1 - tensor2
        return self._convert_tensor_to_numpy(tensor3)

    def multiply(self, v1: "np.ndarray", v2: "np.ndarray") -> "np.ndarray":
        tensor1 = self._convert_numpy_to_tensor(v1)
        tensor2 = self._convert_numpy_to_tensor(v2)
        tensor3 = tensor1 * tensor2
        return self._convert_tensor_to_numpy(tensor3)

    def divide(self, v1: "np.ndarray", v2: "np.ndarray") -> "np.ndarray":
        tensor1 = self._convert_numpy_to_tensor(v1)
        tensor2 = self._convert_numpy_to_tensor(v2)
        tensor3 = tensor1 / tensor2
        return self._convert_tensor_to_numpy(tensor3)

    def matmul(self, v1: "np.ndarray", v2: "np.ndarray") -> "np.ndarray":
        tensor1 = self._convert_numpy_to_tensor(v1)
        tensor2 = self._convert_numpy_to_tensor(v2)
        tensor3 = tensor1 @ tensor2
        return self._convert_tensor_to_numpy(tensor3)

    def sum(self, v1: "np.ndarray") -> "np.ndarray":
        tensor1 = self._convert_numpy_to_tensor(v1)
        tensor2 = tensor1.sum()
        return self._convert_tensor_to_numpy(tensor2)

    def maximum(self, v1: "np.ndarray", scalar: "np.float32") -> "np.ndarray":
        tensor1 = self._convert_numpy_to_tensor(v1)
        tensor2 = self._convert_scalar_to_tensor(scalar)
        tensor3 = tensor1.maximum(tensor2)
        return self._convert_tensor_to_numpy(tensor3)

    def transpose(self, v1: "np.ndarray") -> "np.ndarray":
        tensor1 = self._convert_numpy_to_tensor(v1)
        tensor2 = tensor1.transpose()
        return self._convert_tensor_to_numpy(tensor2)

    def power(self, v1: "np.ndarray", v2: "np.ndarray") -> "np.ndarray":
        tensor1 = self._convert_numpy_to_tensor(v1)
        tensor2 = self._convert_numpy_to_tensor(v2)
        tensor3 = tensor1.pow(tensor2)
        return self._convert_tensor_to_numpy(tensor3)

    def exp(self, v1: "np.ndarray") -> "np.ndarray":
        tensor1 = self._convert_numpy_to_tensor(v1)
        tensor2 = tensor1.exp()
        return self._convert_tensor_to_numpy(tensor2)

    def log(self, v1: "np.ndarray") -> "np.ndarray":
        tensor1 = self._convert_numpy_to_tensor(v1)
        tensor2 = tensor1.log()
        return self._convert_tensor_to_numpy(tensor2)

    def sqrt(self, v1: "np.ndarray") -> "np.ndarray":
        tensor1 = self._convert_numpy_to_tensor(v1)
        tensor2 = tensor1.sqrt()
        return self._convert_tensor_to_numpy(tensor2)

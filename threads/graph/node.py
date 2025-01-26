import numpy as np
from typing import Optional

from threads.operations.operation import Operation


class Node:
    def __init__(
        self,
        value: np.ndarray,
        operation: Optional[Operation],
        parents: tuple = (),
        requires_grad: bool = False,
    ):
        self.value = value
        self.operation = operation
        self.parents = parents
        self.requires_grad = requires_grad
        self.grad = None
        self.name = None

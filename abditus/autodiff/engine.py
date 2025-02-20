from typing import Optional, Tuple

import numpy as np

from abditus.autodiff._node import GraphNode
from abditus.operations.operation import Operation


class Engine:
    def __init__(self):
        pass

    def build_node(
        self,
        data: np.ndarray,
        operation: Optional[Operation] = None,
        parents: Tuple["GraphNode", ...] = (),
        requires_grad: bool = False,
    ):

        return GraphNode(
            value=data,
            operation=operation,
            parents=parents,
            requires_grad=requires_grad,
        )

    def build_leaf_node(self, data, requires_grad):
        return self.build_node(
            data=data, operation=None, parents=(), requires_grad=requires_grad
        )

    def __enter__(self):
        return self

    def __del__(self):
        print("Engine deleted")

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def current(self):
        return self


# if __name__ == "__main__":
#     import time
#     with Engine() as engine:
#         print(engine)
#         time.sleep(5)
#         print(engine.current())

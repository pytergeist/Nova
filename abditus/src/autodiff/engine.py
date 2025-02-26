from typing import Optional, Tuple

import numpy as np

from abditus.src.autodiff._node import Node
from abditus.src.operations.operation import Operation


class Engine:
    def __init__(self) -> None:
        self.node_idx_counter = 0  # TODO: This is in here for dev/debug purposes
        self.created_nodes = (
            []
        )  # TODO: Why node_idx starting at 8 in the print_graph function

    def _update_node_state(self) -> None:
        self.node_idx_counter += 1

    def _set_node_idx(self, node: Node) -> None:
        node.idx = self.node_idx_counter

    def _add_created_node(self, node: Node) -> None:
        self.created_nodes.append(node)

    def build_node(
        self,
        data: np.ndarray,
        operation: Optional[Operation] = None,
        parents: Tuple["Node", ...] = (),
        requires_grad: bool = False,
    ) -> "Node":
        self._update_node_state()
        node = Node(
            value=data,
            operation=operation,
            parents=parents,
            requires_grad=requires_grad,
        )
        self._set_node_idx(node)
        self._add_created_node(node)
        return node

    def build_leaf_node(self, data, requires_grad) -> "Node":
        return self.build_node(
            data=data, operation=None, parents=(), requires_grad=requires_grad
        )

    def __enter__(self) -> "Engine":
        return self

    def __del__(self) -> None:
        print("Engine deleted")

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        return False

    def current(self) -> "Engine":
        return self


# if __name__ == "__main__":
#     import time
#     with Engine() as engine:
#         print(engine)
#         time.sleep(5)
#         print(engine.current())

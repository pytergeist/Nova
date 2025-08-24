import threading
from typing import TYPE_CHECKING, Type

from nova.src.backend.graph import TopologicalSort

if TYPE_CHECKING:
    from nova.src.blocks.block import Block
    from nova.src.blocks.core.input_block import InputBlock

from .node import ModelNode


class Builder:
    _tls = threading.local()

    def __init__(self, sorter_cls: Type[TopologicalSort] = TopologicalSort):
        self.created_model_nodes = []
        self.node_idx_counter = 0
        self.sorter_cls = sorter_cls

    def _update_model_node_idx(self) -> None:
        """Updates the node index counter."""
        self.node_idx_counter += 1

    def _set_model_node_idx(self, model_node: ModelNode) -> None:
        """Sets the node index attribute on the node. Used in the build_node method to
        set the node index on the node instance.

        Args:
            node (Node): The node to set the index on.
        """
        setattr(model_node, "idx", self.node_idx_counter)

    def _add_created_model_node(self, model_node: ModelNode) -> None:
        """Adds the created node to the list of created nodes."""
        self.created_model_nodes.append(model_node)

    def build_model_node(
        self, operator: "Block", parents=(), inbound_tensors=None, outbound_tensors=None
    ) -> ModelNode:
        self._update_model_node_idx()
        node = ModelNode(
            operator=operator,
            parents=parents,
            inbound_tensors=inbound_tensors,
            outbound_tensors=outbound_tensors,
        )
        self._set_model_node_idx(node)
        self._add_created_model_node(node)
        return node

    def build_leaf_model_node(
        self,
        operator: "InputBlock",
        parents=(),
        inbound_tensors=None,
        outbound_tensors=None,
    ) -> ModelNode:
        return ModelNode(
            operator=operator,
            parents=parents,
            inbound_tensors=inbound_tensors,
            outbound_tensors=outbound_tensors,
        )

    def sort_model_graph(self):
        sorter = self.sorter_cls()
        start_node = self.created_model_nodes[-1]
        sorted_nodes = sorter.sort(start_node, mode="iterative", reverse=True)
        return sorted_nodes

    def __enter__(self) -> None: ...

    def __del__(self) -> None: ...

    def __exit__(self, exc_type, exc_value, traceback) -> bool: ...

    @classmethod
    def get_current(cls):
        curr = getattr(cls._tls, "current", None)
        if curr is None:
            raise RuntimeError("No Builder context is currently active.")
        return curr

    @classmethod
    def ensure_current(cls):
        curr = getattr(cls._tls, "current", None)
        if curr is None:
            curr = cls()
            cls._tls.current = curr
            curr.__enter__()
            curr._implicit = True
        return curr

    @classmethod
    def finalise_current(cls):
        curr = getattr(cls._tls, "current", None)
        if curr is not None and getattr(curr, "_implicit", False):
            curr.__exit__(None, None, None)
            cls._tls.current = True

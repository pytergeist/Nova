from typing import TYPE_CHECKING, Optional, Type

from nova.src.backend.graph import TopologicalSort

if TYPE_CHECKING:
    from nova.src.blocks.block import Block
    from nova.src.blocks.core.input_block import InputBlock

from .node import ModelNode


class Builder:
    _current: Optional["Builder"] = None

    def __init__(self, sorter_cls: Type[TopologicalSort] = TopologicalSort):
        self.created_model_nodes = []
        self.node_idx_counter = 0
        self.sorter_cls = sorter_cls

    # def __new__(cls, *args, **kwargs):  # TODO: Change to context manager pattern
    #     if cls._instance is None:
    #         cls._instance = super(Builder, cls).__new__(cls)
    #     return cls._instance

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

    def __enter__(self) -> "Builder":
        """Enter method for context manager pattern."""
        type(self)._current = self
        return self

    def __del__(self) -> None:
        """Destructor method for the Engine."""
        # This is a placeholder for any cleanup logic if needed.
        # Currently, it does nothing but will be extended in the future.
        ...

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        """Exit method for context manager pattern."""
        type(self)._current = None
        return False

    @classmethod
    def get_current(cls) -> "Builder":
        """Returns the current engine instance.

        Designed to be used with the context manager pattern, e.g. with Engine.current()
        as engine: similar to with Gradient.tape() as tape: in TensorFlow.
        """
        if cls._current is None:
            raise RuntimeError(
                "No active Builder context; must be inside `with Builder():`"
            )
        return cls._current

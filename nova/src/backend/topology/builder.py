from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nova.src.blocks.block import Block

from .node import ModelNode


class Builder:

    def __init__(self):
        self.created_model_nodes = []
        self.node_idx_counter = 0

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
        self, operator: "Block", inbound_tensors=None, outbound_tensors=None
    ) -> ModelNode:
        self._update_model_node_idx()
        node = ModelNode(
            operator=operator,
            inbound_tensors=inbound_tensors,
            outbound_tensors=outbound_tensors,
        )
        self._set_model_node_idx(node)
        self._add_created_model_node(node)
        return node

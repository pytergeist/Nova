# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2025 Tom Pope
#
# Nova — a high-performance hybrid physics and deep learning tensor engine.

from typing import TYPE_CHECKING, List, Optional, Tuple, Union

from nova.src.backend.core import Tensor

if TYPE_CHECKING:
    from nova.src.blocks.block import Block
    from nova.src.blocks.core.input_block import InputBlock


class ModelNode:

    def __init__(
        self,
        operator: Union["Block", "InputBlock"],
        parents: Tuple["ModelNode", ...] = (),
        inbound_tensors: Optional[List[Tensor] | Tensor] = None,
        outbound_tensors: Optional[List[Tensor] | Tensor] = None,
    ) -> None:
        self.operator = operator
        self.parents = parents
        self.children: List["ModelNode"] = []
        self.inbound_tensors = inbound_tensors
        self.outbound_tensors = outbound_tensors

    def set_children(self) -> None:
        """Set the children of this node."""
        for parent in self.parents:
            parent.children.append(self)

    def set_inbound_tensor(self, tensor: "Tensor") -> None:
        """Set an inbound tensor for this node."""
        self.inbound_tensors.append(tensor)

    def set_outbound_tensor(self, tensor: "Tensor") -> None:
        """Set an outbound tensor for this node."""
        self.outbound_tensors.append(tensor)

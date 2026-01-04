# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2025 Tom Pope
#
# Nova — a high-performance hybrid physics and deep learning tensor engine.

import uuid

import numpy as np

from nova.src.backend.topology import Builder


class InputBlock:
    def __init__(self, input_shape, dtype=np.float32):
        self.trainable = False
        self._inheritance_lock = False
        self.input_shape = input_shape
        self.output_shape = None
        self.dtype = dtype
        self._built = False
        self.builder = None
        self._node = None
        self.input_block = True
        self._uuid = uuid.uuid4()
        self._ensure_attached()

    @property
    def uuid(self) -> uuid.UUID:
        return self._uuid

    @property
    def node(self):
        self._ensure_attached()
        return self._node

    @property
    def built(self) -> bool:
        """Check if the block has been built."""
        return self._built

    @built.setter
    def built(self, value: bool) -> None:
        """Set the built property for the block."""
        if not isinstance(value, bool):
            raise ValueError("Built property must be a boolean value.")
        self._built = value

    def build_block(self, input_shape):
        self.output_shape = input_shape
        self.built = True

    def _ensure_attached(self):
        if self.builder is None:
            self.builder = Builder.ensure_current()
        if self._node is None:
            self._node = self.builder.build_leaf_model_node(
                self, parents=(), inbound_tensors=None, outbound_tensors=None
            )

    def get_config(self):
        return {"input_shape": self.input_shape, "dtype": str(self.dtype)}

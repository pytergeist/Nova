# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2025 Tom Pope
#
# Nova — a high-performance hybrid physics and deep learning tensor engine.

from nova.src.backend.random import Random
from nova.src.blocks.block import Block

random = Random()


class Dropout(Block):
    def __init__(self, rate):
        super(Dropout, self).__init__()
        self.rate = rate
        self._inheritance_lock = (
            False  # TODO: override super in parent to set inheritance lock
        )

    def call(self, inputs, training=True):
        if training:
            uniform_dist = random.uniform(
                inputs.shape, min=0.0, max=1.0
            )  # This is not a python tensor
            dropout_mask = uniform_dist >= self.rate
            return inputs * dropout_mask / (1 - self.rate)
        return inputs

    def get_config(self):
        return {"rate": self.rate}

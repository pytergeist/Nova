# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2025 Tom Pope
#
# Nova — a high-performance hybrid physics and deep learning tensor engine.

import numpy as np

from .operation import Operation


def relu_forward(inputs: np.ndarray) -> np.ndarray:
    return np.maximum(0, inputs)


def relu_backward(result_node, x_node, grad_output):
    return ((x_node.value > 0) * grad_output,)


relu_op = Operation("relu", relu_forward, relu_backward)

# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2025 Tom Pope
#
# Nova — a high-performance hybrid physics and deep learning tensor engine.

# grad_registry.py
import numpy as np

from .operation import Operation


def greater_than_equal_to_backward(
    result, a, b, grad_output
):  # TODO: check this for accuracy
    zeros_a = np.zeros_like(a)
    zeros_b = np.zeros_like(b)
    return zeros_a, zeros_b


greater_than_equal_to_op = Operation(
    "__ge__", lambda a, b: a >= b, greater_than_equal_to_backward
)

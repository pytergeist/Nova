# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2025 Tom Pope
#
# Nova — a high-performance hybrid physics and deep learning tensor engine.

import numpy as np
import pytest


@pytest.fixture
def binary_fn_numpy():
    def _fn_numpy(x, fn_str):
        fn = getattr(np, fn_str)
        return fn(x, x)

    return _fn_numpy


@pytest.fixture
def unary_fn_numpy():
    def _fn_numpy(x, fn_str):
        fn = getattr(np, fn_str)
        return fn(x)

    return _fn_numpy

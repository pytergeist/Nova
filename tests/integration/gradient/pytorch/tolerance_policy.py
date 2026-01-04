# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2025 Tom Pope
#
# Nova — a high-performance hybrid physics and deep learning tensor engine.

# This file defines tolerances for numerical gradient checks in tests.
# These tolerances are used to compare the analytical gradients with numerical
# gradients computed using pytorch.
# Currently, these tolerances are based on float32 dtype flowing through the system

from enum import Enum


class Tolerance(Enum):
    RTOL = 1e-2
    ATOL = 1e-4

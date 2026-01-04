# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2025 Tom Pope
#
# Nova — a high-performance hybrid physics and deep learning tensor engine.

import functools

from nova.src.backend.core import Grad


def set_grad(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with Grad():
            return func(*args, **kwargs)

    return wrapper

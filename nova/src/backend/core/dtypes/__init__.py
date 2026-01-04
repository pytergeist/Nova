# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2025 Tom Pope
#
# Nova — a high-performance hybrid physics and deep learning tensor engine.

from .dtypes import Bool as bool
from .dtypes import DType
from .dtypes import Float32 as float32
from .dtypes import Float64 as float64
from .dtypes import Int32 as int32
from .dtypes import Int64 as int64
from .dtypes import as_dtype

__all__ = ["as_dtype", "DType", "float32", "float64", "int32", "int64", "bool"]

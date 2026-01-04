# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2025 Tom Pope
#
# Nova — a high-performance hybrid physics and deep learning tensor engine.

from enum import Enum

import numpy as np

from nova.src.backend.core.clib import CppDType


class DType(Enum):
    FLOAT32 = 0
    FLOAT64 = 1
    INT32 = 2
    INT64 = 3
    BOOL = 4


class Float32:
    @staticmethod
    def cpp_type():
        return CppDType(DType.FLOAT32.value)


class Float64:
    @staticmethod
    def cpp_type():
        return CppDType(DType.FLOAT64.value)


class Int32:
    @staticmethod
    def cpp_type():
        return CppDType(DType.INT32.value)


class Int64:
    @staticmethod
    def cpp_type():
        return CppDType(DType.INT64.value)


class Bool:
    @staticmethod
    def cpp_type():
        return CppDType(DType.Bool.value)


def as_dtype(dtype: DType) -> np.dtype:
    """Converts a DType to a numpy dtype.

    Args:
        dtype (DType): The DType to convert.

    Returns:
        np.dtype: The numpy dtype.
    """
    if isinstance(dtype, DType):
        return dtype.value

    if isinstance(dtype, str):
        try:
            return np.dtype(dtype)
        except TypeError:
            raise ValueError(f"Unknown dtype string: {dtype}")
    try:
        return np.dtype(dtype)
    except TypeError:
        raise ValueError(f"Invalid dtype provided: {dtype}")

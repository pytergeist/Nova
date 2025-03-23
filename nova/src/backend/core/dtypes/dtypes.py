from enum import Enum

import numpy as np


class DType(Enum):
    FLOAT32 = np.float32
    FLOAT64 = np.float64
    INT32 = np.int32
    INT64 = np.int64
    BOOL = np.bool_


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

"""
factory functions for Tensor<f>
"""

from __future__ import annotations
import nova.src.backend.core.clib.fusion

__all__ = ["fill", "ones", "ones_like", "zeros", "zeros_like"]

def fill(shape: list[int], value: float, device: Device) -> ...:
    """
    Create a tensor filled with a given value
    """

def ones(shape: list[int], device: Device) -> ...:
    """
    Create a tensor of ones
    """

def ones_like(
    other: nova.src.backend.core.clib.fusion.Tensor,
) -> nova.src.backend.core.clib.fusion.Tensor: ...
def zeros(shape: list[int], device: Device) -> ...:
    """
    Create a tensor of zeros
    """

def zeros_like(
    other: nova.src.backend.core.clib.fusion.Tensor,
) -> nova.src.backend.core.clib.fusion.Tensor: ...

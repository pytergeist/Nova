"""
factory functions for Tensor<f>
"""

from __future__ import annotations
import nova.src.backend.core.clib.fusion

__all__ = ["fill", "ones", "ones_like", "zeros", "zeros_like"]

def fill(shape: list[int], value: float) -> nova.src.backend.core.clib.fusion.Tensor:
    """
    Create a tensor filled with a given value
    """

def ones(shape: list[int]) -> nova.src.backend.core.clib.fusion.Tensor:
    """
    Create a tensor of ones
    """

def ones_like(
    other: nova.src.backend.core.clib.fusion.Tensor,
) -> nova.src.backend.core.clib.fusion.Tensor:
    """
    Create a ones tensor with the same shape as another
    """

def zeros(shape: list[int]) -> nova.src.backend.core.clib.fusion.Tensor:
    """
    Create a tensor of zeros
    """

def zeros_like(
    other: nova.src.backend.core.clib.fusion.Tensor,
) -> nova.src.backend.core.clib.fusion.Tensor:
    """
    Create a zeros tensor with the same shape as another
    """

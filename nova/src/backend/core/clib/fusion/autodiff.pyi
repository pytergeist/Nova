"""
Autodiff control
"""

from __future__ import annotations
import typing

__all__ = ["enabled"]

def enabled(state: typing.Any = None) -> bool:
    """
    Get or set whether autodiff is enabled for this thread. When enabled, a default Engine is installed in the EngineContext.
    """

# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2025 Tom Pope
#
# Nova — a high-performance hybrid physics and deep learning tensor engine.

from typing import TYPE_CHECKING, Literal, Optional

from nova.src.backend.core import clib, io

if TYPE_CHECKING:
    from nova.src.backend.core import Tensor


class Random:

    def __init__(self, seed: Optional[int] = None):
        self._rng = clib.Random() if not seed else clib.Random(seed)
        self.seed = seed

    def uniform(
        self,
        shape: tuple[int, ...],
        min: float = 0.0,
        max: float = 1.0,
        device: Literal["CPU", "GPU", "CUDA", "METAL"] = "CPU",
    ) -> "Tensor":
        cpp_device = io._get_cpp_device(device)
        return io.as_tensor(
            self._rng.uniform_cpp(
                shape=shape, min=min, max=max, device=cpp_device
            ).to_numpy()
        )  # TODO: we're passing data in and out of c-layer to numpy with this

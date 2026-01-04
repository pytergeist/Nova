# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2025 Tom Pope
#
# Nova — a high-performance hybrid physics and deep learning tensor engine.

from enum import Enum

# NB: this file is a python shim for the device type/device inside the cpp core and
# must remain in sync with the cpp device layer, located at Nova/fusion/src/Fusion/device


class DeviceType(Enum):
    CPU = 0
    CUDA = 1
    METAL = 2


class Device:
    def __init__(self, device: DeviceType, index: int = -1):
        self.type = device
        self.index = index


def parse_device(spec: str | None) -> tuple[DeviceType, int]:
    if spec is None:
        return Device(DeviceType.CPU, 0)

    sp = spec.lower()
    if "cpu" in sp:
        return Device(DeviceType.CPU, 0)

    elif "cuda" in sp or "gpu" in sp:
        return Device(DeviceType.CUDA, 1)

    elif "metal" in sp:
        return Device(DeviceType.METAL, 2)

    else:
        raise ValueError("Unknown device type {spec}")

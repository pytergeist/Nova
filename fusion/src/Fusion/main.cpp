// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2025 Tom Pope
//
// Nova — a high-performance hybrid physics and deep learning tensor engine.

#include "Fusion/device/Device.h"
#include "Fusion/device/DeviceType.h"

int main() {
   Device device(DeviceType::CPU, 0);
   return 0;
}

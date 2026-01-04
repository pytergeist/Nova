// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2025 Tom Pope
//
// Nova — a high-performance hybrid physics and deep learning tensor engine.

#ifndef DEVICE_TYPE_HPP
#define DEVICE_TYPE_HPP

#include <cstdint>

using DeviceIdx = std::int8_t;

enum class DeviceType {
   CPU = 0,
   CUDA = 1,
   METAL = 2,
};

constexpr DeviceType kCPU = DeviceType::CPU;
constexpr DeviceType kCUDA = DeviceType::CUDA;
constexpr DeviceType kMETAL = DeviceType::METAL;

#endif // DEVICE_TYPE_HPP

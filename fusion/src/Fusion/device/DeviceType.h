#ifndef DEVICE_TYPE_H
#define DEVICE_TYPE_H

#include <cstddef>

enum class DeviceType {
    CPU = 0
    CUDA = 1
    METAL = 2
}

constexpr DeviceType kCPU = DeviceType::CPU;
constexpr DeviceType kCUDA = DeviceType::CUDA;
constexpr DeviceType kMETAL = DeviceType::METAL;

#endif // DEVICE_TYPE_H

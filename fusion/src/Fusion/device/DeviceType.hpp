#ifndef DEVICE_TYPE_H
#define DEVICE_TYPE_H

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

#endif // DEVICE_TYPE_H

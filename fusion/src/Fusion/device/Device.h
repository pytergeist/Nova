#ifndef DEVICE_HPP
#define DEVICE_HPP

#include <cstdint>

#include "DeviceType.h"
#include "Fusion/common/Checks.hpp"
#include "Fusion/common/Log.hpp"
#include "Fusion/common/error/Check.h"

struct Device final {
   explicit Device(const DeviceType type, const DeviceIdx index = -1)
       : type_(type), index_(index) {
      validate_device();
   }

   DeviceIdx idx() const noexcept { return index_; }
   DeviceType type() const noexcept { return type_; }

   bool operator==(const Device &other) const {
      return type_ == other.type_ && index_ == other.index_;
   }

   bool is_cpu() const { return type_ == DeviceType::CPU; }
   bool is_cuda() const { return type_ == DeviceType::CUDA; }
   bool is_meta() const { return type_ == DeviceType::METAL; }
   bool is_gpu() const {
      return type_ == DeviceType::CUDA || type_ == DeviceType::METAL;
   }

 private:
   DeviceType type_;
   DeviceIdx index_;

   void validate_device() const {
      switch (type_) {
      case DeviceType::CPU:
         FUSION_CHECK(index_ == 0, "CPU device must use index 0");
         break;
      case DeviceType::CUDA:
      case DeviceType::METAL:
         FUSION_CHECK(index_ >= 0, "Accelerator device index must be > 0");
         break;
      }
   }
};

#endif // DEVICE_HPP

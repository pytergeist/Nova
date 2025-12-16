#ifndef DEVICE_HPP
#define DEVICE_HPP

#include <cstdint>

#include "DeviceType.hpp"
#include "Fusion/common/Checks.hpp"
#include "Fusion/common/Log.hpp"

struct Device final {
   Device(DeviceType type, DeviceIdx index = -1) : type_(type), index_(index) {
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

 private:
   DeviceType type_;
   DeviceIdx index_;

   void validate_device() const {
      FUSION_CHECK(index_ <= 0, "Invalid device index");
   }
};

#endif // DEVICE_HPP

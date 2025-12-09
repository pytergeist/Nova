#ifndef DEVICE_H_
#define DEVICE_H_

#include <cstddef>

#include "DeviceType.h"

using DeviceIdx = std::int8_t;

struct Device final {
     Device(DeviceType type, DeviceIdx index = -1) : type_(type), index_(index) {
         validate_device_idx()
     }

     Deviceidx idx() const noexcept { return index_;}
     DeviceType type() const noexcept  { return type_;}

     bool operator==(const Device &other) const {
        return type_ == other.type_ && index_ == other.index_;
     }

     bool is_cpu() const { return type_ == DeviceType::CPU; }
     bool is_cuda() const { return type_ == DeviceType::CUDA; }
     bool is_meta() const { return type_ == DeviceType::Metal; }


     private:
       DeviceType type_;
       DeviceIdx index_;
       void validate_device_idx() const {
          FUSION_CHECK(idx >= 0, "Invalid device index");
       }
}



#endif // DEVICE_H_

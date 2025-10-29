#ifndef TENSOR_STORAGE_H
#define TENSOR_STORAGE_H

#include <cstddef>
#include <vector>
#include "TensorBuffer.h"
#include "../core/Device.h"

template <typename T> class ITensorStorage {
 public:
   virtual ~ITensorStorage() = default;

   // raw data access - this allows access to either
   // mutable or immutable raw data in child class
   virtual TensorBuffer &data() = 0;
   virtual const TensorBuffer &data() const = 0; // check this
   virtual T *data_ptr() = 0;
   virtual const T *data_ptr() const = 0;

   [[nodiscard]] virtual size_t size() const = 0;
   [[nodiscard]] virtual Device device() const = 0;
};

#endif // TENSOR_STORAGE_H

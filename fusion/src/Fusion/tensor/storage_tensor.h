#ifndef TENSOR_STORAGE_H
#define TENSOR_STORAGE_H

#include <cstddef>
#include <memory>

enum class Device {
  CPU,
  // GPU
};

template <typename T> class ITensorStorage {
public:
  virtual ~ITensorStorage() = default;

  // raw data access - this allows access to either
  // mutable or immutable raw data in child class
  virtual T *data() = 0;
  virtual const T *data() const = 0;

  // row/col info -- TODO: examine this how 3D Tensor (e.g. with batch size)
  // may need to create a batched tensor subclass?
  virtual size_t rows() const = 0;
  virtual size_t cols() const = 0;

  virtual Device device() const = 0;
};

#endif // TENSOR_STORAGE_H

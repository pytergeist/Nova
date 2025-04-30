#ifndef TENSOR_STORAGE_H
#define TENSOR_STORAGE_H

#include <cstddef>
#include <vector>

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

  virtual std::vector<size_t> shape() const = 0;
  virtual std::vector<size_t> strides() const = 0;

  virtual Device device() const = 0;
};

#endif // TENSOR_STORAGE_H

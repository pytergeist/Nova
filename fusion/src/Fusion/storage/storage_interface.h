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
  virtual std::vector<T> &data() = 0;
  virtual const std::vector<T> &data() const = 0; // check this

  [[nodiscard]] virtual std::vector<size_t> shape() const = 0;
  [[nodiscard]] virtual std::vector<size_t> strides() const = 0;
  [[nodiscard]] virtual size_t size() const = 0;
  [[nodiscard]] virtual size_t ndims() const = 0;
  [[nodiscard]] virtual Device device() const = 0;
};

#endif // TENSOR_STORAGE_H

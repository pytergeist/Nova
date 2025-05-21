#ifndef TENSOR_H
#define TENSOR_H
#include "kernels/cblas_api.cpp"
#include "kernels/serial_api.cpp"
#include "kernels/xsimd_api.cpp"
#include "storage/dense_storage.h"
#include "storage/storage_interface.h"
#include "xsimd/xsimd.hpp"
#include <iostream>
#include <stdexcept>

template <typename T> class Tensor {
public:
  std::unique_ptr<ITensorStorage<T>> storage;
  std::vector<size_t> shape_;
  size_t rank_;

  explicit Tensor(std::vector<size_t> shape, std::vector<T> data,
                  Device device = Device::CPU)
      : shape_(std::move(shape)) {
    if (device == Device::CPU) {
      // Use the stored shape_ (guaranteed non-empty if you passed one in)
      storage = std::make_unique<NDTensorStorage<T>>(shape_, std::move(data));
      rank_ = storage->ndims();
    } else {
      throw std::invalid_argument("Unsupported device type");
    }
  }

  friend std::ostream &operator<<(std::ostream &os, const Tensor<T> &tensor) {
    const auto *cpuStorage =
        dynamic_cast<const NDTensorStorage<T> *>(tensor.storage.get());
    if (cpuStorage) {
      const std::vector<T> &data = cpuStorage->data();
      size_t size = cpuStorage->size();
      os << "Tensor(";
      for (size_t i = 0; i < size; i++) {
        os << data[i];
        if (i + 1 < size)
          os << ", ";
      }
      os << ")" << std::endl;
    } else {
      os << "Tensor(unsupported storage type)";
    }
    return os;
  }

  std::vector<T> &raw_data() { return storage->data(); }
  const std::vector<T> &raw_data() const { return storage->data(); }
  [[nodiscard]] size_t flat_size() const { return storage->size(); }

  Tensor<T> operator+(const Tensor<T> &other) const {
    // 1) decide on broadcast shape & length
    std::vector<size_t> out_shape =
        (other.flat_size() == 1 ? this->shape_ : other.shape_);
    std::size_t out_size =
        (other.flat_size() == 1 ? this->flat_size() : other.flat_size());

    const T *a_ptr = this->storage->data_ptr();
    const T *b_ptr = other.storage->data_ptr();
    std::size_t na = this->flat_size();
    std::size_t nb = other.flat_size();

    std::vector<T> result(out_size);
    T *r_ptr = result.data();

    using arch = xsimd::default_arch;
    using tag = xsimd::unaligned_mode;
    xsimd_ops::add{}(arch{}, a_ptr, na, b_ptr, nb, r_ptr, out_size, tag{});

    return Tensor<T>(std::move(out_shape), std::move(result), Device::CPU);
  }

  Tensor<T> operator-(const Tensor<T> &other) const {
    // 1) decide on broadcast shape & length
    std::vector<size_t> out_shape =
        (other.flat_size() == 1 ? this->shape_ : other.shape_);
    std::size_t out_size =
        (other.flat_size() == 1 ? this->flat_size() : other.flat_size());

    const T *a_ptr = this->storage->data_ptr();
    const T *b_ptr = other.storage->data_ptr();
    std::size_t na = this->flat_size();
    std::size_t nb = other.flat_size();

    std::vector<T> result(out_size);
    T *r_ptr = result.data();

    using arch = xsimd::default_arch;
    using tag = xsimd::unaligned_mode;
    xsimd_ops::subtract{}(arch{}, a_ptr, na, b_ptr, nb, r_ptr, out_size, tag{});

    return Tensor<T>(std::move(out_shape), std::move(result), Device::CPU);
  }
  //
  //     // overload the - operator for -tensor
  //     Tensor<T> operator-() const;
  //
  //     // overload the / operator
  Tensor<T> operator/(const Tensor<T> &other) const {
    // 1) decide on broadcast shape & length
    std::vector<size_t> out_shape =
        (other.flat_size() == 1 ? this->shape_ : other.shape_);
    std::size_t out_size =
        (other.flat_size() == 1 ? this->flat_size() : other.flat_size());

    const T *a_ptr = this->storage->data_ptr();
    const T *b_ptr = other.storage->data_ptr();
    std::size_t na = this->flat_size();
    std::size_t nb = other.flat_size();

    std::vector<T> result(out_size);
    T *r_ptr = result.data();

    using arch = xsimd::default_arch;
    using tag = xsimd::unaligned_mode;
    xsimd_ops::divide{}(arch{}, a_ptr, na, b_ptr, nb, r_ptr, out_size, tag{});

    return Tensor<T>(std::move(out_shape), std::move(result), Device::CPU);
  }

  //
  //     // overload the + operator
  Tensor<T> operator*(const Tensor<T> &other) const {
    // 1) decide on broadcast shape & length
    std::vector<size_t> out_shape =
        (other.flat_size() == 1 ? this->shape_ : other.shape_);
    std::size_t out_size =
        (other.flat_size() == 1 ? this->flat_size() : other.flat_size());

    const T *a_ptr = this->storage->data_ptr();
    const T *b_ptr = other.storage->data_ptr();
    std::size_t na = this->flat_size();
    std::size_t nb = other.flat_size();

    std::vector<T> result(out_size);
    T *r_ptr = result.data();

    using arch = xsimd::default_arch;
    using tag = xsimd::unaligned_mode;
    xsimd_ops::multiply{}(arch{}, a_ptr, na, b_ptr, nb, r_ptr, out_size, tag{});

    return Tensor<T>(std::move(out_shape), std::move(result), Device::CPU);
  }

  Tensor<T> maximum(const Tensor<T> &other) const {
    // 1) decide on broadcast shape & length
    std::vector<size_t> out_shape =
        (other.flat_size() == 1 ? this->shape_ : other.shape_);
    std::size_t out_size =
        (other.flat_size() == 1 ? this->flat_size() : other.flat_size());

    const T *a_ptr = this->storage->data_ptr();
    const T *b_ptr = other.storage->data_ptr();
    std::size_t na = this->flat_size();
    std::size_t nb = other.flat_size();

    std::vector<T> result(out_size);
    T *r_ptr = result.data();

    using arch = xsimd::default_arch;
    using tag = xsimd::unaligned_mode;
    xsimd_ops::maximum{}(arch{}, a_ptr, na, b_ptr, nb, r_ptr, out_size, tag{});

    return Tensor<T>(std::move(out_shape), std::move(result), Device::CPU);
  }

  //
  Tensor<T> sqrt() {
    std::vector<size_t> shape = this->shape_;
    size_t size = this->flat_size();
    std::vector<T> data;
    data.resize(size);
    std::vector<T> v1 = this->raw_data();
    using arch = xsimd::default_arch; // dispatch to SSE/AVX/NEON as appropriate
    using tag = xsimd::unaligned_mode;
    xsimd_ops::sqrt{}(arch{}, v1, data, tag{});
    return Tensor<T>(shape, data, Device::CPU);
  };
  //
  Tensor<T> exp() {
    std::vector<size_t> shape = this->shape_;
    size_t size = this->flat_size();
    std::vector<T> data;
    data.resize(size);
    std::vector<T> v1 = this->raw_data();
    using arch = xsimd::default_arch; // dispatch to SSE/AVX/NEON as appropriate
    using tag = xsimd::unaligned_mode;
    xsimd_ops::exp{}(arch{}, v1, data, tag{});
    return Tensor<T>(shape, data, Device::CPU);
  };

  //
  Tensor<T> log() {
    std::vector<size_t> shape = this->shape_;
    size_t size = this->flat_size();
    std::vector<T> data;
    data.resize(size);
    std::vector<T> v1 = this->raw_data();
    using arch = xsimd::default_arch; // dispatch to SSE/AVX/NEON as appropriate
    using tag = xsimd::unaligned_mode;
    xsimd_ops::log{}(arch{}, v1, data, tag{});
    return Tensor<T>(shape, data, Device::CPU);
  };

  //
  // Tensor<T> pow(T exponent) const;

  //
  Tensor<T> pow(Tensor<T> &other) const {
    // 1) decide on broadcast shape & length
    std::vector<size_t> out_shape =
        (other.flat_size() == 1 ? this->shape_ : other.shape_);
    std::size_t out_size =
        (other.flat_size() == 1 ? this->flat_size() : other.flat_size());

    const T *a_ptr = this->storage->data_ptr();
    const T *b_ptr = other.storage->data_ptr();
    std::size_t na = this->flat_size();
    std::size_t nb = other.flat_size();

    std::vector<T> result(out_size);
    T *r_ptr = result.data();

    using arch = xsimd::default_arch;
    using tag = xsimd::unaligned_mode;
    xsimd_ops::pow{}(arch{}, a_ptr, na, b_ptr, nb, r_ptr, out_size, tag{});

    return Tensor<T>(std::move(out_shape), std::move(result), Device::CPU);
  }
  //
  Tensor<T> sum() {
    std::vector<T> data(1);          // reserve one slot
    std::vector<T> &in = raw_data(); // reference to your own storage

    using arch = xsimd::default_arch;
    xsimd_ops::sum{}(arch{}, in, data);

    // now `data[0]` holds the total
    return Tensor<T>({1}, std::move(data), Device::CPU);
  }

  //
  Tensor<T> matmul(Tensor<T> &other) {
    auto const &shapeA = this->shape_;
    auto const &shapeB = other.shape_;

    size_t m = shapeA[0], n = shapeB[1];
    std::vector<T> data(m * n);

    cblas_ops::matmul(this->raw_data(), shapeA, other.raw_data(), shapeB, data);

    return Tensor<T>({m, n}, std::move(data), Device::CPU);
  }
  //

  Tensor<T> transpose() {
    std::vector<size_t> new_shape(shape_.rbegin(), shape_.rend());

    size_t size = flat_size();
    std::vector<T> new_data(size);

    serial_ops::transpose(this->raw_data(), this->shape_, new_data);

    return Tensor<T>(std::move(new_shape), std::move(new_data), Device::CPU);
  }
};
//
//     Tensor<T> diagonal() const;
// };
#endif // TENSOR_H

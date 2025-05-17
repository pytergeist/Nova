#ifndef TENSOR_H
#define TENSOR_H
#include "../kernels/cblas_api.cpp"
#include "../kernels/serial_api.cpp"
#include "../kernels/xsimd_api.cpp"
#include "../storage/dense_storage.h"
#include "../storage/storage_interface.h"
#include "xsimd/xsimd.hpp"
#include <cblas.h>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <stdexcept>

template <typename T> class Tensor;

// template<typename T>
// std::ostream &operator<<(std::ostream &os, const Tensor<T> &tensor);

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

  std::vector<T> &raw_data()       { return storage->data(); }
  const std::vector<T> &raw_data() const { return storage->data(); }
  [[nodiscard]] size_t flat_size() const { return storage->size(); }

  // overload the + operator
  Tensor<T> operator+(Tensor<T> &other) {
    std::vector<size_t> shape = other.shape_;
    size_t size = other.flat_size();
    std::vector<T> data;
    data.resize(size);
    std::vector<T> v1 = this->raw_data();
    std::vector<T> v2 = other.raw_data();
    using arch = xsimd::default_arch; // dispatch to SSE/AVX/NEON as appropriate
    using tag = xsimd::unaligned_mode;
    xsimd_ops::add{}(arch{}, v1, v2, data, tag{});
    return Tensor<T>(shape, data, Device::CPU);
  }

  //     // overload the - operator for tensor - tensor
  Tensor<T> operator-(Tensor<T> &other) {
    std::vector<size_t> shape = other.shape_;
    size_t size = other.flat_size();
    std::vector<T> data;
    data.resize(size);
    std::vector<T> v1 = this->raw_data();
    std::vector<T> v2 = other.raw_data();
    using arch = xsimd::default_arch; // dispatch to SSE/AVX/NEON as appropriate
    using tag = xsimd::unaligned_mode;
    xsimd_ops::subtract{}(arch{}, v1, v2, data, tag{});
    return Tensor<T>(shape, data, Device::CPU);
  };
  //
  //     // overload the - operator for -tensor
  //     Tensor<T> operator-() const;
  //
  //     // overload the / operator
  Tensor<T> operator/(Tensor<T> &other) {
    std::vector<size_t> shape = other.shape_;
    size_t size = other.flat_size();
    std::vector<T> data;
    data.resize(size);
    std::vector<T> v1 = this->raw_data();
    std::vector<T> v2 = other.raw_data();
    using arch = xsimd::default_arch; // dispatch to SSE/AVX/NEON as appropriate
    using tag = xsimd::unaligned_mode;
    xsimd_ops::divide{}(arch{}, v1, v2, data, tag{});
    return Tensor<T>(shape, data, Device::CPU);
  }

  //
  //     // overload the + operator
  Tensor<T> operator*(Tensor<T> &other) {
    std::vector<size_t> shape = other.shape_;
    size_t size = other.flat_size();
    std::vector<T> data;
    data.resize(size);
    std::vector<T> v1 = this->raw_data();
    std::vector<T> v2 = other.raw_data();
    using arch = xsimd::default_arch; // dispatch to SSE/AVX/NEON as appropriate
    using tag = xsimd::unaligned_mode;
    xsimd_ops::multiply{}(arch{}, v1, v2, data, tag{});
    return Tensor<T>(shape, data, Device::CPU);
  };

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
  Tensor<T> pow(Tensor<T> &other) {
    std::vector<size_t> shape = other.shape_;
    size_t size = other.flat_size();
    std::vector<T> data;
    data.resize(size);
    std::vector<T> v1 = this->raw_data();
    std::vector<T> v2 = other.raw_data();
    using arch = xsimd::default_arch; // dispatch to SSE/AVX/NEON as appropriate
    using tag = xsimd::unaligned_mode;
    xsimd_ops::pow{}(arch{}, v1, v2, data, tag{});
    return Tensor<T>(shape, data, Device::CPU);
  };
  //
  Tensor<T> sum() {
    std::size_t size = flat_size();
    std::vector<T> data(1);          // reserve one slot
    std::vector<T> &in = raw_data(); // reference to your own storage

    using arch = xsimd::default_arch;
    xsimd_ops::sum{}(arch{}, in, data);

    // now `data[0]` holds the total
    return Tensor<T>({1}, std::move(data), Device::CPU);
  }

  //
  Tensor<T> maximum(const Tensor<T> &other) const {
    std::vector<size_t> shape = this->shape_;
    size_t               size  = this->flat_size();

    const auto &a = this->raw_data();
    std::vector<T> b;
    if (other.flat_size() == 1) {
      b.assign(size, other.raw_data()[0]);
    } else if (other.flat_size() == size) {
      b = other.raw_data();
    } else {
      throw std::invalid_argument("Shapes not compatible for maximum");
    }

    std::vector<T> data(size);
    using arch = xsimd::default_arch;
    using tag  = xsimd::unaligned_mode;
    xsimd_ops::maximum{}(arch{}, a, b, data, tag{});

    return Tensor<T>(shape, std::move(data), Device::CPU);
  }


  //
  Tensor<T> matmul(Tensor<T> &other) {
    auto const &shapeA = this->shape_;
    auto const &shapeB = other.shape_;

    size_t m = shapeA[0], n = shapeB[1];
    std::vector<T> data(m * n);

    cblas_ops::matmul(
      this->raw_data(), shapeA,
      other.raw_data(), shapeB,
      data
    );

    return Tensor<T>({m, n}, std::move(data), Device::CPU);
  }
  //
  Tensor<T> transpose() {
    std::vector<size_t> shape = this->shape_;
    std::vector<T> v1 = this->raw_data();
    std::vector<T> data;
    size_t size = this->flat_size();
    data.resize(size);
    serial_ops::transpose(v1, shape, data);
    return Tensor<T>(shape, data, Device::CPU);
  };
};
//
//     Tensor<T> diagonal() const;
// };
//
// #include "tensor_algorithms.ipp"
// #include "tensor_arithmetic.ipp"
// #include "tensor_constructors.ipp"
// #include "tensor_io.ipp"
// #include "tensor_reductions.ipp"
#endif // TENSOR_H

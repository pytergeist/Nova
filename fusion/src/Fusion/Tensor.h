#ifndef TENSOR_H
#define TENSOR_H
#include <ostream>
#include <stdexcept>
#include <vector>
#include <memory>
#include <utility>
#include <vector>
#include <cstring>

#include "storage/DenseStorage.h"
#include "storage/StorageInterface.h"
#include "core/ElementWise.h"
#include "core/Ffunc.h"
#include "core/Reduce.h"
#include "cpu/SimdTags.h"
#include "cpu/SimdTraits.h"
#include "kernels/Blas.cpp"
#include "kernels/Serial.h"

#include "common/Checks.h"

template <typename T> class Tensor {
public:
  std::shared_ptr<ITensorStorage<T>> storage;
  std::vector<size_t> shape_;
  size_t rank_;
  bool requires_grad_;

  Tensor()
     : storage(nullptr), shape_{}, rank_(0), requires_grad_(false) {}

  explicit Tensor(std::vector<size_t> shape, std::vector<T> data,
                  Device device = Device::CPU, bool requires_grad = false)
      : shape_(std::move(shape)),
        requires_grad_(std::move(requires_grad))  {
    FUSION_CHECK(device == Device::CPU, "Unsupported device type");
    FUSION_CHECK(!shape_.empty(), "Tensor: empty shape");
    size_t n = 1;
    for (auto d : shape_) { FUSION_CHECK(d > 0, "Tensor: non-positive dim"); n *= d; }
    FUSION_CHECK(data.size() == n, "Tensor: data size != product(shape)");
    storage = std::make_shared<NDTensorStorage<T>>(shape_, std::move(data));
    rank_ = storage->ndims();
    }

  Tensor(const Tensor&) = default; // TODO: make this delete once you've built non-owning TensorView
  Tensor& operator=(const Tensor&) = default;
  Tensor(Tensor&&) noexcept = default;
  Tensor& operator=(Tensor&&) noexcept = default;
  ~Tensor() = default;

  friend std::ostream &operator<<(std::ostream &os, const Tensor<T> &tensor) {
    const auto *cpuStorage =
        dynamic_cast<const NDTensorStorage<T> *>(tensor.storage.get());
    if (cpuStorage) {
      const TensorBuffer& buf = cpuStorage->data();
      const size_t n = cpuStorage->size();
      const T* p = buf.template data_as<const T>();
      os << "Tensor(";
      for (size_t i = 0; i < n; i++) {
        os << p[i];
        if (i + 1 < n)
          os << ", ";
      }
      os << ")" << std::endl;
    } else {
      os << "Tensor(unsupported storage type)";
    }
    return os;
  }

//  template <class Callable, class... Ops,
//            typename R = std::invoke_result_t<Callable, T, T>>
//  Tensor(FFunc<Callable, Ops...> const &ffunc) {
//    // 1) pull shape out of the ffunc
//    shape_ = ffunc.shape();
//    rank_ = shape_.size();
//    size_t n = ffunc.flat_size();
//
//    if constexpr (simd_traits<Callable, T>::available) {
//      std::vector<T> data(n);
//      // call your SIMD driver
//      simd_traits<Callable, T>::execute(ffunc, data.data());
//      storage = std::make_unique<NDTensorStorage<T>>(shape_, std::move(data));
//    } else {
//      std::vector<R> data(n);
//      for (size_t i = 0; i < n; ++i)
//        data[i] = ffunc[i];
//      storage = std::make_unique<NDTensorStorage<R>>(shape_, std::move(data));
//    }
//  }

//  Tensor clone() const {
//    return Tensor(
//      this->storage->shape(),
//      this->raw_data(),
//      this->storage->device(),
//      false
//    );
//   }

  T operator[](int idx) const {
    return storage->data().template data_as<const T>()[idx];
  }

  size_t size() const noexcept { return storage->data().template size<T>(); }

  void clear() noexcept {
    if (!storage) return;
    auto& buf = storage->data();
    if (buf.size_bytes() == 0) return;
    std::memset(buf.data(), 0, buf.size_bytes());
  }


  void assign(const Tensor<T> &other) {
    if (!storage) {
      *this = other;
      } else {
    storage->data().assign(other.begin(), other.end());
      }
    };

  bool empty() const noexcept { return !storage || storage->data().empty(); };

  bool is_initialised() const noexcept { return storage != nullptr; }

  TensorBuffer &raw_data() { return storage->data(); }
  const TensorBuffer &raw_data() const { return storage->data(); }
  [[nodiscard]] size_t flat_size() const { return storage->size(); }

  auto operator+(const Tensor &other) const {
    std::vector<size_t> out_shape;
    std::vector<T> out_data;
    ewise::binary_ewise_tag<T, AddSIMD>(*this, other, out_shape, out_data);
    return Tensor(std::move(out_shape), std::move(out_data), Device::CPU);
  }

  auto operator-(const Tensor &other) const {
    std::vector<size_t> out_shape;
    std::vector<T> out_data;
    ewise::binary_ewise_tag<T, SubtractSIMD>(*this, other, out_shape, out_data);
    return Tensor(std::move(out_shape), std::move(out_data), Device::CPU);
  }

  auto &operator-=(const Tensor &other) {
    auto &out_shape = this->shape_;
    auto &out_data = this->storage->data();
    ewise::binary_ewise_tag<T, SubtractSIMD>(*this, other, out_shape, out_data);
    return *this;
  }

  auto operator/(const Tensor &other) const {
    std::vector<size_t> out_shape;
    std::vector<T> out_data;
    ewise::binary_ewise_tag<T, DivideSIMD>(*this, other, out_shape, out_data);
    return Tensor(std::move(out_shape), std::move(out_data), Device::CPU);
  }

  auto operator*(const Tensor &other) const {
    std::vector<size_t> out_shape;
    std::vector<T> out_data;
    ewise::binary_ewise_tag<T, MultiplySIMD>(*this, other, out_shape, out_data);
    return Tensor(std::move(out_shape), std::move(out_data), Device::CPU);
  }

  auto operator>(const Tensor &other) const {
    std::vector<size_t> out_shape;
    std::vector<T> out_data;
    ewise::binary_ewise_tag<T, GreaterThanSIMD>(*this, other, out_shape,
                                                out_data);
    return Tensor(std::move(out_shape), std::move(out_data), Device::CPU);
  }

  auto &operator>=(const Tensor &other) {
    auto &out_shape = this->shape_;
    auto &out_data = this->storage->data();
    ewise::binary_ewise_tag<T, GreaterThanEqualSIMD>(*this, other, out_shape,
                                                     out_data);
    return *this;
  }

  auto operator>=(const Tensor &other) const {
    std::vector<size_t> out_shape;
    std::vector<T> out_data;
    ewise::binary_ewise_tag<T, GreaterThanEqualSIMD>(*this, other, out_shape,
                                                out_data);
    return Tensor(std::move(out_shape), std::move(out_data), Device::CPU);
  }

  auto maximum(const Tensor &other) const {
    std::vector<size_t> out_shape;
    std::vector<T> out_data;
    ewise::binary_ewise_tag<T, MaximumSIMD>(*this, other, out_shape, out_data);
    return Tensor(std::move(out_shape), std::move(out_data), Device::CPU);
  }

  auto sqrt() const {
    std::vector<size_t> out_shape;
    std::vector<T> out_data;
    ewise::unary_ewise_tag<T, SqrtSIMD>(*this, out_shape, out_data);
    return Tensor(std::move(out_shape), std::move(out_data), Device::CPU);
  }

  auto log() const {
    std::vector<size_t> out_shape;
    std::vector<T> out_data;
    ewise::unary_ewise_tag<T, NaturalLogSIMD>(*this, out_shape, out_data);
    return Tensor(std::move(out_shape), std::move(out_data), Device::CPU);
  };

  auto exp() const {
    std::vector<size_t> out_shape;
    std::vector<T> out_data;
    ewise::unary_ewise_tag<T, ExponentialSIMD>(*this, out_shape, out_data);
    return Tensor(std::move(out_shape), std::move(out_data), Device::CPU);
  };

  auto pow(const Tensor &other) const {
    std::vector<size_t> out_shape;
    std::vector<T> out_data;
    ewise::binary_ewise_tag<T, PowerSIMD>(*this, other, out_shape, out_data);
    return Tensor(std::move(out_shape), std::move(out_data), Device::CPU);
  }

  auto sum() const {
    const T *x = this->storage->data_ptr();
    const std::size_t n = this->flat_size();
    T acc = reduce::reduce_tag<T, GlobalSumSIMD>(x, n);
    return Tensor<T>({1}, std::vector<T>{acc});
  };

  Tensor<T> matmul(Tensor<T> &other) {
    auto const &shapeA = this->shape_;
    auto const &shapeB = other.shape_;
    size_t rank = shapeA.size();
    size_t m = shapeA[rank - 2];
    size_t n = shapeB[rank - 1];

    std::vector<size_t> out_shape = shapeA;
    out_shape[rank - 1] = n;

    size_t batch = 1;
    for (size_t i = 0; i < rank - 2; ++i) {
      batch *= shapeA[i];
    }

    std::vector<T> data(batch * m * n);

    blas_ops::matmul<T>(this->raw_data(), shapeA, other.raw_data(), shapeB, data);

    return Tensor<T>(std::move(out_shape), std::move(data), Device::CPU);
  }

  Tensor<T> swapaxes(int axis1, int axis2) {
    std::vector<size_t> out_shape = this->shape_;
    axis1 = serial_ops::normalise_axis(axis1, this->rank_);
    axis2 = serial_ops::normalise_axis(axis2, this->rank_);
    std::swap(out_shape[axis1], out_shape[axis2]);
    std::vector<T> out =
        serial_ops::swapaxes(this->raw_data(), this->shape_, axis1, axis2);
    return Tensor<T>(std::move(out_shape), std::move(out), Device::CPU);
  }

  Tensor<T> diagonal() {
    size_t arr_size =
        std::sqrt(std::accumulate(this->shape_.begin(), this->shape_.end(),
                                  int64_t{1}, std::multiplies<int>()));
    size_t out_dim = std::floor(arr_size);
    std::vector<size_t> out_shape{out_dim, 1};
    std::vector<T> out = serial_ops::diagonal2D(this->raw_data(), this->shape_);
    return Tensor<T>(std::move(out_shape), std::move(out), Device::CPU);
  }

  //
  Tensor<T> transpose() const {
    std::vector<size_t> new_shape(shape_.rbegin(), shape_.rend());

    size_t size = flat_size();
    std::vector<T> new_data(size);

    serial_ops::transpose<T>(*this, this->shape_, new_data);

    return Tensor<T>(std::move(new_shape), std::move(new_data), Device::CPU);
  }

  auto begin() { return storage->data().begin(); }
  auto end() { return storage->data().end(); }
  auto begin() const { return storage->data().begin(); }
  auto end() const { return storage->data().end(); }
};


#endif // TENSOR_H

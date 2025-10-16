#ifndef TENSOR_H
#define TENSOR_H
#include <cstring>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <utility>
#include <vector>

#include "core/ElementWise.h"
#include "core/Ffunc.h"
#include "core/Reduce.h"
#include "cpu/SimdTags.h"
#include "cpu/SimdTraits.h"
#include "kernels/Blas.h"
#include "kernels/Serial.h"
#include "storage/DenseStorage.h"
#include "storage/StorageInterface.h"
#include "ops/Ewise.h"
#include "ops/Reduce.h"
#include "ops/Comparison.h"
#include "ops/Linalg.h"
#include "ops/Transcendental.h"
#include "ops/Linalg.h"
#include "common/Checks.h"
#include "autodiff/AutodiffMode.h"
#include "autodiff/policies/Ewise/Ewise.h"
#include "autodiff/policies/LinAlg/LinAlg.h"
#include "autodiff/Engine.h"
#include "autodiff/EngineContext.h"
#include "autodiff/Dispatch.h"


  template <typename T>
  static inline ValueID ensure_handle(Engine<T>& eng, Tensor<T>& t) {
    if (t.eng_ == &eng && t.vid_.idx >= 0) return t.vid_;
    auto vid = eng.track_input(t);
    t.eng_ = &eng;
    t.vid_ = vid;
    return vid;
  }

template <typename T> class Tensor {
public:
  std::shared_ptr<ITensorStorage<T>> storage;
  std::vector<size_t> shape_;
  size_t rank_;
  ValueID vid_{-1};
  bool requires_grad_;

  Tensor() : storage(nullptr), shape_{}, rank_(0), requires_grad_(false) {}

  explicit Tensor(std::vector<size_t> shape, std::vector<T> data,
                  Device device = Device::CPU, bool requires_grad = false)
      : shape_(std::move(shape)), requires_grad_(std::move(requires_grad)) {
    FUSION_CHECK(device == Device::CPU, "Unsupported device type");
    FUSION_CHECK(!shape_.empty(), "Tensor: empty shape");
    size_t n = 1;
    for (auto d : shape_) {
      FUSION_CHECK(d > 0, "Tensor: non-positive dim");
      n *= d;
    }
    FUSION_CHECK(data.size() == n, "Tensor: data size != product(shape)");
    storage = std::make_shared<NDTensorStorage<T>>(shape_, std::move(data));
    rank_ = storage->ndims();
  }

  bool has_vid() const noexcept { return vid_.idx >= 0; }

  ValueID ensure_vid() {
  if (vid_.idx >= 0) return vid_;
  auto& eng = EngineContext<T>::get();
  vid_ = eng.track_input(*this);
  return vid_;
  }


  Tensor(const Tensor &) =
      default; // TODO: make this delete once you've built non-owning TensorView
  Tensor &operator=(const Tensor &) = default;
  Tensor(Tensor &&) noexcept = default;
  Tensor &operator=(Tensor &&) noexcept = default;
  ~Tensor() = default;

  bool requires_grad() const { return requires_grad_; }

  friend std::ostream &operator<<(std::ostream &os, const Tensor<T> &tensor) {
    const auto *cpuStorage =
        dynamic_cast<const NDTensorStorage<T> *>(tensor.storage.get());
    if (cpuStorage) {
      const TensorBuffer &buf = cpuStorage->data();
      const size_t n = cpuStorage->size();
      const T *p = buf.template data_as<const T>();
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
  //      storage = std::make_unique<NDTensorStorage<T>>(shape_,
  //      std::move(data));
  //    } else {
  //      std::vector<R> data(n);
  //      for (size_t i = 0; i < n; ++i)
  //        data[i] = ffunc[i];
  //      storage = std::make_unique<NDTensorStorage<R>>(shape_,
  //      std::move(data));
  //    }
  //  }


  T operator[](int idx) const {
    return storage->data().template data_as<const T>()[idx];
  }

  size_t size() const noexcept { return storage->data().template size<T>(); }

  void clear() noexcept {
    if (!storage)
      return;
    auto &buf = storage->data();
    if (buf.size_bytes() == 0)
      return;
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



  auto operator+(const Tensor& other) const {
    using AddOp = Operation<T, Add<T>>;
    return autodiff::binary<T, AddOp>(*this, other,
        [](const Tensor& x, const Tensor& y){ return math::add(x, y); });
  }



  auto operator-(const Tensor& other) const {
    using SubOp = Operation<T, Subtract<T>>;
    return autodiff::binary<T, SubOp>(*this, other,
        [](const Tensor& x, const Tensor& y){ return math::sub(x, y); });
  }

  auto &operator-=(const Tensor &other) {
    std::vector<size_t> out_shape;
    std::vector<T> out_data;
    ewise::binary_ewise_tag<T, SubtractSIMD>(*this, other, out_shape, out_data);
    storage =
        std::make_shared<NDTensorStorage<T>>(out_shape, std::move(out_data));
    shape_ = storage->shape();
    rank_ = storage->ndims();
    return *this;
  }

  auto operator/(const Tensor& other) const {
    using DivOp = Operation<T, Divide<T>>;
    return autodiff::binary<T, DivOp>(*this, other,
        [](const Tensor& x, const Tensor& y){ return math::div(x, y); });
  }

  auto operator*(const Tensor& other) const {
    using MulOp = Operation<T, Multiply<T>>;
    return autodiff::binary<T, MulOp>(*this, other,
        [](const Tensor& x, const Tensor& y){ return math::mul(x, y); });
  }

  auto operator>(const Tensor &other) const {return math::greater(*this, other); }

  auto &operator>=(const Tensor &other) {
    auto &out_shape = this->shape_;
    auto &out_data = this->storage->data();
    ewise::binary_ewise_tag<T, GreaterThanEqualSIMD>(*this, other, out_shape,
                                                     out_data);
    return *this;
  }

  auto operator>=(const Tensor &other) const {return math::greater_equal(*this, other); }

  auto maximum(const Tensor &other) const {return math::maximum(*this, other); }

  auto sqrt() const { return math::sqrt(*this); };

  auto log() const { return math::log(*this); };

  auto exp() const { return math::exp(*this); };

  auto pow(const Tensor &other) const { return math::pow(*this, other); };

  auto sum() const { return math::sum(*this); };

  Tensor<T> matmul(const Tensor<T> &other) const {
    using MatMulOp = Operation<T, MatMul<T>>;
    return autodiff::binary<T, MatMulOp>(*this, other,
        [](const Tensor& x, const Tensor& y){ return math::linalg::matmul(x, y); });
   };

  Tensor<T> swapaxes(int axis1, int axis2) const {
    std::vector<size_t> out_shape = this->shape_;
    axis1 = serial_ops::normalise_axis(axis1, this->rank_);
    axis2 = serial_ops::normalise_axis(axis2, this->rank_);
    std::swap(out_shape[axis1], out_shape[axis2]);
    std::vector<T> out =
        serial_ops::swapaxes<T>(*this, this->shape_, axis1, axis2);
    return Tensor<T>(std::move(out_shape), std::move(out), Device::CPU);
  }

  Tensor<T> diagonal() {
    size_t arr_size =
        std::sqrt(std::accumulate(this->shape_.begin(), this->shape_.end(),
                                  int64_t{1}, std::multiplies<int>()));
    size_t out_dim = std::floor(arr_size);
    std::vector<size_t> out_shape{out_dim, 1};
    std::vector<T> out = serial_ops::diagonal2D(*this, this->shape_);
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

  auto begin() { return storage->data().template begin<T>(); }
  auto end() { return storage->data().template end<T>(); }
  auto begin() const { return storage->data().template begin<T>(); }
  auto end() const { return storage->data().template end<T>(); }



};

#endif // TENSOR_H

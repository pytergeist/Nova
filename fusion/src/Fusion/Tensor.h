#ifndef TENSOR_H
#define TENSOR_H
#include <cstring>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <utility>
#include <vector>

#include "autodiff/AutodiffMode.h"
#include "autodiff/Dispatch.h"
#include "autodiff/Engine.h"
#include "autodiff/EngineContext.h"
#include "autodiff/policies/Comparison/Comparison.h"
#include "autodiff/policies/Ewise/Ewise.h"
#include "autodiff/policies/LinAlg/LinAlg.h"
#include "autodiff/policies/Reduction/Reduction.h"
#include "autodiff/policies/Transcendental/Transcendental.h"
#include "common/Checks.h"
#include "core/ElementWise.h"
#include "core/Ffunc.h"
#include "core/Reduce.h"
#include "cpu/SimdTags.h"
#include "cpu/SimdTraits.h"
#include "kernels/Blas.h"
#include "kernels/Serial.h"
#include "ops/Comparison.h"
#include "ops/Ewise.h"
#include "ops/Linalg.h"
#include "ops/Reduce.h"
#include "ops/Transcendental.h"
#include "storage/DenseStorage.h"
#include "storage/StorageInterface.h"
#include "storage/TensorView.h"
#include "ops/Helpers.h"

template <typename T>
static inline ValueID ensure_handle(Engine<T> &eng, Tensor<T> &t) {
   if (t.eng_ == &eng && t.vid_.idx >= 0)
      return t.vid_;
   auto vid = eng.track_input(t);
   t.eng_ = &eng;
   t.vid_ = vid;
   return vid;
}

template <typename T> class Tensor {
 public:
   std::shared_ptr<ITensorStorage<T>> storage;
   std::shared_ptr<Tensor<T>> grad_;
   std::vector<size_t> shape_, strides_;
   ValueID vid_{-1};
   bool requires_grad_;

   Tensor() : storage(nullptr), shape_{}, requires_grad_(false) {}

   explicit Tensor(std::vector<size_t> shape, std::vector<T> data,
                   Device device = Device::CPU, bool requires_grad = false)
       : shape_(std::move(shape)), requires_grad_(std::move(requires_grad)) {
      FUSION_CHECK(device == Device::CPU, "Unsupported device type");
      FUSION_CHECK(!shape_.empty(), "Tensor: empty shape");
      size_t sz = 1;
      strides_.resize(shape_.size());
      for (size_t i = 0; i < shape_.size(); i++) {
         strides_[i] = sz;
         sz *= shape_[i];
      }
      FUSION_CHECK(data.size() == sz, "Tensor: data size != product(shape)");
      storage = std::make_shared<NDTensorStorage<T>>(shape_, std::move(data));
   }

      explicit Tensor(std::vector<size_t> shape, Device device = Device::CPU, bool requires_grad = false)
       : shape_(std::move(shape)), requires_grad_(std::move(requires_grad)) {
      FUSION_CHECK(device == Device::CPU, "Unsupported device type");
      FUSION_CHECK(!shape_.empty(), "Tensor: empty shape");
      size_t sz = 1;
      strides_.resize(shape_.size());
      for (size_t i = 0; i < shape_.size(); i++) {
         strides_[i] = sz;
         sz *= shape_[i];
      }
      storage = std::make_shared<NDTensorStorage<T>>(shape_, sz);
   }

   size_t rank() const { return shape_.size(); }
   size_t ndims() const {
      return shape_.size();
   } // TODO: remove ndims (as == rank)
   std::vector<size_t> shape() const { return shape_; }
   std::vector<size_t> strides() const { return strides_; }

   TensorView<T> view() {
      return TensorView<T>(
          storage->data().template data<T>(),
          this->shape(), this->strides(), this->rank(), this->ndims());
   }

   bool has_vid() const noexcept { return vid_.idx >= 0; }

   ValueID ensure_vid() {
      if (vid_.idx >= 0)
         return vid_;
      auto &eng = EngineContext<T>::get();
      vid_ = eng.track_input(*this);
      return vid_;
   }

   ValueID get_vid() const noexcept { return vid_; }

   Tensor(const Tensor &) = default; // TODO: make this delete once you've built
                                     // non-owning TensorView
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

   std::string shape_str() const {
      std::ostringstream oss;
      oss << '(';
      for (size_t i = 0; i < shape_.size(); ++i) {
         oss << shape_[i];
         if (i + 1 < shape_.size())
            oss << ',';
      }
      oss << ')';
      return oss.str();
   }

   std::string stride_str() const {
      std::ostringstream oss;
      oss << '(';
      for (size_t i = 0; i < strides_.size(); ++i) {
         oss << strides_[i];
         if (i + 1 < strides_.size())
            oss << ',';
      }
      oss << ')';
      return oss.str();
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

   const size_t size() const noexcept {
      return storage->data().template size<T>();
   }

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
   [[nodiscard]] size_t flat_size() const {
      return storage->size();
   } // TODO: wtf is this

   void backward() {
      auto &eng = EngineContext<T>::get();
      FUSION_CHECK(this->has_vid(), "backward(): tensor has no ValueID");
      eng.backward(this->vid_);
   }

   //  Tensor<T> grad() {
   //	auto& eng = EngineContext<T>::get();
   //	return eng.get_grad(get_vid());
   //  }

   const Tensor<T> &grad() const {
      // Lazy, const-safe allocation of a zero-like grad buffer
      if (grad_ == nullptr)
         const_cast<Tensor<T> *>(this)->ensure_grad();
      return *grad_;
   }
   Tensor<T> &mutable_grad() {
      ensure_grad(); // ensures grad_ exists and is zero-like
      return *grad_;
   }
   bool has_grad() const noexcept { return grad_ && grad_->is_initialised(); };

   void ensure_grad() {
      if (!has_grad()) {
         std::vector<T> z(size(), T(0));
         grad_ = std::make_shared<Tensor<T>>(shape_, std::move(z), Device::CPU,
                                             false);
      }
   }

   auto operator+(const Tensor &other) const {
      using AddOp = Operation<T, Add<T>>;
      return autodiff::binary<T, AddOp>(
          *this, other,
          [](const Tensor &x, const Tensor &y) { return math::add(x, y); });
   }

   auto operator+(const T scalar) const {
      Tensor<T> other{{1}, {scalar}, Device::CPU, false};
      using AddOp = Operation<T, Add<T>>;
      return autodiff::binary<T, AddOp>(
          *this, other,
          [](const Tensor &x, const Tensor &y) { return math::add(x, y); });
   }

   auto operator*(const T scalar) const {
      Tensor<T> other{{1}, {scalar}, Device::CPU, false};
      using MulOp = Operation<T, Multiply<T>>;
      return autodiff::binary<T, MulOp>(
          *this, other,
          [](const Tensor &x, const Tensor &y) { return math::mul(x, y); });
   }

   auto operator/(const T scalar) const {
      Tensor<T> other{{1}, {scalar}, Device::CPU, false};
      using DivOp = Operation<T, Divide<T>>;
      return autodiff::binary<T, DivOp>(
          *this, other,
          [](const Tensor &x, const Tensor &y) { return math::div(x, y); });
   }

   auto operator-(const T scalar) const {
      Tensor<T> other{{1}, {scalar}, Device::CPU, false};
      using SubOp = Operation<T, Subtract<T>>;
      return autodiff::binary<T, SubOp>(
          *this, other,
          [](const Tensor &x, const Tensor &y) { return math::sub(x, y); });
   }

   auto maximum(const T scalar) const {
      Tensor<T> other{{1}, {scalar}, Device::CPU, false};
      using MaximumOp = Operation<T, Maximum<T>>;
      return autodiff::binary<T, MaximumOp>(
          *this, other,
          [](const Tensor &x, const Tensor &y) { return math::maximum(x, y); });
   }

   auto operator>=(const T scalar) const {
      Tensor<T> other{{1}, {scalar}, Device::CPU, false};
      using GreaterThanEqualOp = Operation<T, GreaterThanEqual<T>>;
      return autodiff::binary<T, GreaterThanEqualOp>(
          *this, other, [](const Tensor &x, const Tensor &y) {
             return math::greater_equal(x, y);
          });
   }

   auto pow(const T scalar) const {
      Tensor<T> other{{1}, {scalar}, Device::CPU, false};
      using PowOp = Operation<T, Pow<T>>;
      return autodiff::binary<T, PowOp>(
          *this, other,
          [](const Tensor &x, const Tensor &y) { return math::pow(x, y); });
   };

   auto operator-(const Tensor &other) const {
      using SubOp = Operation<T, Subtract<T>>;
      return autodiff::binary<T, SubOp>(
          *this, other,
          [](const Tensor &x, const Tensor &y) { return math::sub(x, y); });
   }

   auto &operator-=(const Tensor &other) { // TODO: fix this impl to go through op kernel layer
      std::vector<size_t> out_shape;
      Tensor<T> out = init_bin_out_tensor(*this, other);
      ewise::binary_ewise_tag<T, SubtractSIMD>(*this, other, out_shape,
                                               out);
//      storage =
//          std::make_shared<NDTensorStorage<T>>(out_shape, std::move(out_data));
      return *this;
   }

   auto operator/(const Tensor &other) const {
      using DivOp = Operation<T, Divide<T>>;
      return autodiff::binary<T, DivOp>(
          *this, other,
          [](const Tensor &x, const Tensor &y) { return math::div(x, y); });
   }

   auto operator*(const Tensor &other) const {
      using MulOp = Operation<T, Multiply<T>>;
      return autodiff::binary<T, MulOp>(
          *this, other,
          [](const Tensor &x, const Tensor &y) { return math::mul(x, y); });
   }

   auto operator>(const Tensor &other) const {
      using GreaterThanOp = Operation<T, GreaterThan<T>>;
      return autodiff::binary<T, GreaterThanOp>(
          *this, other,
          [](const Tensor &x, const Tensor &y) { return math::greater(x, y); });
   }

   auto operator>=(const Tensor &other) const {
      using GreaterThanEqualOp = Operation<T, GreaterThanEqual<T>>;
      return autodiff::binary<T, GreaterThanEqualOp>(
          *this, other, [](const Tensor &x, const Tensor &y) {
             return math::greater_equal(x, y);
          });
   }

   auto maximum(const Tensor &other) const {
      using MaximumOp = Operation<T, Maximum<T>>;
      return autodiff::binary<T, MaximumOp>(
          *this, other,
          [](const Tensor &x, const Tensor &y) { return math::maximum(x, y); });
   }

   auto sqrt() const {
      using SqrtOp = Operation<T, Sqrt<T>>;
      return autodiff::unary<T, SqrtOp>(
          *this, [](const Tensor &x) { return math::sqrt(x); });
   };

   auto log() const {
      using LogOp = Operation<T, Log<T>>;
      return autodiff::unary<T, LogOp>(
          *this, [](const Tensor &x) { return math::log(x); });
   };

   auto exp() const {
      using ExpOp = Operation<T, Exp<T>>;
      return autodiff::unary<T, ExpOp>(
          *this, [](const Tensor &x) { return math::exp(x); });
   };

   auto pow(const Tensor &other) const {
      using PowOp = Operation<T, Pow<T>>;
      return autodiff::binary<T, PowOp>(
          *this, other,
          [](const Tensor &x, const Tensor &y) { return math::pow(x, y); });
   };

   auto sum() const {
      using SumOp = Operation<T, Sum<T>>;
      return autodiff::unary<T, SumOp>(
          *this, [](const Tensor &x) { return math::sum(x); });
   };

   Tensor<T> matmul(const Tensor<T> &other) const {
      using MatMulOp = Operation<T, MatMul<T>>;
      return autodiff::binary<T, MatMulOp>(
          *this, other, [](const Tensor &x, const Tensor &y) {
             return math::linalg::matmul(x, y);
          });
   };

   Tensor<T> swapaxes(const int axis1, const int axis2) const {
      using SwapAxesOp = Operation<T, SwapAxes<T>>;
      using Param = std::vector<int>;
      std::vector<int> params;
      params.reserve(2);
      params.push_back(axis1);
      params.push_back(axis2);
      return autodiff::unary<T, SwapAxesOp, Param>(
          *this, params, [](const Tensor &x, const int axis1, const int axis2) {
             return math::linalg::swapaxes(x, axis1, axis2);
          });
   }

   auto mean() const {
      using MeanOp = Operation<T, Mean<T>>;
      return autodiff::unary<T, MeanOp>(
          *this, [](const Tensor &x) { return math::mean(x); });
   };

   //  Tensor<T> swapaxes(int axis1, int axis2) const {
   //    using SwapOp = Operation<T, SwapAxes<T>>;
   //
   //    auto eager = [axis1, axis2](const Tensor<T>& x) {
   //        std::vector<size_t> out_shape = x.shape_;
   //        int a1 = serial_ops::normalise_axis(axis1, x.rank());
   //        int a2 = serial_ops::normalise_axis(axis2, x.rank());
   //        std::swap(out_shape[a1], out_shape[a2]);
   //        std::vector<T> out = serial::swapaxes<T>(x, x.shape_, a1, a2);
   //        return Tensor<T>(std::move(out_shape), std::move(out), Device::CPU,
   //        x.requires_grad_);
   //    };
   //
   //    if (!autodiff::grad_enabled() || !requires_grad_) {
   //        return eager(*this);
   //    }
   //
   //    auto& eng = EngineContext<T>::get();
   //    AutodiffMeta<T> meta(1);
   //    meta.push_back(*this);
   //    meta.set_param("axis1", axis1);
   //    meta.set_param("axis2", axis2);
   //
   //    ValueID out = eng.template apply<SwapOp>(std::move(meta));
   //    return eng.materialise(out);
   //}

   Tensor<T> diagonal() {
      size_t arr_size =
          std::sqrt(std::accumulate(this->shape_.begin(), this->shape_.end(),
                                    int64_t{1}, std::multiplies<int>()));
      size_t out_dim = std::floor(arr_size);
      std::vector<size_t> out_shape{out_dim, 1};
      std::vector<T> out = serial::diagonal2D(*this, this->shape_);
      return Tensor<T>(std::move(out_shape), std::move(out), Device::CPU);
   }

   //
   Tensor<T> transpose() const {
      std::vector<size_t> new_shape(shape_.rbegin(), shape_.rend());

      size_t size = flat_size();
      std::vector<T> new_data(size);

      serial::transpose<T>(*this, this->shape_, new_data);

      return Tensor<T>(std::move(new_shape), std::move(new_data), Device::CPU);
   }

   auto begin() { return storage->data().template begin<T>(); }
   auto end() { return storage->data().template end<T>(); }
   auto begin() const { return storage->data().template begin<T>(); }
   auto end() const { return storage->data().template end<T>(); }
};

#endif // TENSOR_H

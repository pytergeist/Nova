#ifndef AD_TENSOR_H
#define AD_TENSOR_H

#include <cstring>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <utility>
#include <vector>

#include "Fusion/common/Checks.h"

#include "AutodiffMode.h"
#include "Dispatch.h"
#include "Engine.h"
#include "EngineContext.h"
#include "policies/Comparison/Comparison.h"
#include "policies/Ewise/Ewise.h"
#include "policies/LinAlg/LinAlg.h"
#include "policies/Reduction/Reduction.h"
#include "policies/Transcendental/Transcendental.h"

#include "Fusion/ops/Comparison.h"
#include "Fusion/ops/Ewise.h"
#include "Fusion/ops/Helpers.h"
#include "Fusion/ops/Linalg.h"
#include "Fusion/ops/OpParams.h"
#include "Fusion/ops/Reduce.h"
#include "Fusion/ops/Transcendental.h"

#include "Fusion/core/DTypes.h"
#include "Fusion/core/ElementWise.h"
#include "Fusion/core/Ffunc.h"
#include "Fusion/core/Layout.h"
#include "Fusion/core/Reduce.h"
#include "Fusion/core/TensorBase.h"
#include "Fusion/cpu/SimdTags.h"

#include "Fusion/storage/DenseStorage.h"
#include "Fusion/storage/StorageInterface.h"
#include "Fusion/storage/TensorView.h"

#include "Fusion/alloc/DefaultAllocator.h"

template <typename T> struct ADTensor;

template <typename T>
static inline ValueID ensure_handle(Engine<T> &eng, ADTensor<T> &t) {
   if (t.eng_ == &eng && t.vid() >= 0) {
      return t.vid();
   }
   ValueID vid = eng.track_input(t);
   t.eng_ = &eng; // TODO: make this a setter
   t.set_vid(vid);
   return vid;
}

template <typename T> // TODO: need to either pass in device somehow?
inline ADTensor<T> ad_scalar_t(const T scalar,
                               const DType dtype = DType::Float32,
                               Device device = Device::CPU) {
   return ADTensor<T>{{1}, {scalar}, dtype, device, false};
}

template <typename T> class ADTensor : public TensorBase<T> {
 public:
   static constexpr std::string_view name = "ADTensor";
   using Base = TensorBase<T>;
   using value_type = T;

   ADTensor() : Base(), requires_grad_(false) {}

   explicit ADTensor(Base &&base, bool requires_grad = false)
       : Base(std::move(base)), requires_grad_(requires_grad) {}

   explicit ADTensor(std::vector<std::size_t> shape, // NOLINT
                     std::vector<T> data,            // NOLINT
                     DType dtype = DType::Float32,   // NOLINT
                     Device device = Device::CPU, bool requires_grad = false,
                     IAllocator *allocator = nullptr)
       : Base(std::move(shape), std::move(data), dtype, device, allocator),
         requires_grad_(std::move(requires_grad)) {}

   explicit ADTensor(std::vector<size_t> shape, Device device = Device::CPU,
                     DType dtype = DType::Float32, bool requires_grad = false,
                     IAllocator *allocator = nullptr)
       : Base(std::move(shape), device, dtype, allocator),
         requires_grad_(std::move(requires_grad)) {}

   ValueID vid() { return vid_; }
   ValueID vid() const { return vid_; }

   ValueID set_vid(ValueID vid) noexcept { return vid_ = vid; }

   bool has_vid() const noexcept { return vid_ >= 0; }

   ValueID ensure_vid() {
      if (vid_ >= 0) {
         return vid_;
      }
      Engine<T> &eng = EngineContext<T>::get();
      vid_ = eng.track_input(*this);
      return vid_;
   }

   bool requires_grad() const noexcept { return requires_grad_; }
   void set_requires_grad(bool v) noexcept { requires_grad_ = v; }

   void backward() {
      Engine<T> &eng = EngineContext<T>::get();
      FUSION_CHECK(this->has_vid(), "backward(): ADTensor has no ValueID");
      eng.backward(this->vid_);
   }

   const ADTensor<T> &grad() const {
      if (grad_ == nullptr) {
         // TODO: remove const_cast from here
         const_cast<ADTensor<T> *>(this)->ensure_grad(); // NOLINT
      }
      return *grad_;
   }

   ADTensor<T> &mutable_grad() {
      ensure_grad(); // ensures grad_ exists and is zero-like
      return *grad_;
   }
   bool has_grad() const noexcept { return grad_ && grad_->is_initialised(); };

   void ensure_grad() {
      if (!has_grad()) {
         std::vector<T> z(this->size(), T(0));
         grad_ = std::make_shared<ADTensor<T>>(
             this->shape(), std::move(z), this->dtype(), Device::CPU, false);
      }
   }

   ADTensor<T> operator+(const T scalar) const {
      ADTensor<T> other = ad_scalar_t(scalar, this->dtype(), this->device());
      return apply_binary_op<Add<T>>(
          other, [](const Base &x, const Base &y) { return x + y; });
   }

   ADTensor<T> operator-(const T scalar) const {
      ADTensor<T> other = ad_scalar_t(scalar, this->dtype(), this->device());
      return apply_binary_op<Subtract<T>>(
          other, [](const Base &x, const Base &y) { return x - y; });
   }

   ADTensor<T> operator/(const T scalar) const {
      ADTensor<T> other = ad_scalar_t(scalar, this->dtype(), this->device());
      return apply_binary_op<Divide<T>>(
          other, [](const Base &x, const Base &y) { return x / y; });
   }

   ADTensor<T> operator*(const T scalar) const {
      ADTensor<T> other = ad_scalar_t(scalar, this->dtype(), this->device());
      return apply_binary_op<Multiply<T>>(
          other, [](const Base &x, const Base &y) { return x * y; });
   }

   ADTensor<T> operator>=(const T scalar) const {
      ADTensor<T> other = ad_scalar_t(scalar, this->dtype(), this->device());
      return apply_binary_op<GreaterThanEqual<T>>(
          other, [](const Base &x, const Base &y) { return x >= y; });
   }

   ADTensor<T> maximum(const T scalar) const {
      ADTensor<T> other = ad_scalar_t(scalar, this->dtype(), this->device());
      return apply_binary_op<Maximum<T>>(
          other, [](const Base &x, const Base &y) { return x.maximum(y); });
   }

   ADTensor<T> pow(const T scalar) const {
      ADTensor<T> other = ad_scalar_t(scalar, this->dtype(), this->device());
      return apply_binary_op<Pow<T>>(
          other, [](const Base &x, const Base &y) { return x.pow(y); });
   }

   ADTensor<T> operator+(const ADTensor &other) const {
      return apply_binary_op<Add<T>>(
          other, [](const Base &x, const Base &y) { return x + y; });
   }

   ADTensor<T> operator-(const ADTensor &other) const {
      return apply_binary_op<Subtract<T>>(
          other, [](const Base &x, const Base &y) { return x - y; });
   }

   ADTensor<T> operator/(const ADTensor &other) const {
      return apply_binary_op<Divide<T>>(
          other, [](const Base &x, const Base &y) { return x / y; });
   }

   ADTensor<T> operator*(const ADTensor &other) const {
      return apply_binary_op<Multiply<T>>(
          other, [](const Base &x, const Base &y) { return x * y; });
   }

   ADTensor<T> operator>(const ADTensor &other) const {
      return apply_binary_op<GreaterThan<T>>(
          other, [](const Base &x, const Base &y) { return x > y; });
   }

   ADTensor<T> operator>=(const ADTensor &other) const {
      return apply_binary_op<GreaterThanEqual<T>>(
          other, [](const Base &x, const Base &y) { return x >= y; });
   }

   ADTensor<T> maximum(const ADTensor &other) const {
      return apply_binary_op<Maximum<T>>(
          other, [](const Base &x, const Base &y) { return x.maximum(y); });
   }

   ADTensor<T> pow(const ADTensor &other) const {
      return apply_binary_op<Pow<T>>(
          other, [](const Base &x, const Base &y) { return x.pow(y); });
   }

   ADTensor<T> matmul(const ADTensor<T> &other) const {
      return apply_binary_op<MatMul<T>>(
          other, [](const Base &x, const Base &y) { return x.matmul(y); });
   }

   ADTensor<T> sqrt() const {
      return apply_unary_op<Sqrt<T>>([](const Base &x) { return x.sqrt(); });
   }

   ADTensor<T> log() const {
      return apply_unary_op<Log<T>>([](const Base &x) { return x.log(); });
   }

   ADTensor<T> exp() const {
      return apply_unary_op<Exp<T>>([](const Base &x) { return x.exp(); });
   }

   ADTensor<T> sum() const {
      return apply_unary_op<Sum<T>>([](const Base &x) { return x.sum(); });
   }

   ADTensor<T> mean() const {
      return apply_unary_op<Mean<T>>([](const Base &x) { return x.mean(); });
   }

   ADTensor<T> swapaxes(int axis1, int axis2) const {
      using SwapAxesOp = SwapAxes<T>;
      SwapAxesParam sp{.axis1 = axis1, .axis2 = axis2};

      return apply_unary_op<SwapAxesOp, SwapAxesParam>(
          sp, [](const Base &x, const SwapAxesParam &p) {
             return x.swapaxes(p.axis1, p.axis2);
          });
   }

   ADTensor<T> &operator-=(const ADTensor &other) {
      const Base &bself = static_cast<const Base &>(*this);
      const Base &bother = static_cast<const Base &>(other);

      BinaryEwiseMeta meta = make_binary_meta(bself, bother);

      TensorBase<T> tmp_base = init_out_from_meta(bself, bother, meta);
      ewise::binary_ewise_tag<T, SubtractSIMD>(bself, bother, meta, tmp_base);

      if (!meta.out_shape.empty() && meta.out_shape != tmp_base.shape()) {
         ADTensor<T> corrected(meta.out_shape, Device::CPU, this->dtype(),
                               this->requires_grad());
         ewise::binary_ewise_tag<T, SubtractSIMD>(
             bself, bother, meta, static_cast<Base &>(corrected));
         ad_replace_from(corrected);
      } else {
         ADTensor<T> tmp(std::move(tmp_base), this->requires_grad());
         ad_replace_from(tmp);
      }
      return *this;
   }

 private:
   std::shared_ptr<ADTensor<T>> grad_;
   ValueID vid_{-1};
   bool requires_grad_;

   void ad_replace_from(ADTensor<T> &tmp) {
      const bool shape_changed = (this->shape() != tmp.shape());
      if (shape_changed) {
         grad_.reset();
      }

      this->storage_.swap(tmp.storage());
      this->shape_.swap(tmp.shape_);
      this->strides_.swap(tmp.strides_);

      vid_ = ValueID{-1};
   }

   static bool grad_flow(const ADTensor<T> &x, const ADTensor<T> &y) {
      return x.requires_grad() || y.requires_grad();
   };

   template <typename OpTag, typename F>
   ADTensor<T> apply_binary_op(const ADTensor &other, F &&f) const {
      using Op = Operation<T, OpTag>;
      const ADTensor &self = *this;
      F ff = std::forward<F>(f);
      return autodiff::binary<T, Op>(
          self, other, [&](const ADTensor &x, const ADTensor &y) {
             const Base &xb = static_cast<const Base &>(x);
             const Base &yb = static_cast<const Base &>(y);

             Base out = ff(xb, yb);

             bool req_grad = grad_flow(x, y);
             return ADTensor<T>(std::move(out), req_grad);
          });
   }

   template <typename OpTag, typename F>
   ADTensor<T> apply_unary_op(F &&f) const {
      using Op = Operation<T, OpTag>;
      const ADTensor &self = *this;
      F ff = std::forward<F>(f);
      return autodiff::unary<T, Op>(self, [&](const ADTensor &x) {
         const Base &xb = static_cast<const Base &>(x);
         Base out = ff(xb);
         bool req_grad = x.requires_grad();
         return ADTensor<T>(std::move(out), req_grad);
      });
   }

   template <typename OpTag, typename Param, typename F>
   ADTensor<T> apply_unary_op(const Param &p, F &&f) const {
      using Op = Operation<T, OpTag>;
      const ADTensor &self = *this;
      F ff = std::forward<F>(f);
      return autodiff::unary<T, Op, Param>(
          self, p, [&](const ADTensor &x, const Param &param) {
             const Base &xb = static_cast<const Base &>(x);
             Base out = ff(xb, param);
             bool req_grad = x.requires_grad();
             return ADTensor<T>(std::move(out), req_grad);
          });
   }
};

#endif // AD_TENSOR_H

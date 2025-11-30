#ifndef AD_TENSOR_H
#define AD_TENSOR_H

#include <cstring>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <utility>
#include <vector>

#include "Fusion/alloc/DefaultAllocator.h"
#include "Fusion/autodiff/AutodiffMode.h"
#include "Fusion/autodiff/Dispatch.h"
#include "Fusion/autodiff/Engine.h"
#include "Fusion/autodiff/EngineContext.h"
#include "Fusion/autodiff/policies/Comparison/Comparison.h"
#include "Fusion/autodiff/policies/Ewise/Ewise.h"
#include "Fusion/autodiff/policies/LinAlg/LinAlg.h"
#include "Fusion/autodiff/policies/Reduction/Reduction.h"
#include "Fusion/autodiff/policies/Transcendental/Transcendental.h"
#include "Fusion/common/Checks.h"
#include "Fusion/core/DTypes.h"
#include "Fusion/core/ElementWise.h"
#include "Fusion/core/Ffunc.h"
#include "Fusion/core/Layout.h"
#include "Fusion/core/Reduce.h"
#include "Fusion/cpu/SimdTags.h"
#include "Fusion/cpu/SimdTraits.h"
#include "Fusion/kernels/Serial.h"
#include "Fusion/ops/Comparison.h"
#include "Fusion/ops/Ewise.h"
#include "Fusion/ops/Helpers.h"
#include "Fusion/ops/Linalg.h"
#include "Fusion/ops/OpParams.h"
#include "Fusion/ops/Reduce.h"
#include "Fusion/ops/Transcendental.h"
#include "Fusion/storage/DenseStorage.h"
#include "Fusion/storage/StorageInterface.h"
#include "Fusion/storage/TensorView.h"

#include "Fusion/core/TensorBase.h"

template <typename T> struct ADTensor;

template <typename T>
static inline ValueID ensure_handle(Engine<T> &eng, ADTensor<T> &t) {
   if (t.eng_ == &eng && t.vid().idx >= 0) {
      return t.vid();
   }
   auto vid = eng.track_input(t);
   t.eng_ = &eng; // TODO: make this a setter
   t.set_vid(vid);
   return vid;
}

template <typename T> class ADTensor : public TensorBase<T> {
 public:
   using Base = TensorBase<T>;
   using value_type = T;

   ADTensor() : Base(), requires_grad_(false) {}

   explicit ADTensor(Base&& base, bool requires_grad = false)
       : Base(std::move(base)), requires_grad_(requires_grad) {}

   explicit ADTensor(std::vector<std::size_t> shape, std::vector<T> data,
                     DType dtype = DType::Float32, // NOLINT
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
   ValueID set_vid(ValueID vid) noexcept { return vid_ = vid; }
   const ValueID vid() const { return vid_; }

   bool has_vid() const noexcept { return vid_.idx >= 0; }

   ValueID ensure_vid() {
      if (vid_.idx >= 0) {
         return vid_;
      }
      auto &eng = EngineContext<T>::get();
      vid_ = eng.track_input(*this);
      return vid_;
   }

   ValueID get_vid() const noexcept { return vid_; }

   bool requires_grad() const noexcept { return requires_grad_; }
   void set_requires_grad(bool v) noexcept { requires_grad_ = v; }

   void backward() {
      auto &eng = EngineContext<T>::get();
      FUSION_CHECK(this->has_vid(), "backward(): ADTensor has no ValueID");
      eng.backward(this->vid_);
   }

   const ADTensor<T> &grad() const {
      if (grad_ == nullptr) {
         const_cast<ADTensor<T> *>(this)->ensure_grad();
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

   ADTensor<T> operator+(const ADTensor &other) const {
      return apply_binary_op<Add<T>>(
          other, [](const Base &x, const Base &y) { return x + y; });
   }

   ADTensor<T> operator-(const ADTensor &other) const {
      return apply_binary_op<Subtract<T>>(
          other, [](const Base &x, const Base &y) { return x - y; });
   }

   auto &operator-=(const ADTensor &other) {
      BinaryEwiseMeta meta = make_binary_meta(*this, other);
      ADTensor<T> tmp = init_out_from_meta(*this, other, meta);
      ewise::binary_ewise_tag<T, SubtractSIMD>(*this, other, meta, tmp);
      if (!meta.out_shape.empty() && meta.out_shape != tmp.shape()) {
         ADTensor<T> corrected(meta.out_shape, Device::CPU, this->dtype(),
                               grad_flow(*this, other));
         ewise::binary_ewise_tag<T, SubtractSIMD>(*this, other, meta,
                                                  corrected);
         replace_from_tmp(std::move(corrected));
      } else {
         replace_from_tmp(std::move(tmp));
      }
      return *this;
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

   auto sqrt() const {
      return apply_unary_op<Sqrt<T>>(
          [](const Base &x) { return x.sqrt(); });
   }

   auto log() const {
      return apply_unary_op<Log<T>>(
          [](const Base &x) { return x.log(); });
   }

   auto exp() const {
      return apply_unary_op<Exp<T>>(
          [](const Base &x) { return x.exp(); });
   }

   ADTensor<T> pow(const ADTensor &other) const {
      return apply_binary_op<Pow<T>>(
          other, [](const Base &x, const Base &y) { return x.pow(y); });
   }

   ADTensor<T> sum() const {
      return apply_unary_op<Sum<T>>([](const Base &x) { return x.sum(); });
   }

   ADTensor<T> matmul(const ADTensor<T> &other) const {
      return apply_binary_op<MatMul<T>>(
          other, [](const Base &x, const Base &y) { return x.matmul(y); });
   }

   ADTensor<T> swapaxes(int axis1, int axis2) const {
      using SwapAxesOp = SwapAxes<T>;
      SwapAxesParam sp{axis1, axis2};

      return apply_unary_op<SwapAxesOp, SwapAxesParam>(
          sp,
          [](const Base& x, const SwapAxesParam& p) {
             return x.swapaxes(p.axis1, p.axis2);
          });
   }

   auto mean() const {
      return apply_unary_op<Mean<T>>([](const Base &x) { return x.mean(); });
   }

 private:
   std::shared_ptr<ADTensor<T>> grad_;
   ValueID vid_{-1};
   bool requires_grad_;

   static bool grad_flow(const ADTensor<T> &x, const ADTensor<T> &y) {
   return x.requires_grad() || y.requires_grad();
  };

   template <typename OpTag, typename F>
   auto apply_binary_op(const ADTensor &other, F &&f) const {
      using Op = Operation<T, OpTag>;
      const ADTensor &self = *this;

      return autodiff::binary<T, Op>(
          self, other, [&](const ADTensor &x, const ADTensor &y) {
             const Base &xb = static_cast<const Base &>(x);
             const Base &yb = static_cast<const Base &>(y);

             Base out = f(xb, yb);

             bool req_grad = grad_flow(x, y);
             return ADTensor<T>(std::move(out), req_grad);
          });
   }

   template <typename OpTag, typename F> auto apply_unary_op(F &&f) const {
      using Op = Operation<T, OpTag>;
      const ADTensor &self = *this;

      return autodiff::unary<T, Op>(self, [&](const ADTensor &x) {
         const Base &xb = static_cast<const Base &>(x);
         Base out = f(xb);
         bool req_grad = x.requires_grad();
         return ADTensor<T>(std::move(out), req_grad);
      });
   }

   template <typename OpTag, typename Param, typename F>
   auto apply_unary_op(const Param& p, F&& f) const {
      using Op = Operation<T, OpTag>;
      const ADTensor& self = *this;

      return autodiff::unary<T, Op, Param>(
          self,
          p,
          [&](const ADTensor& x, const Param& param) {
             const Base& xb = static_cast<const Base&>(x);
             Base out = f(xb, param);
         	 bool req_grad = x.requires_grad();
         	 return ADTensor<T>(std::move(out), req_grad);
          });
   }



   void replace_from_tmp(ADTensor<T> &&tmp) {
      this->replace_from(std::move(tmp));
      grad_.reset();
      vid_ = ValueID{-1};
   }
};

#endif // AD_TENSOR_H

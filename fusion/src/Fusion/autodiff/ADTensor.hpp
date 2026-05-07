#ifndef AD_TENSOR_HPP
#define AD_TENSOR_HPP

#include <memory>
#include <utility>
#include <vector>

#include "AutodiffContext.hpp"
#include "AutodiffMode.hpp"
#include "Dispatch.hpp"
#include "registry/Comparison/Comparison.h"
#include "registry/Ewise/Ewise.h"
#include "registry/LinAlg/LinAlg.h"
#include "registry/Reduction/ReductionPolicy.h"
#include "registry/Transcendental/Transcendental.h"

#include "Fusion/ops/OpParams.hpp"

#include "Fusion/alloc/DefaultAllocator.h"

template <typename T> struct ADTensor;

// TODO: this doesn't follow the rule of 5

template <typename T>
ADTensor<T> ad_scalar_t(const T scalar, const DType dtype, Device device) {
   return ADTensor<T>{{1}, {scalar}, dtype, device, false};
}

template <typename T> class ADTensor {
 public:
   static constexpr std::string_view name = "ADTensor";
   using Raw = RawTensor<T>;
   using value_type = T;

   ADTensor() : raw_(), requires_grad_(false) {}

   explicit ADTensor(const Raw &raw, bool requires_grad = false)
       : raw_(raw), requires_grad_(requires_grad) {}

   explicit ADTensor(std::vector<std::size_t> shape, // NOLINT
                     std::vector<T> data,            // NOLINT
                     DType dtype, Device device, bool requires_grad = false,
                     IAllocator *allocator = nullptr)
       : raw_(std::move(shape), std::move(data), dtype, device, allocator),
         requires_grad_(std::move(requires_grad)) {}

   explicit ADTensor(std::vector<std::size_t> shape, DType dtype, Device device,
                     bool requires_grad = false,
                     IAllocator *allocator = nullptr)
       : raw_(std::move(shape), dtype, device, allocator),
         requires_grad_(std::move(requires_grad)) {}

   const RawTensor<T> &raw() const noexcept { return raw_; }
   RawTensor<T> &raw() noexcept { return raw_; }

   bool empty() const noexcept { return raw_.empty(); }
   bool is_initialised() const noexcept { return raw_.is_initialised(); }
   const void *get_storage() const noexcept { return raw_.get_storage(); }

   std::vector<std::size_t> shape() const { return raw_.shape(); }
   std::vector<std::int64_t> strides() const { return raw_.strides(); }

   std::size_t ndims() const { return raw_.ndims(); }
   size_t size() const { return raw_.size(); }

   Device device() const { return raw_.device(); }
   DType dtype() const { return raw_.dtype(); }

   std::size_t flat_size() const { return raw_.flat_size(); }
   std::size_t rank() const { return raw_.rank(); }

   ValueID vid() { return vid_; }
   ValueID vid() const { return vid_; }

   ValueID set_vid(const ValueID vid) {
      FUSION_CHECK(vid_ < 0, "Trying to overwrite Autodiff ValueID");
      return vid_ = vid;
   }

   bool has_vid() const noexcept { return vid_ >= 0; }

   ValueID ensure_vid() {
      Engine<T> &eng = AutodiffContext<T>::get();

      if (vid_ >= 0) {
         if (eng.has_value(vid_)) {
            return vid_;
         }
      }

      vid_ = eng.track_input(raw_, requires_grad_);

      if (requires_grad_) {
         grad_slot_id_ = eng.get_grad_slot(vid_);
      }

      return vid_;
   }

   bool requires_grad() const noexcept { return requires_grad_; }
   void set_requires_grad(const bool requires_grad) noexcept {
      requires_grad_ = requires_grad;
   }

   void backward() {
      Engine<T> &eng = AutodiffContext<T>::get();
      ValueID vid = ensure_vid();
      eng.backward(vid);
   }

   std::optional<ADTensor<T>> grad() const noexcept {
      if (grad_slot_id_ == -1 || !requires_grad_) {
         return std::nullopt;
      }

      GradStore<T> &store = AutodiffContext<T>::runtime().grad_store();
      if (!store.has(grad_slot_id_)) {
         return std::nullopt;
      }
      RawTensor<T> grad = store.get(grad_slot_id_);
      return ADTensor<T>(grad, false);
   }

   ADTensor operator+(const T scalar) const {
      ADTensor other = ad_scalar_t(scalar, raw_.dtype(), raw_.device());
      return apply_binary_op<Add<T>>(
          other, [](const Raw &x, const Raw &y) { return x + y; });
   }

   ADTensor operator-(const T scalar) const {
      ADTensor other = ad_scalar_t(scalar, raw_.dtype(), raw_.device());
      return apply_binary_op<Subtract<T>>(
          other, [](const Raw &x, const Raw &y) { return x - y; });
   }

   ADTensor operator/(const T scalar) const {
      ADTensor other = ad_scalar_t(scalar, raw_.dtype(), raw_.device());
      return apply_binary_op<Divide<T>>(
          other, [](const Raw &x, const Raw &y) { return x / y; });
   }

   ADTensor operator*(const T scalar) const {
      ADTensor other = ad_scalar_t(scalar, raw_.dtype(), raw_.device());
      return apply_binary_op<Multiply<T>>(
          other, [](const Raw &x, const Raw &y) { return x * y; });
   }

   ADTensor operator>=(const T scalar) const {
      ADTensor other = ad_scalar_t(scalar, raw_.dtype(), raw_.device());
      return apply_binary_op<GreaterThanEqual<T>>(
          other, [](const Raw &x, const Raw &y) { return x >= y; });
   }

   ADTensor maximum(const T scalar) const {
      ADTensor other = ad_scalar_t(scalar, raw_.dtype(), raw_.device());
      return apply_binary_op<Maximum<T>>(
          other, [](const Raw &x, const Raw &y) { return x.maximum(y); });
   }

   ADTensor pow(const T scalar) const {
      ADTensor other = ad_scalar_t(scalar, raw_.dtype(), raw_.device());
      return apply_binary_op<Pow<T>>(
          other, [](const Raw &x, const Raw &y) { return x.pow(y); });
   }

   ADTensor operator+(const ADTensor &other) const {
      return apply_binary_op<Add<T>>(
          other, [](const Raw &x, const Raw &y) { return x + y; });
   }

   ADTensor operator-(const ADTensor &other) const {
      return apply_binary_op<Subtract<T>>(
          other, [](const Raw &x, const Raw &y) { return x - y; });
   }

   ADTensor operator/(const ADTensor &other) const {
      return apply_binary_op<Divide<T>>(
          other, [](const Raw &x, const Raw &y) { return x / y; });
   }

   ADTensor operator*(const ADTensor &other) const {
      return apply_binary_op<Multiply<T>>(
          other, [](const Raw &x, const Raw &y) { return x * y; });
   }

   ADTensor operator>(const ADTensor &other) const {
      return apply_binary_op<GreaterThan<T>>(
          other, [](const Raw &x, const Raw &y) { return x > y; });
   }

   ADTensor operator>=(const ADTensor &other) const {
      return apply_binary_op<GreaterThanEqual<T>>(
          other, [](const Raw &x, const Raw &y) { return x >= y; });
   }

   ADTensor maximum(const ADTensor &other) const {
      return apply_binary_op<Maximum<T>>(
          other, [](const Raw &x, const Raw &y) { return x.maximum(y); });
   }

   ADTensor pow(const ADTensor &other) const {
      return apply_binary_op<Pow<T>>(
          other, [](const Raw &x, const Raw &y) { return x.pow(y); });
   }

   ADTensor reciprocal() const {
      return apply_unary_op<Reciprocal<T>>(
          [](const Raw &x) { return x.reciprocal(); });
   }

   ADTensor matmul(const ADTensor &other) const {
      return apply_binary_op<MatMul<T>>(
          other, [](const Raw &x, const Raw &y) { return x.matmul(y); });
   }

   ADTensor sqrt() const {
      return apply_unary_op<Sqrt<T>>([](const Raw &x) { return x.sqrt(); });
   }

   ADTensor log() const {
      return apply_unary_op<Log<T>>([](const Raw &x) { return x.log(); });
   }

   ADTensor exp() const {
      return apply_unary_op<Exp<T>>([](const Raw &x) { return x.exp(); });
   }

   ADTensor sum(const std::size_t axis, const bool keepdim) const {
      ReductionParam rp{.reduction_axis = axis, .keepdim = keepdim};
      return apply_unary_op<Sum<T>, ReductionParam>(
          rp, [](const Raw &x, const ReductionParam &p) {
             return x.sum(p.reduction_axis, p.keepdim);
          });
   }

   ADTensor mean(const std::size_t axis, const bool keepdim) const {
      ReductionParam rp{.reduction_axis = axis, .keepdim = keepdim};
      return apply_unary_op<Mean<T>, ReductionParam>(
          rp, [](const Raw &x, const ReductionParam &p) {
             return x.mean(p.reduction_axis, p.keepdim);
          });
   }

   ADTensor swapaxes(int axis1, int axis2) const {
      using SwapAxesOp = SwapAxes<T>;
      SwapAxesParam sp{.axis1 = axis1, .axis2 = axis2};
      return apply_unary_op<SwapAxesOp, SwapAxesParam>(
          sp, [](const Raw &x, const SwapAxesParam &p) {
             return x.swapaxes(p.axis1, p.axis2);
          });
   }

   friend std::ostream &operator<<(std::ostream &os, const ADTensor &t) {
      return os << t.raw();
   }

   T *begin() { return raw_.begin(); }
   T *end() { return raw_.end(); }
   T *begin() const { return raw_.begin(); }
   T *end() const { return raw_.end(); }

   template <typename OpTag, typename F> ADTensor unary_hook(F &&f) const {
      return apply_unary_op<OpTag>(std::forward<F>(f));
   }

   template <typename OpTag, typename Param, typename F>
   ADTensor unary_hook(const Param &p, F &&f) const {
      return apply_unary_op<OpTag, Param>(p, std::forward<F>(f));
   }

   template <typename OpTag, typename Meta, typename Param, typename F>
   ADTensor unary_meta_hook(Meta &m, const Param &p, F &&f) const {
      return apply_unary_meta_op<OpTag, Meta, Param>(m, p, std::forward<F>(f));
   }

 private:
   RawTensor<T> raw_;
   ValueID vid_{-1};
   GradSlotID grad_slot_id_{-1};
   bool requires_grad_;

   static bool grad_flow(const ADTensor &x, const ADTensor &y) {
      return x.requires_grad() || y.requires_grad();
   };

   template <typename OpTag, typename F>
   ADTensor apply_binary_op(const ADTensor &other, F &&f) const {
      using Op = Operation<T, OpTag>;
      const ADTensor &self = *this;
      F ff = std::forward<F>(f);
      return autodiff::binary<T, Op>(
          self, other, [&](const ADTensor &x, const ADTensor &y) {
             const Raw &xb = x.raw();
             const Raw &yb = y.raw();

             Raw out = ff(xb, yb);

             bool req_grad = grad_flow(x, y) && autodiff::grad_enabled();
             return ADTensor(std::move(out), req_grad);
          });
   }

   template <typename OpTag, typename F> ADTensor apply_unary_op(F &&f) const {
      using Op = Operation<T, OpTag>;
      const ADTensor &self = *this;
      F ff = std::forward<F>(f);
      return autodiff::unary<T, Op>(self, [&](const ADTensor &x) {
         const Raw &xb = x.raw();
         Raw out = ff(xb);
         bool req_grad = x.requires_grad() && autodiff::grad_enabled();
         return ADTensor(std::move(out), req_grad);
      });
   }

   template <typename OpTag, typename Param, typename F>
   ADTensor apply_unary_op(const Param &p, F &&f) const {
      using Op = Operation<T, OpTag>;
      const ADTensor &self = *this;
      F ff = std::forward<F>(f);
      return autodiff::unary<T, Op, Param>(
          self, p, [&](const ADTensor &x, const Param &param) {
             const Raw &xb = x.raw();
             Raw out = ff(xb, param);
             bool req_grad = x.requires_grad() && autodiff::grad_enabled();
             return ADTensor(std::move(out), req_grad);
          });
   }

   template <typename OpTag, typename Meta, typename Param, typename F>
   ADTensor apply_unary_meta_op(Meta &m, const Param &p, F &&f) const {
      using Op = Operation<T, OpTag>;
      const ADTensor &self = *this;
      F ff = std::forward<F>(f);
      return autodiff::unary_meta<T, Op, Meta, Param>(
          self, m, p,
          [&](const ADTensor &x, Meta &meta, const Param &param) {
             const Raw &xb = x.raw();
             Raw out = ff(xb, meta, param);
             bool req_grad = x.requires_grad();
             return ADTensor(std::move(out), req_grad);
          });
   }
};

#endif // AD_TENSOR_HPP

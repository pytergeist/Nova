#ifndef DISPATCH_HPP
#define DISPATCH_HPP

#include <memory>

#include "AutodiffContext.hpp"
#include "AutodiffMeta.hpp"
#include "Engine.hpp"

// TODO: find a way around using const_cast to remove cv qualifier

namespace autodiff {

template <class Op>
static constexpr bool require_ewise_binary_op() {
   return op_category_v<typename Op::tag> == OpCategory::EwiseBinary;
}

template <class Op>
static constexpr bool require_ewise_unary_op() {
   return op_category_v<typename Op::tag> == OpCategory::EwiseUnary;
}

template <class Op>
static constexpr bool require_reduction_op() {
   return op_category_v<typename Op::tag> == OpCategory::Reduction;
}

template <class Op>
static constexpr bool require_contraction_op() {
   return op_category_v<typename Op::tag> == OpCategory::Contraction;
}

template <class Op>
static constexpr bool require_movement_op() {
   return op_category_v<typename Op::tag> == OpCategory::Movement;
}

template <class Op>
consteval void assert_op_valid_for_unary_dispatch() {
   static_assert(require_ewise_unary_op<Op>() || require_reduction_op<Op>() || require_movement_op<Op>(),
      "Invalid operation in binary dispatch path");
}

template <class Op>
consteval void assert_op_valid_for_binary_dispatch() {
   static_assert(require_ewise_binary_op<Op>() || require_contraction_op<Op>(),
      "Invalid operation in binary dispatch path");
}


template <typename T>
AutodiffMeta<T> construct_meta(
    const ADTensor<T> &x,   // NOLINT(bugprone-easily-swappable-parameters)
    const ADTensor<T> &y) { // NOLINT(bugprone-easily-swappable-parameters)
   AutodiffMeta<T> meta;
   meta.push_back(x.raw());
   meta.push_back(y.raw());
   return meta;
}

template <typename T>
AutodiffMeta<T> construct_meta(const ADTensor<T> &x) {
   AutodiffMeta<T> meta;
   meta.push_back(x.raw());
   return meta;
}

template <typename T, typename Param>
AutodiffMeta<T> construct_meta(const ADTensor<T> &x,
                                      const Param &param) {
   AutodiffMeta<T> meta;
   meta.push_back(x.raw());
   meta.op_param = param;
   return meta;
}

template <typename T, class Op, typename Param, class EagerFn>
ADTensor<T> unary(const ADTensor<T> &x, const Param &params,
                         EagerFn &&eager) {
   assert_op_valid_for_unary_dispatch<Op>();

   EagerFn feager = std::forward<EagerFn>(eager);
   const bool needs_grad = grad_enabled() && x.requires_grad();
   if (!needs_grad || !should_trace(x)) {
      return feager(x, params);
   }
   Engine<T> &eng = AutodiffContext<T>::get();
   ValueID vx = const_cast<ADTensor<T> &>(x).ensure_vid();
   AutodiffMeta<T> meta = construct_meta<T>(x, params);
   std::vector<ValueID> vids{vx};
   ValueID out = eng.template apply_single<Op>(meta, vids);
   RawTensor<T> raw = eng.materialise(out);
   ADTensor<T> result(std::move(raw), x.requires_grad());
   result.set_vid(out);
   return result;
}

template <typename T, class Op, class EagerFn>
ADTensor<T> unary(const ADTensor<T> &x, EagerFn &&eager) {
   assert_op_valid_for_unary_dispatch<Op>();

   EagerFn feager = std::forward<EagerFn>(eager);
   const bool needs_grad = grad_enabled() && x.requires_grad();
   if (!needs_grad || !should_trace(x)) {
      return feager(x);
   }
   Engine<T> &eng = AutodiffContext<T>::get();
   ValueID vx =
       const_cast<ADTensor<T> &>(x)
           .ensure_vid(); // NOLINT(cppcoreguidelines-pro-type-const-cast)
   AutodiffMeta<T> meta = construct_meta<T>(x);
   std::vector<ValueID> vids{vx};
   ValueID out = eng.template apply_single<Op>(meta, vids);
   RawTensor<T> raw = eng.materialise(out);
   ADTensor<T> result(std::move(raw), needs_grad);
   result.set_vid(out);
   return result;
}

template <typename T, class Op, class EagerFn>
ADTensor<T> binary(const ADTensor<T> &x, const ADTensor<T> &y,
                          EagerFn &&eager) {
   assert_op_valid_for_binary_dispatch<Op>();

   EagerFn feager = std::forward<EagerFn>(eager);
   const bool needs_grad =
       grad_enabled() && (x.requires_grad() || y.requires_grad());
   if (!needs_grad || !should_trace(x, y)) {
      return feager(x, y);
   }
   Engine<T> &eng = AutodiffContext<T>::get();
   ValueID vx = const_cast<ADTensor<T> &>(x).ensure_vid();
   ValueID vy = const_cast<ADTensor<T> &>(y).ensure_vid();
   AutodiffMeta<T> meta = construct_meta<T>(x, y);
   std::vector<ValueID> vids{vx, vy};
   ValueID out = eng.template apply_single<Op>(meta, vids);
   RawTensor<T> raw = eng.materialise(out);
   ADTensor<T> result(std::move(raw), needs_grad);
   result.set_vid(out);
   return result;
}

} // namespace autodiff

#endif // DISPATCH_HPP

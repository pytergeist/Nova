#ifndef DISPATCH_HPP
#define DISPATCH_HPP

#include <memory>
#include <string>
#include <unordered_map>
#include <variant>

#include "AutodiffMeta.hpp"
#include "Engine.hpp"
#include "EngineContext.hpp"

// TODO: find a way around using const_cast to remove cv qualifier

namespace autodiff {

template <typename T>
inline AutodiffMeta<T> construct_meta(
    const ADTensor<T> &x,   // NOLINT(bugprone-easily-swappable-parameters)
    const ADTensor<T> &y) { // NOLINT(bugprone-easily-swappable-parameters)
   AutodiffMeta<T> meta;
   meta.push_back(x.raw());
   meta.push_back(y.raw());
   return meta;
}

template <typename T>
inline AutodiffMeta<T> construct_meta(const ADTensor<T> &x) {
   AutodiffMeta<T> meta;
   meta.push_back(x.raw());
   return meta;
}

template <typename T, typename Param>
inline AutodiffMeta<T> construct_meta(const ADTensor<T> &x,
                                      const Param &param) {
   AutodiffMeta<T> meta;
   meta.push_back(x.raw());
   meta.op_param = param;
   return meta;
}

template <typename T, class Op, typename Param, class EagerFn>
inline ADTensor<T> unary(const ADTensor<T> &x, const Param &params,
                         EagerFn &&eager) {
   EagerFn feager = std::forward<EagerFn>(eager);
   const bool needs_grad = grad_enabled() && x.requires_grad();
   if (!needs_grad || !should_trace(x)) {
      return feager(x, params);
   }
   Engine<T> &eng = EngineContext<T>::get();
   ValueID vx = const_cast<ADTensor<T> &>(x).ensure_vid();
   AutodiffMeta<T> meta = construct_meta<T>(x, params);
   std::vector<ValueID> vids{vx};
   ValueID out = eng.template apply<Op>(meta, vids);
   RawTensor<T> raw = eng.materialise(out);
   ADTensor<T> result(std::move(raw), x.requires_grad());
   result.set_vid(out);
   return result;
}

template <typename T, class Op, class EagerFn>
inline ADTensor<T> unary(const ADTensor<T> &x, EagerFn &&eager) {
   EagerFn feager = std::forward<EagerFn>(eager);
   const bool needs_grad = grad_enabled() && x.requires_grad();
   if (!needs_grad || !should_trace(x)) {
      return feager(x);
   }
   Engine<T> &eng = EngineContext<T>::get();
   ValueID vx =
       const_cast<ADTensor<T> &>(x)
           .ensure_vid(); // NOLINT(cppcoreguidelines-pro-type-const-cast)
   AutodiffMeta<T> meta = construct_meta<T>(x);
   std::vector<ValueID> vids{vx};
   ValueID out = eng.template apply<Op>(meta, vids);
   RawTensor<T> raw = eng.materialise(out);
   ADTensor<T> result(std::move(raw), needs_grad);
   result.set_vid(out);
   return result;
}

template <typename T, class Op, class EagerFn>
inline ADTensor<T> binary(const ADTensor<T> &x, const ADTensor<T> &y,
                          EagerFn &&eager) {
   EagerFn feager = std::forward<EagerFn>(eager);
   const bool needs_grad =
       grad_enabled() && (x.requires_grad() || y.requires_grad());
   if (!needs_grad || !should_trace(x, y)) {
      return feager(x, y);
   }
   Engine<T> &eng = EngineContext<T>::get();
   ValueID vx = const_cast<ADTensor<T> &>(x).ensure_vid();
   ValueID vy = const_cast<ADTensor<T> &>(y).ensure_vid();
   AutodiffMeta<T> meta = construct_meta<T>(x, y);
   std::vector<ValueID> vids{vx, vy};
   ValueID out = eng.template apply<Op>(meta, vids);
   RawTensor<T> raw = eng.materialise(out);
   ADTensor<T> result(std::move(raw), needs_grad);
   result.set_vid(out);
   return result;
}

} // namespace autodiff

#endif // DISPATCH_HPP

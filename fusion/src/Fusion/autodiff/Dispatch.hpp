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
   meta.push_back(x);
   meta.push_back(y);
   return meta;
}

template <typename T>
inline AutodiffMeta<T> construct_meta(const ADTensor<T> &x) {
   AutodiffMeta<T> meta;
   meta.push_back(x);
   return meta;
}

template <typename T, typename Param>
inline AutodiffMeta<T> construct_meta(const ADTensor<T> &x,
                                      const Param &param) {
   AutodiffMeta<T> meta;
   meta.push_back(x);
   meta.op_param = param;
   return meta;
}

template <typename T, class Op, typename Param, class EagerFn>
inline ADTensor<T> unary(const ADTensor<T> &x, const Param &params,
                         EagerFn &&eager) {
   EagerFn feager = std::forward<EagerFn>(eager);
   if (!should_trace(x)) {
      return eager(x, params);
   };
   if (!grad_enabled() || !x.requires_grad()) {
      return eager(x, params);
   }
   Engine<T> &eng = EngineContext<T>::get();
   ValueID vx = const_cast<ADTensor<T> &>(x).ensure_vid();
   AutodiffMeta<T> meta = construct_meta<T>(x, params);
   ValueID out = eng.template apply<Op>(meta);
   return eng.materialise(out);
}

template <typename T, class Op, class EagerFn>
inline ADTensor<T> unary(const ADTensor<T> &x, EagerFn &&eager) {
   EagerFn feager = std::forward<EagerFn>(eager);
   if (!grad_enabled() || !x.requires_grad()) {
      return feager(x);
   }
   if (!should_trace(x)) {
      return feager(x);
   }
   Engine<T> &eng = EngineContext<T>::get();
   ValueID vx =
       const_cast<ADTensor<T> &>(x)
           .ensure_vid(); // NOLINT(cppcoreguidelines-pro-type-const-cast)
   AutodiffMeta<T> meta = construct_meta<T>(x);
   ValueID out = eng.template apply<Op>(meta);
   return eng.materialise(out);
}

template <typename T, class Op, class EagerFn>
inline ADTensor<T> binary(const ADTensor<T> &x, const ADTensor<T> &y,
                          EagerFn &&eager) {
   EagerFn feager = std::forward<EagerFn>(eager);
   if (!grad_enabled() || (!x.requires_grad() && !y.requires_grad())) {
      return feager(x, y);
   }
   if (!should_trace(x, y)) {
      return feager(x, y);
   }
   Engine<T> &eng = EngineContext<T>::get();
   ValueID vx = const_cast<ADTensor<T> &>(x).ensure_vid();
   ValueID vy = const_cast<ADTensor<T> &>(y).ensure_vid();
   AutodiffMeta<T> meta = construct_meta<T>(x, y);
   ValueID out = eng.template apply<Op>(meta);
   return eng.materialise(out);
}

} // namespace autodiff

#endif // DISPATCH_HPP

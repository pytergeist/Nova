#ifndef DISPATCH_H
#define DISPATCH_H

#include <memory>
#include <string>
#include <variant>
#include <unordered_map>

#include "Engine.h"
#include "EngineContext.h"


namespace autodiff {

template <typename T>
inline AutodiffMeta<T> construct_meta(const Tensor<T>& x, const Tensor<T>& y) {
  AutodiffMeta<T> meta;
  meta.push_back(x);
  meta.push_back(y);
  return meta;
}


template <typename T>
inline AutodiffMeta<T> construct_meta(const Tensor<T>& x) {
  AutodiffMeta<T> meta;
  meta.push_back(x);
  return meta;
}


template <typename T, typename Param> // TODO: make this generic for params
inline AutodiffMeta<T> construct_meta(const Tensor<T>& x, Param& param) {
  AutodiffMeta<T> meta;
  meta.push_back(x);
  meta.template set_param<int>("axis1", param[0]);
  meta.template set_param<int>("axis2", param[1]);
  return meta;
}


template <typename T, class Op, typename Param, class EagerFn>
inline Tensor<T> unary(const Tensor<T>& x, Param& params, EagerFn&& eager) {
    if (!should_trace(x)) {
      int a = params[0];
      int b = params[1];
      return eager(x, a, b);};
    if (!grad_enabled() || !x.requires_grad()) {
      int a = params[0];
      int b = params[1];
      return eager(x, a, b);}
    auto& eng = EngineContext<T>::get();
    auto vx = const_cast<Tensor<T>&>(x).ensure_vid();
    AutodiffMeta<T> meta = construct_meta<T>(x, params);
    int a = meta.template get_param<int>("axis1");
    int b = meta.template get_param<int>("axis2");

    ValueID out = eng.template apply<Op>(meta);
    return eng.materialise(out);

}



template <typename T, class Op, class EagerFn>
inline Tensor<T> unary(const Tensor<T>& x, EagerFn&& eager) {
    if (!grad_enabled() || !x.requires_grad()) {return eager(x);}
    if (!should_trace(x)) return eager(x);
    auto& eng = EngineContext<T>::get();
    auto vx = const_cast<Tensor<T>&>(x).ensure_vid();
    AutodiffMeta<T> meta = construct_meta<T>(x);
    ValueID out = eng.template apply<Op>(meta);
    return eng.materialise(out);

}

template <typename T, class Op, class EagerFn>
inline Tensor<T> binary(const Tensor<T>& x, const Tensor<T>& y, EagerFn&& eager) {
    if (!grad_enabled() || (!x.requires_grad() && !y.requires_grad())) {return eager(x, y);}
    if (!should_trace(x, y)) return eager(x, y);
    auto& eng = EngineContext<T>::get();
    auto vx = const_cast<Tensor<T>&>(x).ensure_vid();
    auto vy = const_cast<Tensor<T>&>(y).ensure_vid();
    AutodiffMeta<T> meta = construct_meta<T>(x, y);
    ValueID out = eng.template apply<Op>(meta);
    return eng.materialise(out);

}
}

#endif // DISPATCH_H

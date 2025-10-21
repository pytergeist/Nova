#ifndef DISPATCH_H
#define DISPATCH_H

#include <memory>

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



template <typename T, class Op, class EagerFn>
inline Tensor<T> unary(const Tensor<T>& x, EagerFn&& eager) {
    if (!grad_enabled() || !x.requires_grad()) {return eager(x);}
    auto& eng = EngineContext<T>::get();
    auto vx = const_cast<Tensor<T>&>(x).ensure_vid();
    AutodiffMeta<T> meta = construct_meta<T>(x);
    ValueID out = eng.template apply<Op>(meta);
    return eng.materialise(out);

}

template <typename T, class Op, class EagerFn>
inline Tensor<T> binary(const Tensor<T>& x, const Tensor<T>& y, EagerFn&& eager) {
    if (!grad_enabled() || (!x.requires_grad() && !y.requires_grad())) {return eager(x, y);}
    auto& eng = EngineContext<T>::get();
    auto vx = const_cast<Tensor<T>&>(x).ensure_vid();
    auto vy = const_cast<Tensor<T>&>(y).ensure_vid();
    AutodiffMeta<T> meta = construct_meta<T>(x, y);
    ValueID out = eng.template apply<Op>(meta);
    return eng.materialise(out);

}
}

#endif // DISPATCH_H

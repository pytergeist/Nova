#ifndef OPS_TRANSCENENTAL_HPP
#define OPS_TRANSCENENTAL_HPP

#include <string_view>
#include <vector>

#include "Fusion/core/ElementWise.hpp"
#include "Fusion/core/TensorBase.hpp"

namespace fusion {

namespace math {

template <typename T> inline TensorBase<T> sqrt(const TensorBase<T> &x) {
   UnaryEwiseMeta meta = make_binary_meta(x);
   TensorBase<T> out = init_out_from_meta(x, meta);
   ewise::unary_ewise_tag<T, SqrtSIMD>(x, meta, out);
   return out;
}

template <typename T> inline TensorBase<T> log(const TensorBase<T> &x) {
   UnaryEwiseMeta meta = make_binary_meta(x);
   TensorBase<T> out = init_out_from_meta(x, meta);
   ewise::unary_ewise_tag<T, NaturalLogSIMD>(x, meta, out);
   return out;
}

template <typename T> inline TensorBase<T> exp(const TensorBase<T> &x) {
   UnaryEwiseMeta meta = make_binary_meta(x);
   TensorBase<T> out = init_out_from_meta(x, meta);
   ewise::unary_ewise_tag<T, ExponentialSIMD>(x, meta, out);
   return out;
}

} // namespace math

} // namespace fusion

#endif // OPS_TRANSCENENTAL_HPP

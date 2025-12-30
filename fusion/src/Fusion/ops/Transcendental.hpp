#ifndef OPS_TRANSCENENTAL_HPP
#define OPS_TRANSCENENTAL_HPP

#include <string_view>
#include <vector>

#include "Fusion/core/EwiseIter.hpp"
#include "Fusion/core/RawTensor.hpp"

namespace fusion {

namespace math {

template <typename T> inline RawTensor<T> sqrt(const RawTensor<T> &x) {
   UnaryEwiseMeta meta = make_binary_meta(x);
   RawTensor<T> out = init_out_from_meta(x, meta);
   ewise::unary_ewise_tag<T, SqrtSIMD>(x, meta, out);
   return out;
}

template <typename T> inline RawTensor<T> log(const RawTensor<T> &x) {
   UnaryEwiseMeta meta = make_binary_meta(x);
   RawTensor<T> out = init_out_from_meta(x, meta);
   ewise::unary_ewise_tag<T, NaturalLogSIMD>(x, meta, out);
   return out;
}

template <typename T> inline RawTensor<T> exp(const RawTensor<T> &x) {
   UnaryEwiseMeta meta = make_binary_meta(x);
   RawTensor<T> out = init_out_from_meta(x, meta);
   ewise::unary_ewise_tag<T, ExponentialSIMD>(x, meta, out);
   return out;
}

} // namespace math

} // namespace fusion

#endif // OPS_TRANSCENENTAL_HPP

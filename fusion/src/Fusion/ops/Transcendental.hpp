#ifndef OPS_TRANSCENENTAL_HPP
#define OPS_TRANSCENENTAL_HPP

#include <string_view>
#include <vector>

#include "Fusion/core/RawTensor.hpp"
#include "Fusion/core/TensorIter.hpp"

namespace fusion {

namespace math {

template <typename T> RawTensor<T> sqrt(const RawTensor<T> &x) {
   require_ewise_unary_out_of_place<SqrtTag>();
   UnaryEwiseMeta meta = make_unary_meta(x);
   RawTensor<T> out = init_out_from_meta(x, meta);
   fusion::iter::unary_ewise_tag<T, SqrtSIMD>(x, meta, out);
   return out;
}

template <typename T> RawTensor<T> log(const RawTensor<T> &x) {
   require_ewise_unary_out_of_place<LogTag>();
   UnaryEwiseMeta meta = make_unary_meta(x);
   RawTensor<T> out = init_out_from_meta(x, meta);
   fusion::iter::unary_ewise_tag<T, NaturalLogSIMD>(x, meta, out);
   return out;
}

template <typename T> RawTensor<T> exp(const RawTensor<T> &x) {
   require_ewise_unary_out_of_place<ExpTag>();
   UnaryEwiseMeta meta = make_unary_meta(x);
   RawTensor<T> out = init_out_from_meta(x, meta);
   fusion::iter::unary_ewise_tag<T, ExponentialSIMD>(x, meta, out);
   return out;
}

} // namespace math

} // namespace fusion

#endif // OPS_TRANSCENENTAL_HPP
